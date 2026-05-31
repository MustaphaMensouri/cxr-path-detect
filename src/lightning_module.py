import torch
from torchmetrics import AUROC, Recall, Precision, F1Score, AveragePrecision, MetricCollection
import lightning as L
from omegaconf import OmegaConf
from src.factories import build_backbone, build_loss
import torch.distributed as dist

class XrayClassifier(L.LightningModule):
    def __init__(self, cfg, num_classes, max_epochs, class_names=None):
        super().__init__()

        cfg_to_save = (
            OmegaConf.to_container(cfg, resolve=True)
            if not isinstance(cfg, dict)
            else cfg
        )

        self.save_hyperparameters({
            "cfg": cfg_to_save,
            "num_classes": num_classes,
            "max_epochs": max_epochs,
            "class_names": class_names,
        })
        self.cfg = OmegaConf.create(cfg_to_save)
        self.max_epochs = max_epochs
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]

        # ── backbone ──────────────────────────────────────────────────────────
        self.model = build_backbone(cfg.model, num_classes)
        self.loss = build_loss(cfg.loss)

        # ── metrics ───────────────────────────────────────────────────────────
        # average="macro" aggregates across classes internally inside torchmetrics,
        # so .compute() returns a scalar — safe for DDP sync via self.log()

        def _global_metrics():
            return MetricCollection({
                "auc_macro": AUROC(task="multilabel", num_labels=num_classes, average="macro", sync_on_compute=True),
                "auc_micro": AUROC(task="multilabel", num_labels=num_classes, average="micro", sync_on_compute=True),
                "f1_macro": F1Score(task="multilabel", num_labels=num_classes, average="macro", sync_on_compute=True),
                "f1_micro": F1Score(task="multilabel", num_labels=num_classes, average="micro", sync_on_compute=True),
                "precision_macro": Precision(task="multilabel", num_labels=num_classes, average="macro", sync_on_compute=True),
                "recall_macro": Recall(task="multilabel", num_labels=num_classes, average="macro", sync_on_compute=True),
                "ap_macro": AveragePrecision(task="multilabel", num_labels=num_classes, average="macro", sync_on_compute=True),
                "ap_micro": AveragePrecision(task="multilabel", num_labels=num_classes, average="micro", sync_on_compute=True),
            })
        self.val_metrics = _global_metrics()
        self.test_metrics = _global_metrics()
        # thresholded tuning metrics 
        self.val_probs = []
        self.val_targets = []
        self.register_buffer(
            "best_thresholds",
            torch.full((num_classes,), 0.5),
            persistent=True,
        )

        self.test_probs = []
        self.test_targets = []

        # ── separate per-class metrics only for val/test (cheaper) ────────────

        def _per_class_metrics():
            return MetricCollection({
                "auc": AUROC(task="multilabel", num_labels=num_classes, average="none", sync_on_compute=True),
                "f1": F1Score(task="multilabel", num_labels=num_classes, average="none", sync_on_compute=True),
                "precision": Precision(task="multilabel", num_labels=num_classes, average="none", sync_on_compute=True),
                "recall": Recall(task="multilabel", num_labels=num_classes, average="none", sync_on_compute=True),
                "ap": AveragePrecision(task="multilabel", num_labels=num_classes, average="none", sync_on_compute=True),
            })
        self.test_per_class_metrics = _per_class_metrics()

    def forward(self, x):
        return self.model(x)

    # ── shared step ───────────────────────────────────────────────────────────
    def _step(self, batch, stage: str):
        x, y   = batch
        logits = self(x)
        loss   = self.loss(logits, y)
        probs  = torch.sigmoid(logits)
        y_int  = y.int()

        self.log(
        f"{stage}/loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        if stage == "train":
            return loss
        
        # val/test: global metrics
        global_metrics = getattr(self, f"{stage}_metrics")
        global_metrics.update(probs, y_int)
        
        if stage == "val":
            self.val_probs.append(probs.detach().cpu())
            self.val_targets.append(y_int.detach().cpu())

        self.log_dict(
            {f"{stage}/{k}": metric for k, metric in global_metrics.items()},
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        # test only: per-class metrics
        if stage == "test":
            self.test_probs.append(probs.detach().cpu())
            self.test_targets.append(y_int.detach().cpu())
            self.test_per_class_metrics.update(probs, y_int)

        return loss
    
    def _gather_from_all_ranks(self, local_tensor):
        if not (dist.is_available() and dist.is_initialized()):
            return local_tensor

        gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, local_tensor)

        return torch.cat(gathered, dim=0)

    # ── per-class logging at epoch end (val/test only) ────────────────────────
    def _log_per_class(self, stage: str):
        per_class = getattr(self, f"{stage}_per_class_metrics")

        # .compute() is safe here ONLY because we first barrier-sync by letting
        # Lightning already have called all_gather via the log_dict above.
        # But to be fully safe with DDP we use self.log() per scalar value.
        results = per_class.compute()  # dict of {metric_name: tensor[num_classes]}
        per_class.reset()

        for metric_name, values in results.items():
            for i, val in enumerate(values):
                self.log(
                    f"{stage}/{metric_name}/{self.class_names[i]}",
                    val,
                    sync_dist=True,
                )

    def tune_thresholds(self, probs, targets):
        candidates = torch.arange(0.01, 1.00, 0.01)

        num_classes = probs.shape[1]
        best_thresholds = torch.full((num_classes,), 0.5)
        best_f1s = torch.zeros(num_classes)

        eps = 1e-8

        for c in range(num_classes):
            p = probs[:, c]
            y = targets[:, c]

            best_f1 = 0.0
            best_t = 0.5

            for t in candidates:
                pred = (p >= t).int()

                tp = ((pred == 1) & (y == 1)).sum().float()
                fp = ((pred == 1) & (y == 0)).sum().float()
                fn = ((pred == 0) & (y == 1)).sum().float()

                precision = tp / (tp + fp + eps)
                recall = tp / (tp + fn + eps)
                f1 = 2 * precision * recall / (precision + recall + eps)

                if f1 > best_f1:
                    best_f1 = f1
                    best_t = float(t)

            best_thresholds[c] = best_t
            best_f1s[c] = best_f1

        return best_thresholds, best_f1s


    def compute_tuned_metrics(self, probs, targets, thresholds):
        preds = (probs >= thresholds.view(1, -1)).int()
        eps = 1e-8

        tp = ((preds == 1) & (targets == 1)).sum(dim=0).float()
        fp = ((preds == 1) & (targets == 0)).sum(dim=0).float()
        fn = ((preds == 0) & (targets == 1)).sum(dim=0).float()

        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)

        tp_micro = tp.sum()
        fp_micro = fp.sum()
        fn_micro = fn.sum()

        precision_micro = tp_micro / (tp_micro + fp_micro + eps)
        recall_micro = tp_micro / (tp_micro + fn_micro + eps)
        f1_micro = 2 * precision_micro * recall_micro / (
            precision_micro + recall_micro + eps
        )

        return {
            "f1_macro": f1.mean(),
            "f1_micro": f1_micro,
            "precision_macro": precision.mean(),
            "recall_macro": recall.mean(),
        }


    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            self.val_probs.clear()
            self.val_targets.clear()
            return

        if len(self.val_probs) == 0:
            return

        local_probs = torch.cat(self.val_probs, dim=0)
        local_targets = torch.cat(self.val_targets, dim=0)

        probs = self._gather_from_all_ranks(local_probs)
        targets = self._gather_from_all_ranks(local_targets)
        
        probs = probs.to(self.device)
        targets = targets.to(self.device)
        
        if self.global_rank == 0:
            thresholds, _ = self.tune_thresholds(probs, targets)
        else:
            thresholds = torch.full_like(self.best_thresholds.cpu(), 0.5)

        if dist.is_available() and dist.is_initialized():
            thresholds = thresholds.to(self.device)
            dist.broadcast(thresholds, src=0)
        else:
            thresholds = thresholds.to(self.device)

        self.best_thresholds.copy_(thresholds)

        metrics = self.compute_tuned_metrics(
            probs,
            targets,
            thresholds,
        )
        if self.global_rank == 0:
            print("[Thresholds]", self.best_thresholds[:10])

        self.log("val/f1_macro_tuned", metrics["f1_macro"], prog_bar=True, sync_dist=True)
        self.log("val/f1_micro_tuned", metrics["f1_micro"], sync_dist=True)
        self.log("val/precision_macro_tuned", metrics["precision_macro"], sync_dist=True)
        self.log("val/recall_macro_tuned", metrics["recall_macro"], sync_dist=True)

        self.val_probs.clear()
        self.val_targets.clear()
    def on_test_epoch_end(self):
        self._log_per_class("test")

        if len(self.test_probs) == 0:
            return

        local_probs = torch.cat(self.test_probs, dim=0)
        local_targets = torch.cat(self.test_targets, dim=0)

        probs = self._gather_from_all_ranks(local_probs)
        targets = self._gather_from_all_ranks(local_targets)

        probs = probs.to(self.device)
        targets = targets.to(self.device)
        thresholds = self.best_thresholds.detach().to(self.device)

        metrics = self.compute_tuned_metrics(probs, targets, thresholds)
        if self.global_rank == 0:
            print("[Thresholds]", self.best_thresholds[:10])
        self.log("test/f1_macro_tuned", metrics["f1_macro"], prog_bar=True, sync_dist=True)
        self.log("test/f1_micro_tuned", metrics["f1_micro"], sync_dist=True)
        self.log("test/precision_macro_tuned", metrics["precision_macro"], sync_dist=True)
        self.log("test/recall_macro_tuned", metrics["recall_macro"], sync_dist=True)

        preds = (probs >= thresholds.view(1, -1)).int()

        tp = ((preds == 1) & (targets == 1)).sum(dim=0).float()
        fp = ((preds == 1) & (targets == 0)).sum(dim=0).float()
        fn = ((preds == 0) & (targets == 1)).sum(dim=0).float()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        for i, label in enumerate(self.class_names):
            self.log(f"test/f1_tuned/{label}", f1[i], sync_dist=False, rank_zero_only=True,)
            self.log(f"test/precision_tuned/{label}", precision[i], sync_dist=False, rank_zero_only=True,)
            self.log(f"test/recall_tuned/{label}", recall[i], sync_dist=False, rank_zero_only=True,)
        for i, label in enumerate(self.class_names):
            self.log(
                f"threshold/{label}",
                thresholds[i].detach(),
                rank_zero_only=True,
            )
        self.test_probs.clear()
        self.test_targets.clear()

    def training_step(self, batch, _):   return self._step(batch, "train")
    def validation_step(self, batch, _): return self._step(batch, "val")
    def test_step(self, batch, _):       return self._step(batch, "test")

    # ── optimiser ─────────────────────────────────────────────────────────────
    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.cfg.model.lr, weight_decay=self.cfg.model.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.max_epochs)
        return {"optimizer": opt, "lr_scheduler": scheduler}