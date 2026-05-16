import torch
from torchmetrics import AUROC, Recall, Precision, F1Score, AveragePrecision, MetricCollection
import lightning as L
from src.factories import build_backbone, build_loss

class XrayClassifier(L.LightningModule):
    def __init__(self, cfg, num_classes, max_epochs, class_names=None):
        super().__init__()
        self.save_hyperparameters(ignore=["class_names"])
        self.cfg = cfg
        self.max_epochs = max_epochs
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]

        # ── backbone ──────────────────────────────────────────────────────────
        self.model = build_backbone(cfg.model, num_classes)
        self.loss = build_loss({"name": "focal"}) + build_loss({"name": "weighted_bce"})

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

        self.log_dict(
            {f"{stage}/{k}": metric for k, metric in global_metrics.items()},
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        # test only: per-class metrics
        if stage == "test":
            self.test_per_class_metrics.update(probs, y_int)

        return loss

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

    def on_test_epoch_end(self):       self._log_per_class("test")

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