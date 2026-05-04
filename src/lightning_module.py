import torch
import torch.nn as nn
from torchvision import models
from torchmetrics import AUROC, Precision, Recall, F1Score, MetricCollection
import lightning as L


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 1.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits).clamp(1e-7, 1 - 1e-7)
        bce   = -(targets * torch.log(probs) + (1 - targets) * torch.log(1 - probs))
        pt    = targets * probs + (1 - targets) * (1 - probs)
        focal = self.alpha * ((1 - pt) ** self.gamma) * bce
        return focal.mean()


class XrayClassifier(L.LightningModule):
    def __init__(self, cfg, num_classes: int, max_epochs: int, class_names: list[str] = None):
        super().__init__()
        self.save_hyperparameters(ignore=["cfg"])
        self.cfg         = cfg
        self.max_epochs  = max_epochs
        self.num_classes = num_classes
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]

        backbone = getattr(models, cfg.backbone)(
            weights="DEFAULT" if cfg.pretrained else None
        )
        backbone.classifier = nn.Linear(backbone.classifier.in_features, num_classes)
        self.model = backbone

        self.loss_fn = FocalLoss(
            gamma=getattr(cfg, "focal_gamma", 2.0),
            alpha=getattr(cfg, "focal_alpha", 1.0),
        )

        macro_kwargs = dict(task="multilabel", num_labels=num_classes, average="macro")

        def _macro_metrics():
            return MetricCollection({
                "auc":       AUROC     (**macro_kwargs),
                "precision": Precision (**macro_kwargs),
                "recall":    Recall    (**macro_kwargs),
                "f1":        F1Score   (**macro_kwargs),
            })

        # prefix= makes keys like "train/auc" automatically
        self.train_metrics = _macro_metrics().clone(prefix="train/")
        self.val_metrics   = _macro_metrics().clone(prefix="val/")
        self.test_metrics  = _macro_metrics().clone(prefix="test/")

        per_class_kwargs = dict(task="multilabel", num_labels=num_classes, average="none")

        def _per_class_metrics():
            return MetricCollection({
                "auc": AUROC   (**per_class_kwargs),
                "f1":  F1Score (**per_class_kwargs),
            })

        self.val_per_class_metrics  = _per_class_metrics()
        self.test_per_class_metrics = _per_class_metrics()

        self.register_buffer("thresholds", torch.full((num_classes,), 0.5))

    def forward(self, x):
        return self.model(x)

    def _step(self, batch, stage: str):
        x, y   = batch
        logits = self(x)
        loss   = self.loss_fn(logits, y)
        probs  = torch.sigmoid(logits)
        y_int  = y.int()

        # for test, binarise with tuned thresholds for P/R/F1
        # AUC still uses raw probabilities regardless
        if stage == "test":
            preds = (probs >= self.thresholds.to(probs.device)).int()
            self.test_metrics["auc"].update(probs, y_int)
            self.test_metrics["precision"].update(preds, y_int)
            self.test_metrics["recall"].update(preds, y_int)
            self.test_metrics["f1"].update(preds, y_int)
            self.test_per_class_metrics["auc"].update(probs, y_int)
            self.test_per_class_metrics["f1"].update(preds, y_int)
        else:
            macro = getattr(self, f"{stage}_metrics")
            macro.update(probs, y_int)
            if stage == "val":
                self.val_per_class_metrics.update(probs, y_int)

        self.log(f"{stage}/loss", loss,
                 prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    # ── epoch-end logging ─────────────────────────────────────────────────────
    def _log_macro_and_per_class(self, stage: str):
        macro = getattr(self, f"{stage}_metrics")

        # compute() + log scalars, then reset
        macro_results = macro.compute()          # keys already have prefix e.g. "val/auc"
        self.log_dict(macro_results, prog_bar=True, sync_dist=True)
        macro.reset()

        # per-class (val/test only)
        if stage in ("val", "test"):
            per_class = getattr(self, f"{stage}_per_class_metrics")
            pc_results = per_class.compute()     # {"auc": [C], "f1": [C]}
            per_class.reset()

            for metric_name, values in pc_results.items():
                for i, val in enumerate(values):
                    self.log(
                        f"{stage}/{metric_name}/{self.class_names[i]}",
                        val,
                        sync_dist=True,   # torchmetrics already synced state across GPUs
                    )

    def on_train_epoch_end(self):      self._log_macro_and_per_class("train")
    def on_validation_epoch_end(self): self._log_macro_and_per_class("val")
    def on_test_epoch_end(self):       self._log_macro_and_per_class("test")

    def training_step(self, batch, _):   return self._step(batch, "train")
    def validation_step(self, batch, _): return self._step(batch, "val")
    def test_step(self, batch, _):       return self._step(batch, "test")

    def set_thresholds(self, t):
        self.thresholds.copy_(torch.tensor(t, dtype=torch.float32))

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.max_epochs
        )
        return {"optimizer": opt, "lr_scheduler": scheduler}