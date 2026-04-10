import torch
import torch.nn as nn
from torchvision import models
from torchmetrics import AUROC, Accuracy, Recall, F1Score, MetricCollection
import lightning as L


class WeightedBCELoss(nn.Module):
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        P = targets.sum(dim=0).clamp(min=1)
        N = (1 - targets).sum(dim=0).clamp(min=1)
        total = P + N

        beta_p = (total / P).clamp(max=10.0)
        beta_n = (total / N).clamp(max=10.0)

        probs = torch.sigmoid(logits).clamp(1e-7, 1 - 1e-7)

        pos_loss = -beta_p * targets       * torch.log(probs)
        neg_loss = -beta_n * (1 - targets) * torch.log(1 - probs)

        return (pos_loss + neg_loss).mean()


class XrayClassifier(L.LightningModule):
    def __init__(self, cfg, num_classes, max_epochs, class_names=None):
        super().__init__()
        self.save_hyperparameters(ignore=["cfg"])
        self.cfg = cfg
        self.max_epochs = max_epochs
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]

        # ── backbone ──────────────────────────────────────────────────────────
        backbone = getattr(models, cfg.backbone)(weights="DEFAULT" if cfg.pretrained else None)
        backbone.classifier = nn.Linear(backbone.classifier.in_features, num_classes)
        self.model = backbone

        self.loss = WeightedBCELoss()

        # ── metrics ───────────────────────────────────────────────────────────
        # average="macro" aggregates across classes internally inside torchmetrics,
        # so .compute() returns a scalar — safe for DDP sync via self.log()
        metric_kwargs = dict(task="multilabel", num_labels=num_classes, average="macro")

        def _metrics():
            return MetricCollection({
                "auc":      AUROC    (**metric_kwargs),
                "accuracy": Accuracy (**metric_kwargs),
                "recall":   Recall   (**metric_kwargs),
                "f1":       F1Score  (**metric_kwargs),
            })

        self.train_metrics = _metrics()
        self.val_metrics   = _metrics()
        self.test_metrics  = _metrics()

        # ── separate per-class metrics only for val/test (cheaper) ────────────
        per_class_kwargs = dict(task="multilabel", num_labels=num_classes, average="none")

        def _per_class_metrics():
            return MetricCollection({
                "auc":      AUROC    (**per_class_kwargs),
                "f1":       F1Score  (**per_class_kwargs),
            })

        self.val_per_class_metrics  = _per_class_metrics()
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

        # macro metrics — scalar output, DDP-safe via self.log_dict
        macro_metrics = getattr(self, f"{stage}_metrics")
        macro_metrics.update(probs, y_int)

        self.log(f"{stage}/loss", loss,
                 prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log_dict(
            {f"{stage}/{k}": macro_metrics[k] for k in macro_metrics},
            prog_bar=True, on_step=False, on_epoch=True, sync_dist=True,
        )

        # per-class metrics only for val/test
        if stage in ("val", "test"):
            per_class = getattr(self, f"{stage}_per_class_metrics")
            per_class.update(probs, y_int)

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

    def on_validation_epoch_end(self): self._log_per_class("val")
    def on_test_epoch_end(self):       self._log_per_class("test")

    def training_step(self, batch, _):   return self._step(batch, "train")
    def validation_step(self, batch, _): return self._step(batch, "val")
    def test_step(self, batch, _):       return self._step(batch, "test")

    # ── optimiser ─────────────────────────────────────────────────────────────
    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.max_epochs)
        return {"optimizer": opt, "lr_scheduler": scheduler}