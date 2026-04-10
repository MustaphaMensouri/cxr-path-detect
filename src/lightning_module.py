import torch
import torch.nn as nn
from torchvision import models
from torchmetrics import AUROC, Accuracy, Recall, F1Score
import lightning as L


class WeightedBCELoss(nn.Module):
    """
    Weighted Cross-Entropy Loss from ChestX-ray8 paper (Wang et al., 2017).
    
    For each class c in a batch:
        L = βP * Σ(yc=1) [-log f(xc)] + βN * Σ(yc=0) [-log(1 - f(xc))]
    
    where:
        βP = (|P| + |N|) / |P|
        βN = (|P| + |N|) / |N|
    and |P|, |N| are the number of 1s and 0s in the batch labels.
    """
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits, targets: [B, C]  (raw scores, float labels 0/1)
        
        P = targets.sum(dim=0).clamp(min=1)          # positive count per class [C]
        N = (1 - targets).sum(dim=0).clamp(min=1)    # negative count per class [C]
        total = P + N                                 # = batch_size

        beta_p = total / P   # [C]
        beta_n = total / N   # [C]

        # Numerically stable BCE
        probs = torch.sigmoid(logits)
        probs = probs.clamp(1e-7, 1 - 1e-7)

        pos_loss = -beta_p * targets       * torch.log(probs)
        neg_loss = -beta_n * (1 - targets) * torch.log(1 - probs)

        return (pos_loss + neg_loss).mean()


class XrayClassifier(L.LightningModule):
    def __init__(self, cfg, num_classes, max_epochs, class_names=None):
        super().__init__()
        self.save_hyperparameters(ignore=["cfg"])
        self.cfg = cfg
        self.max_epochs = max_epochs
        self.class_names = class_names

        backbone    = getattr(models, cfg.backbone)(weights="DEFAULT" if cfg.pretrained else None)
        backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)
        self.model  = backbone

        self.loss = WeightedBCELoss()
        metric_kwargs = dict(task="multilabel", num_labels=num_classes, average="none")

        self.train_auc      = AUROC    (**metric_kwargs)
        self.train_accuracy = Accuracy (**metric_kwargs)
        self.train_recall   = Recall   (**metric_kwargs)
        self.train_f1       = F1Score  (**metric_kwargs)

        self.val_auc        = AUROC    (**metric_kwargs)
        self.val_accuracy   = Accuracy (**metric_kwargs)
        self.val_recall     = Recall   (**metric_kwargs)
        self.val_f1         = F1Score  (**metric_kwargs)

        self.test_auc       = AUROC    (**metric_kwargs)
        self.test_accuracy  = Accuracy (**metric_kwargs)
        self.test_recall    = Recall   (**metric_kwargs)
        self.test_f1        = F1Score  (**metric_kwargs)

    def forward(self, x):
        return self.model(x)
    
    def _get_metrics(self, stage):
        return {
            "auc":      getattr(self, f"{stage}_auc"),
            "accuracy": getattr(self, f"{stage}_accuracy"),
            "recall":   getattr(self, f"{stage}_recall"),
            "f1":       getattr(self, f"{stage}_f1"),
        }

    def _step(self, batch, stage):
        x, y   = batch
        logits = self(x)
        loss   = self.loss(logits, y)
        probs  = torch.sigmoid(logits)
        y_int  = y.int()

        metrics = self._get_metrics(stage)

        # Update all per-class metrics
        metrics["auc"].update(probs, y_int)
        metrics["accuracy"].update(probs, y_int)
        metrics["recall"].update(probs, y_int)
        metrics["f1"].update(probs, y_int)

        self.log(f"{stage}/loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return loss
    
    def _log_per_class_metrics(self, stage):
        metrics = self._get_metrics(stage)
        names   = self.class_names  # may be None

        for metric_name, metric_obj in metrics.items():
            per_class_values = metric_obj.compute()    # shape: [num_classes]
            metric_obj.reset()

            for i, val in enumerate(per_class_values):
                label = names[i] if names else f"class_{i}"
                self.log(f"{stage}/{metric_name}/{label}", val, prog_bar=False, sync_dist=True)

            # Also log the macro-average for convenience
            self.log(f"{stage}/{metric_name}/mean", per_class_values.mean(),
                     prog_bar=(metric_name == "auc"), sync_dist=True)

    def on_train_epoch_end(self):
        self._log_per_class_metrics("train")

    def on_validation_epoch_end(self):
        self._log_per_class_metrics("val")

    def on_test_epoch_end(self):
        self._log_per_class_metrics("test")

    def training_step(self, batch, _):   return self._step(batch, "train")
    def validation_step(self, batch, _): return self._step(batch, "val")
    def test_step(self, batch, _):       return self._step(batch, "test")



    def configure_optimizers(self):
        opt       = torch.optim.AdamW(self.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.max_epochs)
        return {"optimizer": opt, "lr_scheduler": scheduler}