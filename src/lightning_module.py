import torch
import torch.nn as nn
from torchvision import models
from torchmetrics import AUROC, AveragePrecision
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
    def __init__(self, cfg, num_classes, max_epochs):
        super().__init__()
        self.save_hyperparameters(ignore=["cfg"])
        self.cfg = cfg
        self.max_epochs = max_epochs

        backbone    = getattr(models, cfg.backbone)(weights="DEFAULT" if cfg.pretrained else None)
        backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)
        self.model  = backbone

        self.loss = WeightedBCELoss()
        self.auc  = AUROC(task="multilabel", num_labels=num_classes)
        self.ap   = AveragePrecision(task="multilabel", num_labels=num_classes)

    def forward(self, x):
        return self.model(x)

    def _step(self, batch, stage):
        x, y    = batch
        logits  = self(x)
        loss    = self.loss(logits, y)
        probs   = torch.sigmoid(logits)
        self.log_dict(
            {f"{stage}/loss": loss, f"{stage}/auc": self.auc(probs, y.int()), f"{stage}/ap": self.ap(probs, y.int())},
            prog_bar=True, on_step=False, on_epoch=True, sync_dist=True,
        )
        return loss

    def training_step(self, batch, _):   return self._step(batch, "train")
    def validation_step(self, batch, _): return self._step(batch, "val")
    def test_step(self, batch, _):       return self._step(batch, "test")

    def configure_optimizers(self):
        opt       = torch.optim.AdamW(self.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.max_epochs)
        return {"optimizer": opt, "lr_scheduler": scheduler}