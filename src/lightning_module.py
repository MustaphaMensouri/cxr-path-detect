import torch
import torch.nn as nn
from torchvision import models
from torchmetrics import AUROC, AveragePrecision
import lightning as L
from weighted_bce import WeightedBCELoss


class XrayClassifier(L.LightningModule):
    def __init__(self, cfg, num_classes):
        super().__init__()
        self.save_hyperparameters(ignore=["cfg"])
        self.cfg = cfg

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
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.cfg.epochs)
        return {"optimizer": opt, "lr_scheduler": scheduler}