import torch
import torch.nn as nn
from torchvision import models
from torchmetrics import Accuracy, AUROC
import lightning as L


class XrayClassifier(L.LightningModule):
    def __init__(self, cfg, num_classes):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg

        backbone = getattr(models, cfg.backbone)(weights="DEFAULT" if cfg.pretrained else None)
        backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)
        self.model = backbone

        self.loss = nn.CrossEntropyLoss()
        self.acc  = Accuracy(task="multiclass", num_classes=num_classes)
        self.auc  = AUROC(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def _step(self, batch, stage):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        self.log_dict({f"{stage}/loss": loss, f"{stage}/acc": self.acc(logits, y), f"{stage}/auc": self.auc(logits, y)}, prog_bar=True)
        return loss

    def training_step(self, batch, _):   return self._step(batch, "train")
    def validation_step(self, batch, _): return self._step(batch, "val")
    def test_step(self, batch, _):       return self._step(batch, "test")

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
        return {"optimizer": opt, "lr_scheduler": scheduler}