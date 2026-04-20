import torch
import torch.nn as nn
from torchvision import models
from torchmetrics import AUROC, AveragePrecision, Precision, Recall, F1Score
import lightning as L


class WeightedBCELoss(nn.Module):
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
    def __init__(self, cfg, num_classes: int, max_epochs: int, class_names: list[str]):
        super().__init__()
        self.save_hyperparameters(ignore=["cfg"])
        self.cfg = cfg
        self.max_epochs = max_epochs
        self.class_names = class_names
        self.num_classes = num_classes

        backbone    = getattr(models, cfg.backbone)(weights="DEFAULT" if cfg.pretrained else None)
        backbone.classifier = nn.Linear(backbone.classifier.in_features, num_classes)
        self.model  = backbone

        self.loss = WeightedBCELoss()

        self.train_auc = AUROC(task="multilabel", num_labels=num_classes)
        self.val_auc   = AUROC(task="multilabel", num_labels=num_classes)
        self.test_auc  = AUROC(task="multilabel", num_labels=num_classes)
        metric_kwargs = dict(task="multilabel", num_labels=num_classes, average="none")
        self.val_precision = Precision(**metric_kwargs)
        self.val_recall    = Recall(**metric_kwargs)
        self.val_f1        = F1Score(**metric_kwargs)
        self.val_perclass_auc = AUROC(task="multilabel", num_labels=num_classes, average="none")
 
        self.test_precision = Precision(**metric_kwargs)
        self.test_recall    = Recall(**metric_kwargs)
        self.test_f1        = F1Score(**metric_kwargs)
        self.test_perclass_auc = AUROC(task="multilabel", num_labels=num_classes, average="none")

    def forward(self, x):
        return self.model(x)

    def _step(self, batch, stage):
        x, y    = batch
        logits  = self(x)
        loss    = self.loss(logits, y)
        probs   = torch.sigmoid(logits)
        y_int  = y.int()
        if stage == "train":
            auc = self.train_auc(probs, y_int)
        elif stage == "val":
            auc = self.val_auc(probs, y_int)
            self.val_precision.update(probs, y_int)
            self.val_recall.update(probs, y_int)
            self.val_f1.update(probs, y_int)
            self.val_perclass_auc.update(probs, y_int)
        else:  # test
            auc = self.test_auc(probs, y_int)
            self.test_precision.update(probs, y_int)
            self.test_recall.update(probs, y_int)
            self.test_f1.update(probs, y_int)
            self.test_perclass_auc.update(probs, y_int)
 
        self.log_dict(
            {f"{stage}/loss": loss, f"{stage}/auc_macro": auc},
            prog_bar=True, on_step=False, on_epoch=True, sync_dist=True,
        )
        return loss
    def _log_perclass_metrics(self, stage: str):
        if stage == "val":
            precision = self.val_precision.compute()   # [C]
            recall    = self.val_recall.compute()      # [C]
            f1        = self.val_f1.compute()          # [C]
            auc       = self.val_perclass_auc.compute() # [C]
            self.val_precision.reset()
            self.val_recall.reset()
            self.val_f1.reset()
            self.val_perclass_auc.reset()
        else:  # test
            precision = self.test_precision.compute()
            recall    = self.test_recall.compute()
            f1        = self.test_f1.compute()
            auc       = self.test_perclass_auc.compute()
            self.test_precision.reset()
            self.test_recall.reset()
            self.test_f1.reset()
            self.test_perclass_auc.reset()
 
        metrics = {}
        for i, name in enumerate(self.class_names):
            metrics[f"{stage}/precision/{name}"] = precision[i]
            metrics[f"{stage}/recall/{name}"]    = recall[i]
            metrics[f"{stage}/f1/{name}"]        = f1[i]
            metrics[f"{stage}/auc/{name}"]       = auc[i]
 
        # sync_dist=False: state was already synced by torchmetrics internally
        self.log_dict(metrics, prog_bar=False, on_epoch=True, sync_dist=True)

    def training_step(self, batch, _):   return self._step(batch, "train")
    def validation_step(self, batch, _): return self._step(batch, "val")
    def test_step(self, batch, _):       return self._step(batch, "test")

    def on_validation_epoch_start(self):
        self.val_precision.reset()
        self.val_recall.reset()
        self.val_f1.reset()
        self.val_perclass_auc.reset()

    def on_validation_epoch_end(self):
        self._log_perclass_metrics("val")
 
    def on_test_epoch_end(self):
        self._log_perclass_metrics("test")

    def configure_optimizers(self):
        opt       = torch.optim.AdamW(self.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.max_epochs)
        return {"optimizer": opt, "lr_scheduler": scheduler}