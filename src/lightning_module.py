import torch
import torch.nn as nn
from torchvision import models
from torchmetrics import AUROC, Precision, Recall, F1Score
import lightning as L

import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        # Flatten the tensors for multi-label calculation
        logits = logits.view(-1)
        targets = targets.view(-1)
        
        # Calculate standard Binary Cross Entropy
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # Get probability of the positive class
        p = torch.sigmoid(logits)
        
        # p_t is the probability associated with the true label (targets)
        p_t = p * targets + (1 - p) * (1 - targets)
        
        # Calculate the focal component: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma
        
        # Combine everything
        loss = self.alpha * focal_weight * bce_loss
        
        return loss.mean()


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

        self.loss = FocalLoss()

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