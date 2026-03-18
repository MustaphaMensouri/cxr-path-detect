import pytorch_lightning as pl
import timm
import torch
import torch.nn.functional as F
import torchmetrics

class CVModel(pl.LightningModule):
    def __init__(self, model_name: str, num_classes: int, lr: float):
        super().__init__()
        # Saves arguments to self.hparams and logs them to wandb automatically
        self.save_hyperparameters() 
        
        # timm -> gives the model
        self.model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        
        # Metrics
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = self.train_acc(logits, y)
        
        # Log to wandb
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = self.val_acc(logits, y)
        
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        self.log('val_acc', acc, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)