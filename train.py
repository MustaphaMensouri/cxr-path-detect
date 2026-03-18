import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from src.lightning_module import CVModel
from src.datamodule import CVDataModule

@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # 1. Initialize DataModule
    datamodule = CVDataModule(
        data_dir=cfg.data.data_dir,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers
    )

    # 2. Initialize Model
    model = CVModel(
        model_name=cfg.model.model_name,
        num_classes=cfg.model.num_classes,
        lr=cfg.model.lr
    )

    # 3. Initialize wandb Logger
    wandb_logger = WandbLogger(
        project=cfg.logger.project,
        name=cfg.logger.name,
        log_model="all" # Logs model artifacts directly to W&B
    )

    # 4. Set up Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        dirpath="checkpoints/"
    )

    # 5. Initialize Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        log_every_n_steps=10
    )

    # 6. Run Training
    trainer.fit(model=model, datamodule=datamodule)

if __name__ == "__main__":
    main()