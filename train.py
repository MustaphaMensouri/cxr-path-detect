import hydra
import wandb
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf

from src.datamodule import XrayDataModule, LABELS
from src.lightning_module import XrayClassifier
from src.tune_thresholds import ThresholdTuner



@hydra.main(config_path="configs", config_name="config", version_base=None)
def train(cfg: DictConfig):
    L.seed_everything(42, workers=True)

    dm     = XrayDataModule(cfg.data)
    model  = XrayClassifier(cfg.model, num_classes=len(LABELS), max_epochs=cfg.train.max_epochs, class_names=LABELS)
    
    api = wandb.Api()
    try:
        runs = api.runs(f"{cfg.wandb.entity}/{cfg.wandb.project}")
        run_number = len(runs) + 1
    except Exception:
        run_number = 1

    run_name = f"experiment_{run_number}"
    logger = WandbLogger(project=cfg.wandb.project, name=run_name, notes=cfg.wandb.notes, tags=list(cfg.wandb.tags), log_model=True, config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))
    checkpoint_cb = ModelCheckpoint(
        monitor="val/auc_macro",
        mode="max",
        save_top_k=1,
        filename="best",
        verbose=True,
    )
    callbacks = [
        checkpoint_cb,
        EarlyStopping(
            monitor="val/auc_macro",
            mode="max",
            patience=cfg.train.get("early_stopping_patience", 3),  # from config, not hardcoded
            min_delta=1e-3,       # ignore improvements
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),   # see LR curve in wandb
        TQDMProgressBar(refresh_rate=50),
        ThresholdTuner(dm),
    ]
    trainer = L.Trainer(
        max_epochs=cfg.train.max_epochs,
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        strategy=cfg.train.strategy,
        precision=cfg.train.precision,
        log_every_n_steps=cfg.train.log_every_n_steps,
        enable_progress_bar=cfg.train.enable_progress_bar,
        enable_model_summary=cfg.train.enable_model_summary,
        logger=logger,
        callbacks= callbacks,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        num_sanity_val_steps=2,
    )

    trainer.fit(model, dm)
    if trainer.global_rank == 0:
        best_ckpt = checkpoint_cb.best_model_path

        test_trainer = L.Trainer(
            accelerator=cfg.train.accelerator,
            devices=1,                       # ← single GPU
            strategy="auto",                 # no DDP
            precision=cfg.train.precision,
            logger=logger,                   # same wandb run
            enable_progress_bar=True,
            enable_model_summary=False,
        )

        # Load best weights into a fresh module so thresholds are already set
        test_model = XrayClassifier.load_from_checkpoint(
            best_ckpt,
            cfg=cfg.model,
        )
        # Copy tuned thresholds from the trained model
        test_model.set_thresholds(model.thresholds.cpu().numpy())

        test_trainer.test(test_model, datamodule=dm)
    wandb.finish()


if __name__ == "__main__":
    train()