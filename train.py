import hydra
import wandb
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf

from src.datamodule import XrayDataModule, LABELS
from src.lightning_module import XrayClassifier
from src.tune_thresholds import optimize_thresholds



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
    
    callbacks = [
        ModelCheckpoint(
            monitor="val/auc_macro",
            mode="max",
            save_top_k=1,
            filename="best",
            verbose=True,   
        ),
        EarlyStopping(
            monitor="val/auc_macro",
            mode="max",
            patience=cfg.train.get("early_stopping_patience", 3),  # from config, not hardcoded
            min_delta=1e-3,       # ignore improvements
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),   # see LR curve in wandb
        TQDMProgressBar(refresh_rate=50),
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
    device = next(model.parameters()).device
    best_thresholds = optimize_thresholds(model, dm.val_dataloader(), device=str(device))
    model.set_thresholds(best_thresholds)
 
    # Log chosen thresholds to W&B for reproducibility
    threshold_dict = {
        f"threshold/{name}": float(best_thresholds[i])
        for i, name in enumerate(LABELS)
    }
    wandb.log(threshold_dict)
    trainer.test(model, dm, ckpt_path="best")
    wandb.finish()


if __name__ == "__main__":
    train()