import hydra
import wandb
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger, CSVLogger
from omegaconf import DictConfig, OmegaConf

from src.datamodule import XrayDataModule
from src.lightning_module import XrayClassifier

import torch
torch.serialization.add_safe_globals([DictConfig])

import json
from pathlib import Path



@hydra.main(config_path="configs", config_name="config", version_base=None)
def train(cfg: DictConfig):
    L.seed_everything(42, workers=True)

    dm     = XrayDataModule(cfg)
    class_names=dm.labels
    model  = XrayClassifier(cfg, num_classes=len(class_names), class_names=class_names, max_epochs=cfg.train.max_epochs)
    use_wandb = cfg.wandb.get("enabled", True)

    if use_wandb:
        api = wandb.Api()
        try:
            runs = api.runs(f"{cfg.wandb.entity}/{cfg.wandb.project}")
            run_number = len(runs) + 1
        except Exception:
            run_number = 1

        run_name = f"experiment_{run_number}"
        logger = WandbLogger(project=cfg.wandb.project, name=run_name, entity=cfg.wandb.entity, notes=cfg.wandb.notes, tags=list(cfg.wandb.tags), log_model=False, config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))
    else:
        logger = CSVLogger(save_dir="logs/", name="local_run")
        print("[Logger] W&B disabled — logging to terminal + CSV (logs/local_run/)")
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
            patience=cfg.train.get("early_stopping_patience", 3),
            min_delta=1e-4,
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
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
        callbacks=callbacks,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        num_sanity_val_steps=2,
    )

    trainer.fit(model, dm)

    if use_wandb and trainer.is_global_zero:
        best_ckpt = trainer.checkpoint_callback.best_model_path

        best_model = XrayClassifier.load_from_checkpoint(
            best_ckpt,
            cfg=cfg,
            num_classes=len(class_names),
            class_names=class_names,
            max_epochs=cfg.train.max_epochs,
        )

        thresholds = best_model.best_thresholds.detach().cpu().tolist()
        threshold_data = {
            label: float(th)
            for label, th in zip(class_names, thresholds)
        }

        threshold_path = Path("thresholds.json")
        threshold_path.write_text(json.dumps(threshold_data, indent=2), encoding="utf-8")

        artifact = wandb.Artifact(
            name=cfg.wandb.artifact_name,
            type="model",
            metadata={
                "run_name": run_name,
                "monitor": "val/auc_macro",
                "best_model_path": best_ckpt,
                "num_classes": len(class_names),
                "backbone": cfg.model.backbone,
                "loss": cfg.loss.name,
            },
        )

        artifact.add_file(threshold_path, name="thresholds.json")
        artifact.add_file(best_ckpt, name="model.ckpt")
        artifact.add_file(cfg.data.labels_path, name="labels_used.txt")

        logger.experiment.log_artifact(artifact, aliases=["candidate"])    
    
    if cfg.train.get("run_test", False):
        trainer.test(model, dm, ckpt_path="best")
    
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    train()