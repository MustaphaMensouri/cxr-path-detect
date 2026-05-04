import hydra
import wandb
import torch
import lightning as L
from lightning.pytorch.callbacks import (
    ModelCheckpoint, EarlyStopping, TQDMProgressBar, LearningRateMonitor
)
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
from src.datamodule import XrayDataModule, LABELS
from src.lightning_module import XrayClassifier
from src.tune_thresholds import ThresholdTuner
import sys


@hydra.main(config_path="configs", config_name="config", version_base=None)
def train(cfg: DictConfig):
    L.seed_everything(42, workers=True)

    dm    = XrayDataModule(cfg.data)
    model = XrayClassifier(
        cfg.model,
        num_classes=len(LABELS),
        max_epochs=cfg.train.max_epochs,
        class_names=LABELS,
    )

    try:
        api = wandb.Api()
        runs = api.runs(f"{cfg.wandb.entity}/{cfg.wandb.project}")
        run_number = len(runs) + 1
    except Exception:
        run_number = 1

    logger = WandbLogger(
        project=cfg.wandb.project,
        name=f"experiment_{run_number}",
        notes=cfg.wandb.notes,
        tags=list(cfg.wandb.tags),
        log_model=True,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
    )

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
            patience=cfg.train.get("early_stopping_patience", 3),
            min_delta=1e-3,
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
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
        callbacks=callbacks,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        num_sanity_val_steps=2,
    )

    print(">>> calling trainer.fit()", flush=True)
    trainer.fit(model, dm)
    print(">>> trainer.fit() returned", flush=True)
    sys.stdout.flush()

    # ── single-GPU test, rank 0 only ─────────────────────────────────────────
    print(f">>> global_rank={trainer.global_rank}, entering test block", flush=True)

    if trainer.global_rank == 0:
        best_ckpt = checkpoint_cb.best_model_path
        print(f">>> best checkpoint: {best_ckpt}", flush=True)

        if not best_ckpt:
            print(">>> WARNING: no checkpoint found, using current weights", flush=True)

        print(">>> building single-GPU test_trainer", flush=True)
        test_trainer = L.Trainer(
            accelerator=cfg.train.accelerator,
            devices=1,
            strategy="auto",
            precision=cfg.train.precision,
            logger=logger,
            enable_progress_bar=True,
            enable_model_summary=False,
        )
        print(">>> test_trainer built", flush=True)

        print(">>> loading checkpoint into test_model", flush=True)
        test_model = XrayClassifier.load_from_checkpoint(
            best_ckpt,
            cfg=cfg.model,
        )
        print(">>> checkpoint loaded", flush=True)

        test_model.set_thresholds(model.thresholds.cpu().numpy())
        print(f">>> thresholds set: {model.thresholds.cpu().numpy()}", flush=True)

        print(">>> calling test_trainer.test()", flush=True)
        sys.stdout.flush()
        test_trainer.test(test_model, datamodule=dm)
        print(">>> test_trainer.test() returned", flush=True)

    else:
        print(f">>> rank {trainer.global_rank} skipping test block", flush=True)

    sys.stdout.flush()
    wandb.finish()
    print(">>> wandb.finish() done", flush=True)


if __name__ == "__main__":
    train()