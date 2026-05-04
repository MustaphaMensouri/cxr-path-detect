import hydra
import wandb
import torch
import numpy as np
from sklearn.metrics import f1_score
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from src.datamodule import XrayDataModule, LABELS
from src.lightning_module import XrayClassifier

def tune_thresholds_single_gpu(model, val_dataset, class_names, search_range=np.arange(0.05, 0.95, 0.01)):
    """
    Run on rank 0 only, on a single GPU, outside of DDP context.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    all_probs, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            probs = torch.sigmoid(model(x))
            all_probs.append(probs.cpu())
            all_labels.append(y)

    probs_np = torch.cat(all_probs).numpy()
    labels_np = torch.cat(all_labels).numpy()

    best_thresholds = np.zeros(len(class_names), dtype=np.float32)
    for c in range(len(class_names)):
        best_f1 = -1.0
        best_th = 0.5
        for th in search_range:
            preds = (probs_np[:, c] >= th).astype(int)
            f1 = f1_score(labels_np[:, c].astype(int), preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_th = th
        best_thresholds[c] = best_th

    print("\nPer-class F1 thresholds:")
    for name, thr in zip(class_names, best_thresholds):
        print(f"  {name:25s}: {thr:.2f}")

    return best_thresholds


@hydra.main(config_path="configs", config_name="config", version_base=None)
def train(cfg: DictConfig):
    L.seed_everything(42, workers=True)
    dm = XrayDataModule(cfg.data)
    model = XrayClassifier(
        cfg.model,
        num_classes=len(LABELS),
        max_epochs=cfg.train.max_epochs,
        class_names=LABELS,
    )

    # WandB setup
    api = wandb.Api()
    try:
        runs = api.runs(f"{cfg.wandb.entity}/{cfg.wandb.project}")
        run_number = len(runs) + 1
    except Exception:
        run_number = 1
    run_name = f"experiment_{run_number}"

    logger = WandbLogger(
        project=cfg.wandb.project,
        name=run_name,
        notes=cfg.wandb.notes,
        tags=list(cfg.wandb.tags),
        log_model=True,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
    )

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
            min_delta=1e-3,
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

    # ── TRAIN & VALIDATE ─────────────────────────────────────────────
    trainer.fit(model, dm)

    # ── THRESHOLD TUNING (single GPU, rank 0 only) ─────────────────
    # This runs OUTSIDE the DDP loop, avoiding barrier deadlocks.
    if trainer.global_rank == 0:
        # Load best checkpoint for threshold tuning
        best_path = trainer.checkpoint_callback.best_model_path
        if best_path:
            ckpt = torch.load(best_path, map_location="cpu")
            # Load weights into a fresh copy to avoid DDP wrapper issues
            tune_model = XrayClassifier(
                cfg.model,
                num_classes=len(LABELS),
                max_epochs=cfg.train.max_epochs,
                class_names=LABELS,
            )
            tune_model.load_state_dict(ckpt["state_dict"])
        else:
            tune_model = model  # fallback to current weights

        best_thresholds = tune_thresholds_single_gpu(
            tune_model,
            dm.val_dataset,
            LABELS,
        )
        # Save thresholds to a file so all ranks can load them if needed
        np.save("best_thresholds.npy", best_thresholds)
    else:
        best_thresholds = None

    # If running DDP, make sure rank 0 has finished writing before test
    trainer.strategy.barrier()

    # Load thresholds on all ranks
    if trainer.global_rank != 0:
        best_thresholds = np.load("best_thresholds.npy")

    model.set_thresholds(best_thresholds)

    # ── TEST ────────────────────────────────────────────────────────
    trainer.test(model, dm, ckpt_path="best")

    wandb.finish()


if __name__ == "__main__":
    train()