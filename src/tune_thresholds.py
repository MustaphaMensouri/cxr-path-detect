import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import lightning as L


class ThresholdTuner(L.Callback):
    def __init__(self, dm, search_range=np.arange(0.05, 0.95, 0.01)):
        self.dm = dm
        self.search_range = search_range

    def on_fit_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        val_dataset = self.dm.val_dataset
        if val_dataset is None:
            raise RuntimeError(
                "val_dataset is None — val_dataloader was never called."
            )

        num_classes = pl_module.num_classes
        best = np.full(num_classes, 0.5, dtype=np.float32)  # safe default

        # ── Only rank 0 runs inference ────────────────────────────────────────
        if trainer.global_rank == 0:
            device = pl_module.device  # cuda:0 on rank 0

            # CRITICAL: num_workers=0 — spawning DataLoader workers inside a
            # DDP-spawned process causes a deadlock with num_workers > 0.
            loader = DataLoader(
                val_dataset,
                batch_size=64,
                shuffle=False,
                num_workers=0,
                pin_memory=False,
            )

            raw_model = pl_module.model
            raw_model.eval()

            all_probs, all_labels = [], []
            with torch.no_grad():
                for x, y in loader:
                    probs = torch.sigmoid(raw_model(x.to(device)))
                    all_probs.append(probs.cpu().numpy())
                    all_labels.append(y.numpy())

            probs_np  = np.concatenate(all_probs)   # [N, C]
            labels_np = np.concatenate(all_labels)  # [N, C]

            best = np.array(
                [
                    max(
                        self.search_range,
                        key=lambda th, c=c: f1_score(
                            labels_np[:, c].astype(int),
                            (probs_np[:, c] >= th).astype(int),
                            zero_division=0,
                        ),
                    )
                    for c in range(num_classes)
                ],
                dtype=np.float32,
            )

            print("\nPer-class F1 thresholds:")
            for name, thr in zip(pl_module.class_names, best):
                print(f"  {name:25s}: {thr:.2f}")

        # ── Broadcast from rank 0 to all other ranks ─────────────────────────
        # Use Lightning's strategy API instead of raw dist calls — safer during
        # on_fit_end when Lightning may be mid-teardown of the process group.
        t = torch.tensor(best, dtype=torch.float32)
        trainer.strategy.barrier()                      # sync before broadcast
        t = trainer.strategy.broadcast(t, src=0)       # rank 0 → all ranks

        pl_module.set_thresholds(t.cpu().numpy())