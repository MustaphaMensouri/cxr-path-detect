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
        t = torch.zeros(num_classes, dtype=torch.float32)

        # ── ALL ranks reach this point ────────────────────────────────────────
        # Only rank 0 does inference, but the barrier + broadcast below
        # must be called on every rank simultaneously.
        if trainer.global_rank == 0:
            device = pl_module.device

            loader = DataLoader(
                val_dataset,
                batch_size=64,
                shuffle=False,
                num_workers=0,   # must be 0 inside DDP worker process
                pin_memory=False,
            )

            pl_module.model.eval()
            all_probs, all_labels = [], []

            with torch.no_grad():
                for x, y in loader:
                    probs = torch.sigmoid(pl_module.model(x.to(device)))
                    all_probs.append(probs.cpu())
                    all_labels.append(y)

            probs_np  = torch.cat(all_probs).numpy()
            labels_np = torch.cat(all_labels).numpy()

            best = np.array([
                max(
                    self.search_range,
                    key=lambda th, c=c: f1_score(
                        labels_np[:, c].astype(int),
                        (probs_np[:, c] >= th).astype(int),
                        zero_division=0,
                    ),
                )
                for c in range(num_classes)
            ], dtype=np.float32)

            t.copy_(torch.tensor(best))

            print("\nPer-class F1 thresholds:")
            for name, thr in zip(pl_module.class_names, best):
                print(f"  {name:25s}: {thr:.2f}")

        # ── Both ranks participate in these two calls ─────────────────────────
        trainer.strategy.barrier()            # rank 1 was here all along
        t = trainer.strategy.broadcast(t, src=0)

        pl_module.set_thresholds(t.cpu().numpy())