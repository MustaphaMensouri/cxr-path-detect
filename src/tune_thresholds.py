import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import lightning as L
import sys


class ThresholdTuner(L.Callback):
    def __init__(self, dm, search_range=np.arange(0.05, 0.95, 0.01)):
        self.dm = dm
        self.search_range = search_range

    def on_fit_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        rank = trainer.global_rank
        print(f"[rank {rank}] ThresholdTuner.on_fit_end — entered", flush=True)

        if rank != 0:
            print(f"[rank {rank}] ThresholdTuner — skipping (not rank 0)", flush=True)
            return

        val_dataset = self.dm.val_dataset
        if val_dataset is None:
            raise RuntimeError("val_dataset is None — val_dataloader was never called.")

        print(f"[rank 0] val_dataset size: {len(val_dataset)}", flush=True)
        print(f"[rank 0] pl_module.device: {pl_module.device}", flush=True)

        device = pl_module.device
        loader = DataLoader(
            val_dataset,
            batch_size=64,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

        print(f"[rank 0] starting inference loop over {len(loader)} batches", flush=True)
        pl_module.model.eval()
        all_probs, all_labels = [], []

        with torch.no_grad():
            for i, (x, y) in enumerate(loader):
                if i % 20 == 0:
                    print(f"[rank 0] inference batch {i}/{len(loader)}", flush=True)
                probs = torch.sigmoid(pl_module.model(x.to(device)))
                all_probs.append(probs.cpu())
                all_labels.append(y)

        print(f"[rank 0] inference done, running threshold search", flush=True)

        probs_np  = torch.cat(all_probs).numpy()
        labels_np = torch.cat(all_labels).numpy()
        num_classes = pl_module.num_classes

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

        print(f"[rank 0] threshold search done", flush=True)
        pl_module.set_thresholds(best)

        print("\nPer-class F1 thresholds (rank 0):", flush=True)
        for name, thr in zip(pl_module.class_names, best):
            print(f"  {name:25s}: {thr:.2f}", flush=True)

        print(f"[rank 0] ThresholdTuner.on_fit_end — done", flush=True)
        sys.stdout.flush()