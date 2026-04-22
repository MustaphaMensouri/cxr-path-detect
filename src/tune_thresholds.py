import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import lightning as L


class ThresholdTuner(L.Callback):
    def __init__(self, dm, search_range=np.arange(0.05, 0.95, 0.01)):
        self.dm = dm
        self.search_range = search_range

    def on_fit_end(self, trainer, pl_module):
        val_dataset = self.dm.val_dataset
        if val_dataset is None:
            raise RuntimeError("val_dataset is None — val_dataloader was never called.")
        num_classes = pl_module.num_classes
        device      = pl_module.device
        t = torch.zeros(num_classes, dtype=torch.float32, device=device)

        if not dist.is_initialized() or dist.get_rank() == 0:
            # Single-process loader — no DDP sampler, no cross-rank sync needed
            loader = DataLoader(
                val_dataset,
                batch_size=64,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
            )
            raw_model = pl_module.model
            raw_model.eval()
            all_probs, all_labels = [], []
            with torch.no_grad():
                for x, y in loader:
                    probs = torch.sigmoid(raw_model(x.to(device)))
                    all_probs.append(probs.cpu().numpy())
                    all_labels.append(y.numpy())

            probs  = np.concatenate(all_probs)
            labels = np.concatenate(all_labels)

            best = np.array([
                max(
                    self.search_range,
                    key=lambda th, c=c: f1_score(
                        labels[:, c].astype(int),
                        (probs[:, c] >= th).astype(int),
                        zero_division=0,
                    ),
                )
                for c in range(num_classes)
            ])
            t.copy_(torch.tensor(best, dtype=torch.float32, device=device))
            print("\nPer-class thresholds:")
            for name, threshold in zip(pl_module.class_names, best):
                print(f"  {name:20s}: {threshold:.2f}")

        if dist.is_initialized():
            dist.barrier()
            dist.broadcast(t, src=0)

        pl_module.set_thresholds(t.cpu().numpy())