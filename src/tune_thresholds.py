import numpy as np
import torch
import torch.distributed as dist
from sklearn.metrics import f1_score
import lightning as L


class ThresholdTuner(L.Callback):
    """
    After training ends, sweep per-class F1 thresholds on the val set
    and write them back into the model via model.set_thresholds().
    """

    def __init__(self, val_loader, search_range: np.ndarray = np.arange(0.05, 0.95, 0.01)):
        self.val_loader   = val_loader
        self.search_range = search_range

    def on_fit_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        raw_model   = pl_module.model          # unwrapped — pl_module is never DDP-wrapped itself
        num_classes = pl_module.num_classes
        device      = pl_module.device

        t = torch.zeros(num_classes, dtype=torch.float32, device=device)

        if not dist.is_initialized() or dist.get_rank() == 0:
            raw_model.eval()
            all_probs, all_labels = [], []
            with torch.no_grad():
                for x, y in self.val_loader:
                    probs = torch.sigmoid(raw_model(x.to(device)))
                    all_probs.append(probs.cpu().numpy())
                    all_labels.append(y.numpy())

            probs  = np.concatenate(all_probs)
            labels = np.concatenate(all_labels)

            best = np.array([
                max(
                    self.search_range,
                    key=lambda th: f1_score(
                        labels[:, c].astype(int),
                        (probs[:, c] >= th).astype(int),
                        zero_division=0,
                    ),
                )
                for c in range(num_classes)
            ])
            t.copy_(torch.tensor(best, dtype=torch.float32, device=device))

        if dist.is_initialized():
            dist.barrier()
            dist.broadcast(t, src=0)

        pl_module.set_thresholds(t.cpu().numpy())

        if not dist.is_initialized() or dist.get_rank() == 0:
            print("\nPer-class thresholds:")
            for name, threshold in zip(pl_module.class_names, t.cpu().numpy()):
                print(f"  {name:20s}: {threshold:.2f}")