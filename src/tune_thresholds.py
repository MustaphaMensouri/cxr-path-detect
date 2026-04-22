import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score


def optimize_thresholds(
    model: torch.nn.Module,
    val_loader: DataLoader,
    search_range: np.ndarray = np.arange(0.05, 0.95, 0.01),
) -> np.ndarray:
    raw_model   = model.module if hasattr(model, "module") else model
    num_classes = raw_model.num_classes
    device      = next(raw_model.parameters()).device

    t = torch.zeros(num_classes, dtype=torch.float32, device=device)

    if not dist.is_initialized() or dist.get_rank() == 0:
        raw_model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                probs = torch.sigmoid(raw_model(x.to(device)))
                all_probs.append(probs.cpu().numpy())
                all_labels.append(y.numpy())

        probs  = np.concatenate(all_probs)   # [N, C]
        labels = np.concatenate(all_labels)  # [N, C]

        best = np.array([
            max(search_range, key=lambda th: f1_score(labels[:, c].astype(int), (probs[:, c] >= th).astype(int), zero_division=0))
            for c in range(num_classes)
        ])
        t.copy_(torch.tensor(best, dtype=torch.float32, device=device))

    if dist.is_initialized():
        dist.barrier()
        dist.broadcast(t, src=0)

    return t.cpu().numpy()