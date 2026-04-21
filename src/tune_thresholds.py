"""
Per-class F1 threshold optimization — DDP-safe.

Only rank-0 collects probs and finds best thresholds.
Thresholds are then broadcast to all ranks via torch.distributed.
"""

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score


@torch.no_grad()
def collect_probs_and_labels(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect sigmoid probabilities and true labels. Always uses the unwrapped model."""
    model.eval()
    all_probs, all_labels = [], []
    for x, y in loader:
        x      = x.to(device)
        logits = model(x)
        probs  = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(y.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


def optimize_thresholds(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: str,
    search_range: np.ndarray | None = None,
) -> np.ndarray:
    """
    DDP-safe per-class F1 threshold search.

    - Rank 0 collects all val probs and finds best thresholds.
    - Thresholds are broadcast to every other rank so all processes
      call model.set_thresholds() with the same values.

    Returns best_thresholds on every rank.
    """
    if search_range is None:
        search_range = np.arange(0.05, 0.95, 0.01)

    is_distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if is_distributed else 0

    # Unwrap DDP if needed so we call the raw forward()
    raw_model = model.module if hasattr(model, "module") else model
    num_classes = raw_model.num_classes

    # Allocate tensor on every rank so broadcast has a destination
    thresholds_tensor = torch.zeros(num_classes, dtype=torch.float32, device=device)

    if rank == 0:
        probs, labels   = collect_probs_and_labels(raw_model, val_loader, device)
        best_thresholds = np.full(num_classes, 0.5)

        for c in range(num_classes):
            best_f1 = -1.0
            for t in search_range:
                preds = (probs[:, c] >= t).astype(int)
                f1    = f1_score(labels[:, c].astype(int), preds, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_thresholds[c] = t

        thresholds_tensor.copy_(torch.tensor(best_thresholds, dtype=torch.float32))

    # Broadcast rank-0 result to all other ranks
    if is_distributed:
        dist.barrier()
        dist.broadcast(thresholds_tensor, src=0)

    return thresholds_tensor.cpu().numpy()