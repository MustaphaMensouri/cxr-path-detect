import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score


@torch.no_grad()
def collect_probs_and_labels(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray]:
    """Collect all sigmoid probabilities and true labels from a dataloader."""
    model.eval()
    model.to(device)
    all_probs, all_labels = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        probs  = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(y.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


def optimize_thresholds(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: str = "cuda",
    search_range: np.ndarray | None = None,
) -> np.ndarray:
    if search_range is None:
        search_range = np.arange(0.05, 0.95, 0.01)

    probs, labels = collect_probs_and_labels(model, val_loader, device)
    num_classes   = probs.shape[1]
    best_thresholds = np.full(num_classes, 0.5)

    for c in range(num_classes):
        best_f1 = -1.0
        for t in search_range:
            preds = (probs[:, c] >= t).astype(int)
            # zero_division=0 avoids warnings on all-zero predictions
            f1 = f1_score(labels[:, c].astype(int), preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresholds[c] = t

    return best_thresholds