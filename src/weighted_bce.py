import torch
import torch.nn as nn

class WeightedBCELoss(nn.Module):
    """
    Weighted Cross-Entropy Loss from ChestX-ray8 paper (Wang et al., 2017).
    
    For each class c in a batch:
        L = βP * Σ(yc=1) [-log f(xc)] + βN * Σ(yc=0) [-log(1 - f(xc))]
    
    where:
        βP = (|P| + |N|) / |P|
        βN = (|P| + |N|) / |N|
    and |P|, |N| are the number of 1s and 0s in the batch labels.
    """
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits, targets: [B, C]  (raw scores, float labels 0/1)
        
        P = targets.sum(dim=0).clamp(min=1)          # positive count per class [C]
        N = (1 - targets).sum(dim=0).clamp(min=1)    # negative count per class [C]
        total = P + N                                 # = batch_size

        beta_p = total / P   # [C]
        beta_n = total / N   # [C]

        # Numerically stable BCE
        probs = torch.sigmoid(logits)
        probs = probs.clamp(1e-7, 1 - 1e-7)

        pos_loss = -beta_p * targets       * torch.log(probs)
        neg_loss = -beta_n * (1 - targets) * torch.log(1 - probs)

        return (pos_loss + neg_loss).mean()