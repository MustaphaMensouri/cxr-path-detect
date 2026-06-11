import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms


class WeightedBCELoss(nn.Module):
    def __init__(self, max_weight=10.0):
        super().__init__()
        self.max_weight = max_weight

    def forward(self, logits, targets):
        logits = logits.float()
        targets = targets.float()

        P = targets.sum(dim=0).clamp(min=1.0)
        N = (1.0 - targets).sum(dim=0).clamp(min=1.0)
        total = P + N

        beta_p = (total / P).clamp(max=self.max_weight)
        beta_n = (total / N).clamp(max=self.max_weight)

        bce = nn.functional.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
        )

        weights = targets * beta_p + (1.0 - targets) * beta_n

        return (weights * bce).mean()

class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4.0, gamma_pos=1.0, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits, targets):
        targets = targets.float()

        probs = torch.sigmoid(logits)
        probs_pos = probs
        probs_neg = 1.0 - probs

        # asymmetric probability clipping for negatives
        if self.clip is not None and self.clip > 0:
            probs_neg = (probs_neg + self.clip).clamp(max=1.0)

        loss_pos = targets * torch.log(probs_pos.clamp(min=self.eps))
        loss_neg = (1.0 - targets) * torch.log(probs_neg.clamp(min=self.eps))

        # asymmetric focusing
        pt = probs_pos * targets + probs_neg * (1.0 - targets)
        gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
        weight = (1.0 - pt).pow(gamma)

        loss = -weight * (loss_pos + loss_neg)
        return loss.mean()
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )

        probs = torch.sigmoid(logits)
        pt = probs * targets + (1 - probs) * (1 - targets)

        focal_weight = (1 - pt).pow(self.gamma)

        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_weight = alpha_t * focal_weight

        loss = focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
class CombinedLoss(nn.Module):
    def __init__(self, losses, weights=None):
        super().__init__()
        self.losses = nn.ModuleList(losses)

        if weights is None:
            weights = [1.0] * len(losses)

        self.weights = weights

    def forward(self, logits, targets):
        total_loss = 0.0

        for loss_fn, weight in zip(self.losses, self.weights):
            total_loss = total_loss + weight * loss_fn(logits, targets)

        return total_loss


def build_loss(cfg):
    if cfg.name == "weighted_bce":
        return WeightedBCELoss(max_weight=cfg.max_weight)
    if cfg.name == "asl":
        return AsymmetricLoss(
            gamma_neg=cfg.gamma_neg,
            gamma_pos=cfg.gamma_pos,
            clip=cfg.clip,
        )
    if cfg.name == "focal":
        return FocalLoss(
            alpha=cfg.alpha,
            gamma=cfg.gamma,
            reduction=cfg.reduction,
        )

    elif cfg.name == "combined":
        losses = [build_loss(loss_cfg) for loss_cfg in cfg.losses]
        return CombinedLoss(losses, cfg.weights)

    raise ValueError(f"Unknown loss: {cfg.name}")


def build_backbone(cfg, num_classes):
    weights = "DEFAULT" if cfg.pretrained else None
    model = getattr(models, cfg.backbone)(weights=weights)

    if hasattr(model, "classifier"):
        if isinstance(model.classifier, nn.Linear):
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, num_classes)
        elif isinstance(model.classifier, nn.Sequential):
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, num_classes)

    elif hasattr(model, "fc"):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif hasattr(model, "heads"):
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(f"Unsupported backbone head: {cfg.backbone}")

    return model


def build_transforms(cfg_aug):
    train_tf = transforms.Compose([
        transforms.Resize((cfg_aug.resize, cfg_aug.resize)),
        transforms.RandomCrop(cfg_aug.crop),
        transforms.RandomAffine(
            degrees=cfg_aug.affine_degrees,
            translate=(cfg_aug.translate, cfg_aug.translate),
            scale=(cfg_aug.scale_min, cfg_aug.scale_max),
        ),
        transforms.RandomAutocontrast(p=cfg_aug.autocontrast_p),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((cfg_aug.resize, cfg_aug.resize)),
        transforms.CenterCrop(cfg_aug.crop),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    return train_tf, val_tf