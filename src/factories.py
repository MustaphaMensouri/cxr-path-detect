import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms


class WeightedBCELoss(nn.Module):
    def __init__(self, max_weight=10.0):
        super().__init__()
        self.max_weight = max_weight

    def forward(self, logits, targets):
        P = targets.sum(dim=0).clamp(min=1)
        N = (1 - targets).sum(dim=0).clamp(min=1)
        total = P + N

        beta_p = (total / P).clamp(max=self.max_weight)
        beta_n = (total / N).clamp(max=self.max_weight)

        probs = torch.sigmoid(logits).clamp(1e-7, 1 - 1e-7)

        pos_loss = -beta_p * targets * torch.log(probs)
        neg_loss = -beta_n * (1 - targets) * torch.log(1 - probs)

        return (pos_loss + neg_loss).mean()


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


def build_loss(cfg):
    if cfg.name == "weighted_bce":
        return WeightedBCELoss(max_weight=cfg.max_weight)

    if cfg.name == "focal":
        return FocalLoss(
            alpha=cfg.alpha,
            gamma=cfg.gamma,
            reduction=cfg.reduction,
        )

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