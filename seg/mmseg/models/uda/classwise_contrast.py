"""Class-wise contrastive alignment helpers for UDA segmentation."""

# ---------------------------------------------------------------
# Copyright (c) 2024.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClasswiseContrastiveLoss(nn.Module):
    """Class-wise contrastive alignment for UDA segmentation.

    The module builds class prototypes from source and target feature maps and
    optimizes an InfoNCE objective so that prototypes from the same class
    become closer while prototypes from different classes are pushed apart.

    Args:
        num_classes (int): Number of semantic classes.
        temperature (float): Softmax temperature used for the InfoNCE loss.
        loss_weight (float): Multiplicative weight for the resulting loss.
        min_pixels (int): Minimal number of valid pixels per class that is
            required to form a prototype.
        max_samples (int, optional): Maximum number of pixels sampled per class
            to build the prototype. ``None`` keeps all pixels.
        normalize (bool): Whether to L2-normalize prototypes before computing
            similarities.
    """

    def __init__(
        self,
        num_classes: int,
        temperature: float = 0.1,
        loss_weight: float = 1.0,
        min_pixels: int = 0,
        max_samples: Optional[int] = None,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.temperature = temperature
        self.loss_weight = loss_weight
        self.min_pixels = min_pixels
        self.max_samples = max_samples
        self.normalize = normalize

    @staticmethod
    def _maybe_resize(
        tensor: torch.Tensor,
        size_hw: Sequence[int],
        mode: str = 'nearest',
    ) -> torch.Tensor:
        if tensor.shape[-2:] == tuple(size_hw):
            return tensor
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(1)
            resized = F.interpolate(tensor.float(), size=size_hw, mode=mode)
            return resized.squeeze(1)
        return F.interpolate(tensor.float(), size=size_hw, mode=mode)

    def _flatten_features(
        self,
        feat: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Reshape spatial features and labels into flat tensors."""
        if labels.dim() == 4:
            labels = labels.squeeze(1)
        feat = feat.permute(0, 2, 3, 1).reshape(-1, feat.shape[1])
        labels = labels.reshape(-1)
        if mask is not None:
            mask = mask.reshape(-1)
        return feat, labels, mask

    def _compute_prototypes(
        self,
        feat: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[int, torch.Tensor], Dict[int, float]]:
        """Compute class-wise prototypes using (optional) weights."""
        if mask is not None:
            valid = (labels >= 0) & (labels < self.num_classes) & mask.bool()
            weights = mask[valid].float()
        else:
            valid = (labels >= 0) & (labels < self.num_classes)
            weights = None
        feat = feat[valid]
        labels = labels[valid]
        if weights is not None:
            weights = weights.float()
        if feat.numel() == 0:
            return {}, {}

        classes, inverse, counts = labels.unique(
            return_inverse=True, return_counts=True)
        prototypes: Dict[int, torch.Tensor] = {}
        used_counts: Dict[int, float] = {}
        for idx, cls in enumerate(classes.tolist()):
            cls_mask = inverse == idx
            indices = torch.nonzero(cls_mask, as_tuple=False).squeeze(1)
            if indices.numel() == 0:
                continue
            if self.min_pixels and indices.numel() < self.min_pixels:
                continue
            perm = None
            if self.max_samples is not None and \
                    indices.numel() > self.max_samples:
                perm = torch.randperm(indices.numel(), device=feat.device)
                perm = perm[:self.max_samples]
                indices = indices[perm]
            cls_feat = feat[indices]
            if weights is not None:
                cls_weights = weights[cls_mask]
                if perm is not None:
                    cls_weights = cls_weights[perm]
                weight_sum = cls_weights.sum()
                if weight_sum.item() <= 0:
                    continue
                prototype = (cls_feat * cls_weights.unsqueeze(1)).sum(0) / weight_sum
                used_counts[cls] = float(weight_sum.item())
            else:
                prototype = cls_feat.mean(0)
                used_counts[cls] = float(cls_feat.shape[0])
            prototypes[cls] = prototype
        return prototypes, used_counts

    def forward(
        self,
        src_feat: torch.Tensor,
        src_label: torch.Tensor,
        tgt_feat: torch.Tensor,
        tgt_label: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> Optional[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]:
        """Compute the class-wise contrastive alignment loss."""
        feat_hw = src_feat.shape[-2:]
        src_label = self._maybe_resize(src_label, feat_hw, mode='nearest').long()
        tgt_label = self._maybe_resize(tgt_label, feat_hw, mode='nearest').long()
        tgt_mask_tensor: Optional[torch.Tensor] = None
        if tgt_mask is not None:
            tgt_mask_tensor = self._maybe_resize(tgt_mask.float(), feat_hw, mode='nearest')
            tgt_mask_tensor = tgt_mask_tensor.squeeze(1) if tgt_mask_tensor.dim() == 4 else tgt_mask_tensor

        src_feat_flat, src_label_flat, _ = self._flatten_features(src_feat, src_label)
        tgt_feat_flat, tgt_label_flat, tgt_mask_flat = self._flatten_features(
            tgt_feat, tgt_label, tgt_mask_tensor)

        src_proto, src_counts = self._compute_prototypes(src_feat_flat, src_label_flat)
        tgt_proto, tgt_counts = self._compute_prototypes(
            tgt_feat_flat, tgt_label_flat, tgt_mask_flat)

        common = sorted(set(src_proto.keys()) & set(tgt_proto.keys()))
        if not common:
            return None

        src_stack = torch.stack([src_proto[c] for c in common], dim=0)
        tgt_stack = torch.stack([tgt_proto[c] for c in common], dim=0)
        if self.normalize:
            src_stack = F.normalize(src_stack, dim=1)
            tgt_stack = F.normalize(tgt_stack, dim=1)
        logits = torch.mm(src_stack, tgt_stack.t()) / self.temperature
        targets = torch.arange(len(common), device=logits.device)

        row_loss = F.cross_entropy(logits, targets, reduction='none')
        col_loss = F.cross_entropy(logits.t(), targets, reduction='none')
        weights = src_stack.new_tensor([
            max(min(src_counts[c], tgt_counts[c]), 1.0) for c in common
        ])
        weights = weights / weights.sum()
        loss = 0.5 * ((row_loss * weights).sum() + (col_loss * weights).sum())
        loss = loss * self.loss_weight

        diag_sim = logits.diag().detach()
        stats = {
            'num_classes': logits.new_tensor(float(len(common))),
            'pos_sim': diag_sim.mean() if diag_sim.numel() > 0 else logits.new_tensor(0.0),
        }
        losses = {'loss_contrast': loss}
        return losses, stats
