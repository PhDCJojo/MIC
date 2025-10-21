"""Adapters for integrating IJepa components.

This module implements light-weight versions of the components required to
apply the IJepa masking strategy inside the UDA segmentation pipeline.  The
goal is not to provide a drop-in replacement for the original IJepa training
code, but to offer the same high-level interfaces (``VisionTransformerPredictor``
and ``MaskCollator``) so that target domain images can be processed with
IJepa-style masking.

The implementation is intentionally self contained in order to work in the
restricted execution environment used for the MIC benchmark.  The code mirrors
the behaviour of
``facebookresearch/ijepa/src/models/vision_transformer.py`` and
``facebookresearch/ijepa/src/masks/multiblock.py`` at a functional level while
remaining compact and easy to audit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _to_2tuple(value: int | Iterable[int]) -> Tuple[int, int]:
    """Utility to normalise spatial arguments."""

    if isinstance(value, Iterable):
        value = list(value)
        assert len(value) == 2, "expected a pair"
        return int(value[0]), int(value[1])
    return int(value), int(value)


class MLP(nn.Module):
    """Simple feed-forward network used inside the transformer blocks."""

    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:  # noqa: D401 - standard forward
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """A minimal ViT encoder block."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=attn_drop, batch_first=True
        )
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0 else nn.Identity()
        hidden_dim = int(dim * mlp_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, hidden_dim, drop)

    def forward(self, x: Tensor) -> Tensor:  # noqa: D401 - standard forward
        skip = x
        x = self.norm1(x)
        attn, _ = self.attn(x, x, x, need_weights=False)
        x = skip + self.drop_path(attn)

        skip = x
        x = self.norm2(x)
        x = skip + self.drop_path(self.mlp(x))
        return x


class VisionTransformerPredictor(nn.Module):
    """Light-weight Vision Transformer used as IJepa predictor."""

    def __init__(
        self,
        img_size: int | Tuple[int, int] = 224,
        patch_size: int | Tuple[int, int] = 16,
        in_chans: int = 3,
        embed_dim: int = 384,
        depth: int = 6,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        img_size = _to_2tuple(img_size)
        patch_size = _to_2tuple(patch_size)
        assert (
            img_size[0] % patch_size[0] == 0
            and img_size[1] % patch_size[1] == 0
        ), "img_size must be divisible by patch_size"
        self.img_size = img_size
        self.patch_size = patch_size

        self.patch_embed = nn.Conv2d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        grid_size = (
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        )
        self.base_grid_size = grid_size
        num_patches = grid_size[0] * grid_size[1]
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        dpr = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.pred = nn.Linear(embed_dim, in_chans * patch_size[0] * patch_size[1])

        self.reset_parameters()

        self._last_prediction: Optional[Tensor] = None

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.pred.weight, std=0.02)
        if self.pred.bias is not None:
            nn.init.zeros_(self.pred.bias)

    def _interpolate_pos_encoding(self, h: int, w: int, device: torch.device) -> Tensor:
        base_h, base_w = self.base_grid_size
        if (h, w) == (base_h, base_w):
            return self.pos_embed.to(device)
        pos = self.pos_embed.reshape(1, base_h, base_w, self.embed_dim)
        pos = pos.permute(0, 3, 1, 2)
        pos = F.interpolate(pos, size=(h, w), mode="bicubic", align_corners=False)
        pos = pos.permute(0, 2, 3, 1).reshape(1, h * w, self.embed_dim)
        return pos

    @property
    def last_prediction(self) -> Optional[Tensor]:
        """Return the last high-resolution prediction (for debugging)."""

        return self._last_prediction

    def forward(self, imgs: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Predict masked patches and blend them back into the image."""

        B, C, H, W = imgs.shape
        patches = self.patch_embed(imgs)
        Hp, Wp = patches.shape[-2:]
        tokens = patches.flatten(2).transpose(1, 2)
        pos_embed = self._interpolate_pos_encoding(Hp, Wp, imgs.device)
        tokens = tokens + pos_embed

        if mask is not None:
            mask = mask.reshape(B, -1, 1)
            tokens = tokens * (1.0 - mask)

        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        preds = self.pred(tokens)
        preds = preds.view(B, Hp, Wp, C, self.patch_size[0], self.patch_size[1])
        preds = preds.permute(0, 3, 1, 4, 2, 5).reshape(B, C, Hp * self.patch_size[0], Wp * self.patch_size[1])

        if mask is None:
            self._last_prediction = preds.detach()
            return preds

        mask_img = mask.view(B, 1, Hp, Wp)
        mask_img = mask_img.repeat_interleave(self.patch_size[0], -2)
        mask_img = mask_img.repeat_interleave(self.patch_size[1], -1)
        blended = imgs * (1.0 - mask_img) + preds * mask_img
        self._last_prediction = blended.detach()
        return blended


@dataclass
class MultiBlockConfig:
    """Configuration used by :class:`MaskCollator`."""

    input_size: Tuple[int, int]
    patch_size: Tuple[int, int]
    mask_scale: Tuple[float, float] = (0.15, 0.3)
    aspect_ratio: Tuple[float, float] = (0.75, 1.333)
    min_num_blocks: int = 4
    max_num_blocks: int = 8


class MaskCollator:
    """Generate multi-block masks similar to IJepa's implementation."""

    def __init__(
        self,
        input_size: int | Tuple[int, int],
        patch_size: int | Tuple[int, int],
        mask_scale: Tuple[float, float] = (0.15, 0.3),
        aspect_ratio: Tuple[float, float] = (0.75, 1.333),
        min_num_blocks: int = 4,
        max_num_blocks: int = 8,
    ) -> None:
        self.cfg = MultiBlockConfig(
            input_size=_to_2tuple(input_size),
            patch_size=_to_2tuple(patch_size),
            mask_scale=mask_scale,
            aspect_ratio=aspect_ratio,
            min_num_blocks=min_num_blocks,
            max_num_blocks=max_num_blocks,
        )

        Hp = self.cfg.input_size[0] // self.cfg.patch_size[0]
        Wp = self.cfg.input_size[1] // self.cfg.patch_size[1]
        self.grid_size = (Hp, Wp)

    def _sample_block(self) -> Tuple[int, int]:
        area = torch.empty(1).uniform_(*self.cfg.mask_scale).item()
        area = area * self.grid_size[0] * self.grid_size[1]
        log_min = math.log(self.cfg.aspect_ratio[0])
        log_max = math.log(self.cfg.aspect_ratio[1])
        log_ratio = torch.empty(1).uniform_(log_min, log_max).item()
        ratio = math.exp(log_ratio)
        h = int(round((area * ratio) ** 0.5))
        w = int(round((area / ratio) ** 0.5))
        h = max(1, min(h, self.grid_size[0]))
        w = max(1, min(w, self.grid_size[1]))
        return h, w

    def __call__(self, imgs: Tensor) -> Dict[str, Tensor]:
        """Create a batch of random binary masks."""

        device = imgs.device
        B = imgs.shape[0]
        Hp, Wp = self.grid_size
        mask = torch.zeros((B, 1, Hp, Wp), device=device)

        for i in range(B):
            num_blocks = torch.randint(
                self.cfg.min_num_blocks,
                self.cfg.max_num_blocks + 1,
                (1,),
                device=device,
            ).item()
            for _ in range(num_blocks):
                bh, bw = self._sample_block()
                top = torch.randint(
                    0,
                    max(Hp - bh + 1, 1),
                    (1,),
                    device=device,
                ).item()
                left = torch.randint(
                    0,
                    max(Wp - bw + 1, 1),
                    (1,),
                    device=device,
                ).item()
                mask[i, 0, top:top + bh, left:left + bw] = 1.0

        return {"mask": mask}

    def upsample(self, mask: Tensor, size: Tuple[int, int]) -> Tensor:
        """Upsample a patch-level mask to the image resolution."""

        return F.interpolate(mask.float(), size=size, mode="nearest")

