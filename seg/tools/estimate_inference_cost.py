"""Utility to estimate inference parameters and FLOPs for segmentation models.

This script builds the model defined by a config file, runs a dummy forward
pass that mimics inference, and records the floating-point operations executed
by convolutional, linear, normalization, pooling, and attention layers.  The
parameter count is computed directly from the instantiated model.  For
HRDA-style architectures, the dummy forward pass triggers the multi-scale
sliding-window inference code path, so the reported FLOPs reflect the aggregate
cost of the detail crops as well as the low-resolution context pass.

Example:
    python seg/tools/estimate_inference_cost.py \
        seg/configs/mic/gtaHR2csHR_mic_hrda.py
"""

import argparse
import contextlib
import importlib
from collections import defaultdict
from copy import deepcopy
from typing import Dict, Iterable, Optional, Tuple

import torch

try:
    from mmcv import Config
except ModuleNotFoundError as exc:  # pragma: no cover - dependency message
    raise ModuleNotFoundError(
        'mmcv is required to parse segmentation configs. Please install '
        'mmcv-full before using this script.'
    ) from exc

from mmseg.models import build_segmentor

# The HRDA implementation ships the MixVisionTransformer attention block in the
# segmentation codebase.  Importing locally avoids relying on external mmcv
# attention helpers for FLOP accounting.  Some configs also use a CLIP-based
# adapter that defines its own attention block with a compatible interface, so
# both variants are detected dynamically.
from mmseg.models.backbones.mix_transformer import Attention as MixAttention

ClipAttention = None
for _module_name in (
    'mmseg.models.backbones.clip_vision_adapter',
    'mmseg.models.backbones.clip_adapter',
):
    with contextlib.suppress(ModuleNotFoundError, ImportError, AttributeError):
        ClipAttention = getattr(importlib.import_module(_module_name), 'Attention')
        break


UNITS = ['', 'K', 'M', 'G', 'T', 'P']


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Estimate inference parameter and FLOP cost for a config.')
    parser.add_argument('config', help='Path to the segmentation config file.')
    parser.add_argument(
        '--input-shape',
        type=int,
        nargs=2,
        metavar=('HEIGHT', 'WIDTH'),
        default=None,
        help='Explicit input resolution (defaults to the config test pipeline).',
    )
    parser.add_argument(
        '--device',
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device used for the dummy forward pass.',
    )
    parser.add_argument(
        '--no-summary',
        action='store_true',
        help='Only print the total counts (omit the detailed breakdown).',
    )
    return parser.parse_args()


def human_readable(num: float, suffixes: Iterable[str]) -> str:
    value = float(num)
    for suffix in suffixes:
        if abs(value) < 1000:
            return f'{value:.2f}{suffix}'
        value /= 1000.0
    return f'{value:.2f}{suffixes[-1]}'


def infer_input_shape(cfg: Config) -> Optional[Tuple[int, int]]:
    """Extract an input tensor shape from the test pipeline if possible."""

    data_cfg = cfg.get('data') or {}
    test_cfg = data_cfg.get('test') or {}
    pipeline = test_cfg.get('pipeline', [])
    for step in pipeline:
        if step.get('type') == 'MultiScaleFlipAug':
            img_scale = step.get('img_scale') or step.get('img_scales')
            if isinstance(img_scale, (tuple, list)):
                # mmseg specifies scales as (W, H)
                if isinstance(img_scale[0], (list, tuple)):
                    # Take the first scale if multi-scale inference is listed.
                    width, height = img_scale[0]
                else:
                    width, height = img_scale
                return height, width
    return None


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def prod(values: Iterable[int]) -> int:
    result = 1
    for value in values:
        result *= int(value)
    return result


def conv2d_flops(module: torch.nn.Conv2d, inputs, outputs) -> int:
    out = outputs if isinstance(outputs, torch.Tensor) else outputs[0]
    batch_size, out_channels, out_h, out_w = out.shape
    kernel_h, kernel_w = module.kernel_size
    in_channels = module.in_channels
    groups = module.groups
    filters_per_channel = in_channels // groups
    conv_per_position_flops = kernel_h * kernel_w * filters_per_channel * 2
    active_elements_count = batch_size * out_h * out_w * out_channels
    total_flops = conv_per_position_flops * active_elements_count
    if module.bias is not None:
        total_flops += batch_size * out_channels * out_h * out_w
    return int(total_flops)


def linear_flops(module: torch.nn.Linear, inputs, outputs) -> int:
    input_tensor = inputs[0]
    if input_tensor.dim() == 1:
        num_instances = 1
        in_features = input_tensor.numel()
    else:
        num_instances = prod(input_tensor.shape[:-1])
        in_features = input_tensor.shape[-1]
    out_features = module.out_features
    total_flops = 2 * num_instances * in_features * out_features
    if module.bias is not None:
        total_flops += num_instances * out_features
    return int(total_flops)


def batchnorm_flops(module: torch.nn.BatchNorm2d, inputs, outputs) -> int:
    out = outputs if isinstance(outputs, torch.Tensor) else outputs[0]
    return int(2 * out.numel())


def layernorm_flops(module: torch.nn.LayerNorm, inputs, outputs) -> int:
    inp = inputs[0]
    return int(5 * inp.numel())


def avgpool_flops(module: torch.nn.AvgPool2d, inputs, outputs) -> int:
    inp = inputs[0]
    output = outputs if isinstance(outputs, torch.Tensor) else outputs[0]
    kernel_h = module.kernel_size if isinstance(module.kernel_size, int) else module.kernel_size[0]
    kernel_w = module.kernel_size if isinstance(module.kernel_size, int) else module.kernel_size[1]
    return int(prod(output.shape) * (kernel_h * kernel_w))


def adaptive_avgpool_flops(module: torch.nn.AdaptiveAvgPool2d, inputs, outputs) -> int:
    inp = inputs[0]
    output = outputs if isinstance(outputs, torch.Tensor) else outputs[0]
    num_elements = prod(inp.shape)
    out_elements = prod(output.shape)
    kernel_ops = num_elements // out_elements
    return int(out_elements * kernel_ops)


def attention_flops(module, inputs, outputs) -> int:
    # MixVisionTransformer attention receives (x, H, W) whereas the CLIP vision
    # adapter passes explicit query/key/value tensors followed by (H, W).
    if len(inputs) >= 3 and isinstance(inputs[1], torch.Tensor):
        query = inputs[0]
        key = inputs[1]
        value = inputs[2]
        B, num_queries, channels = query.shape
        num_heads = getattr(module, 'num_heads', 1)
        head_dim = channels // max(num_heads, 1)
        sr_ratio = getattr(module, 'sr_ratio', 1)
        kv_tokens = key.shape[1]
        if sr_ratio > 1 and len(inputs) >= 5:
            try:
                height = int(inputs[3])
                width = int(inputs[4])
            except (TypeError, ValueError):
                height = width = 0
            if height > 0 and width > 0:
                kv_tokens = (height // sr_ratio) * (width // sr_ratio)
        flops_qk = B * num_heads * num_queries * kv_tokens * head_dim * 2
        flops_attn_v = B * num_heads * num_queries * head_dim * kv_tokens * 2
        return int(flops_qk + flops_attn_v)

    x, H, W = inputs
    B, N, C = x.shape
    num_heads = module.num_heads
    head_dim = C // num_heads
    if module.sr_ratio > 1:
        kv_tokens = (H // module.sr_ratio) * (W // module.sr_ratio)
    else:
        kv_tokens = N
    flops_qk = B * num_heads * N * kv_tokens * head_dim * 2
    flops_attn_v = B * num_heads * N * head_dim * kv_tokens * 2
    return int(flops_qk + flops_attn_v)


def multihead_attention_flops(module: torch.nn.MultiheadAttention, inputs, outputs) -> int:
    query = inputs[0]
    key = inputs[1] if len(inputs) > 1 and inputs[1] is not None else query
    value = inputs[2] if len(inputs) > 2 and inputs[2] is not None else key

    if module.batch_first:
        batch_size, query_len, q_embed = query.shape
        _, key_len, k_embed = key.shape
        _, value_len, v_embed = value.shape
    else:
        query_len, batch_size, q_embed = query.shape
        key_len, _, k_embed = key.shape
        value_len, _, v_embed = value.shape

    embed_dim = module.embed_dim
    num_heads = module.num_heads
    head_dim = embed_dim // max(num_heads, 1)

    q_proj = 2 * batch_size * query_len * q_embed * embed_dim
    k_proj = 2 * batch_size * key_len * k_embed * embed_dim
    v_proj = 2 * batch_size * value_len * v_embed * embed_dim
    bias_ops = 0
    if module.in_proj_bias is not None:
        bias_ops = batch_size * (query_len + key_len + value_len) * embed_dim

    attn_scores = 2 * batch_size * num_heads * query_len * key_len * head_dim
    attn_weighted = 2 * batch_size * num_heads * query_len * head_dim * value_len

    return int(q_proj + k_proj + v_proj + bias_ops + attn_scores + attn_weighted)


FLOP_HANDLERS: Dict[type, callable] = {
    torch.nn.Conv2d: conv2d_flops,
    torch.nn.Linear: linear_flops,
    torch.nn.BatchNorm2d: batchnorm_flops,
    torch.nn.LayerNorm: layernorm_flops,
    torch.nn.AvgPool2d: avgpool_flops,
    torch.nn.AdaptiveAvgPool2d: adaptive_avgpool_flops,
    torch.nn.MultiheadAttention: multihead_attention_flops,
}

if MixAttention is not None:
    FLOP_HANDLERS[MixAttention] = attention_flops
if ClipAttention is not None:
    FLOP_HANDLERS[ClipAttention] = attention_flops


class FlopAnalyzer:

    def __init__(self, module: torch.nn.Module):
        self.module = module
        self.handles = []
        self.flops: Dict[str, int] = defaultdict(int)

    def _make_hook(self, cls):
        handler = FLOP_HANDLERS[cls]

        def hook(mod, inputs, outputs):
            self.flops[cls.__name__] += handler(mod, inputs, outputs)

        return hook

    def add_hooks(self):
        for mod in self.module.modules():
            for cls in FLOP_HANDLERS:
                if isinstance(mod, cls):
                    self.handles.append(mod.register_forward_hook(self._make_hook(cls)))

    def clear(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def total(self) -> int:
        return int(sum(self.flops.values()))


def build_model(cfg: Config) -> torch.nn.Module:
    cfg = cfg.copy()
    model_cfg = deepcopy(cfg.model)
    model_cfg.setdefault('train_cfg', None)
    model_cfg.setdefault('test_cfg', None)
    if 'pretrained' in model_cfg:
        model_cfg['pretrained'] = None
    backbone_cfg = model_cfg.get('backbone')
    if isinstance(backbone_cfg, dict) and backbone_cfg.get('init_cfg'):
        backbone_cfg = deepcopy(backbone_cfg)
        backbone_cfg['init_cfg'] = None
        model_cfg['backbone'] = backbone_cfg
    model = build_segmentor(model_cfg)
    model.eval()
    return model


def compute_hrda_crop_count(model: torch.nn.Module, height: int, width: int) -> Optional[int]:
    if not hasattr(model, 'scales') or not hasattr(model, 'hr_slide_inference'):
        return None
    scales = getattr(model, 'scales')
    if len(scales) <= 1 or not getattr(model, 'hr_slide_inference'):
        return None
    crop_size = getattr(model, 'crop_size', None)
    if crop_size is None:
        return None
    hr_scale = scales[-1]
    scaled_h = int(round(height * hr_scale))
    scaled_w = int(round(width * hr_scale))
    crop_h, crop_w = crop_size
    if getattr(model, 'hr_slide_overlapping', True):
        stride_h = crop_h // 2
        stride_w = crop_w // 2
    else:
        stride_h, stride_w = crop_h, crop_w
    h_grids = max(scaled_h - crop_h + stride_h - 1, 0) // stride_h + 1
    w_grids = max(scaled_w - crop_w + stride_w - 1, 0) // stride_w + 1
    return h_grids * w_grids


def analyze(cfg_path: str, input_shape: Optional[Tuple[int, int]], device: str,
            verbose: bool) -> None:
    cfg = Config.fromfile(cfg_path)
    inferred_shape = input_shape or infer_input_shape(cfg)
    if inferred_shape is None:
        raise ValueError(
            'Unable to infer an input resolution from the config. Please pass '
            '--input-shape H W explicitly.')
    height, width = inferred_shape
    model = build_model(cfg)

    dummy = torch.randn(1, 3, height, width)
    device_ctx = contextlib.nullcontext()
    if device == 'cuda':
        device_ctx = torch.cuda.device('cuda:0')
    with device_ctx:
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but no GPU is available.')
        model = model.to(device)
        dummy = dummy.to(device)
        analyzer = FlopAnalyzer(model)
        analyzer.add_hooks()
        try:
            with torch.no_grad():
                if hasattr(model, 'forward_dummy'):
                    model.forward_dummy(dummy)
                elif hasattr(model, 'encode_decode'):
                    model.encode_decode(dummy, None)
                else:
                    raise AttributeError(
                        'The model does not implement forward_dummy() or ' \
                        'encode_decode(), so inference FLOPs cannot be ' \
                        'estimated.'
                    )
        finally:
            analyzer.clear()

    total_flops = analyzer.total()
    params = count_parameters(model)
    crop_count = compute_hrda_crop_count(model, height, width)

    print(f'Config: {cfg_path}')
    print(f'Input shape: {height}x{width}')
    if crop_count is not None:
        crop_size = getattr(model, 'crop_size')
        print(
            f'HRDA detail crops: {crop_count} (crop size {crop_size[0]}x{crop_size[1]}, '
            f'scales={list(getattr(model, "scales"))})'
        )
    print(f'Parameters: {params:,} ({human_readable(params, UNITS)})')
    print(
        f'FLOPs: {total_flops:,} '
        f'({human_readable(total_flops, UNITS)})'
    )
    if verbose:
        for name, value in sorted(analyzer.flops.items()):
            print(f'  {name:<18}: {human_readable(value, UNITS)}')


def main():
    args = parse_args()
    analyze(
        cfg_path=args.config,
        input_shape=tuple(args.input_shape) if args.input_shape else None,
        device=args.device,
        verbose=not args.no_summary,
    )


if __name__ == '__main__':
    main()
