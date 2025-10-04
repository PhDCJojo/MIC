#!/usr/bin/env python3
"""Plot a t-SNE visualization for the target domain of a UDA segmentation model.

The script loads a model configuration and checkpoint, extracts logits for
pixels belonging to each class from the target domain dataset configured in the
UDA training recipe, and renders a t-SNE scatter plot where each point
represents a pixel-level logit vector colored by its (ground-truth or predicted)
class.
"""

import argparse
import copy
import random
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE

import mmcv
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmcv.utils import DictAction

from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the visualization script."""

    default_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    parser = argparse.ArgumentParser(
        description='Plot a t-SNE embedding for target-domain logits',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('config', help='path to the mmsegmentation config file')
    parser.add_argument('checkpoint', help='path to a trained checkpoint file')
    parser.add_argument(
        '--output',
        default='work_dirs/tsne_target.png',
        help='file to store the resulting plot',
    )
    parser.add_argument(
        '--save-data',
        default=None,
        help='optional path to store the sampled logits and labels as a .npz file',
    )
    parser.add_argument(
        '--save-embedding',
        default=None,
        help='optional path to store the computed 2-D t-SNE embedding (.npy)',
    )
    parser.add_argument(
        '--max-images',
        type=int,
        default=0,
        help='limit the number of target images processed (0 means all images)',
    )
    parser.add_argument(
        '--max-per-class',
        type=int,
        default=2048,
        help='maximum number of pixel logits sampled per class (<=0 disables the limit)',
    )
    parser.add_argument(
        '--perplexity',
        type=float,
        default=30.0,
        help='perplexity parameter passed to sklearn.manifold.TSNE',
    )
    parser.add_argument(
        '--point-size',
        type=float,
        default=6.0,
        help='marker size used when plotting the scatter points',
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.65,
        help='marker alpha channel used for the scatter plot',
    )
    parser.add_argument(
        '--figsize',
        type=float,
        nargs=2,
        default=(10.0, 10.0),
        metavar=('WIDTH', 'HEIGHT'),
        help='size of the matplotlib figure in inches',
    )
    parser.add_argument(
        '--device',
        default=default_device,
        help='device used for forward passes during feature extraction',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='random seed for reproducible sampling and t-SNE initialization',
    )
    parser.add_argument(
        '--use-predictions',
        action='store_true',
        help='color samples by predicted classes instead of ground-truth labels',
    )
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config file, key=value',
    )
    return parser.parse_args()


def set_random_seed(seed: Optional[int]) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducibility."""

    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_target_dataset(cfg: Config) -> object:
    """Instantiate the target-domain dataset defined in the training config."""

    if 'train' not in cfg.data or 'target' not in cfg.data.train:
        raise KeyError('The provided config does not define a target training dataset.')
    target_cfg = copy.deepcopy(cfg.data.train.target)
    target_cfg.setdefault('test_mode', False)
    dataset = build_dataset(target_cfg)
    if not hasattr(dataset, 'CLASSES'):
        raise AttributeError('The constructed dataset does not expose class metadata.')
    collect = getattr(getattr(dataset, 'pipeline', None), 'transforms', None)
    collect = collect[-1] if collect else None
    keys = getattr(collect, 'keys', [])
    if 'gt_semantic_seg' not in keys:
        mmcv.print_log(
            'Warning: target pipeline does not collect ground-truth annotations. '
            'Consider enabling --use-predictions to color by inferred classes.',
            'tsne')
    return dataset


def prepare_model(cfg: Config, checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """Build the segmentation model and restore weights from ``checkpoint_path``."""

    cfg = copy.deepcopy(cfg)
    cfg.model.setdefault('pretrained', None)
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, checkpoint_path, map_location='cpu')
    model.to(device)
    model.eval()
    return model


def _extract_sample(
        dataset,
        model: torch.nn.Module,
        device: torch.device,
        index: int,
        use_predictions: bool,
        generator: torch.Generator,
        max_per_class: Optional[int],
        class_buffers: List[List[torch.Tensor]],
        class_counts: List[int],
        max_classes: int,
) -> None:
    """Process a single dataset sample and append logits to ``class_buffers``."""

    sample = dataset[index]
    if 'img' not in sample:
        raise KeyError('Sample does not contain an ``img`` key required for inference.')

    img = sample['img']
    if hasattr(img, 'data'):
        img_tensor = img.data
    else:
        img_tensor = img
    if img_tensor.dim() == 3:
        img_tensor = img_tensor.unsqueeze(0)
    if img_tensor.dim() != 4 or img_tensor.size(0) != 1:
        raise ValueError(f'Unexpected image tensor shape: {tuple(img_tensor.shape)}')
    img_tensor = img_tensor.to(device)

    img_metas_container = sample.get('img_metas')
    if img_metas_container is None:
        raise KeyError('Sample does not contain ``img_metas`` required by encode_decode.')
    img_metas = img_metas_container.data if hasattr(img_metas_container, 'data') else img_metas_container
    img_metas = [img_metas]

    gt_container = sample.get('gt_semantic_seg')
    if gt_container is not None:
        gt = gt_container.data if hasattr(gt_container, 'data') else gt_container
        gt = gt.squeeze(0).long().cpu()
    else:
        gt = None

    with torch.no_grad():
        logits = model.encode_decode(img_tensor, img_metas)
    logits = logits.squeeze(0).cpu()  # (num_classes, H, W)
    pred_labels = logits.argmax(dim=0)

    if gt is None and not use_predictions:
        raise RuntimeError(
            'Ground-truth annotations are required unless --use-predictions is enabled.')

    if gt is not None:
        valid_mask = gt != dataset.ignore_index
    else:
        valid_mask = torch.ones_like(pred_labels, dtype=torch.bool)

    label_source = pred_labels if use_predictions or gt is None else gt

    flat_logits = logits.permute(1, 2, 0).reshape(-1, logits.shape[0])
    flat_labels = label_source.reshape(-1)
    valid_mask = valid_mask.reshape(-1)

    flat_logits = flat_logits[valid_mask]
    flat_labels = flat_labels[valid_mask]

    if flat_logits.numel() == 0:
        return

    unique_labels = flat_labels.unique()
    for cls_idx_tensor in unique_labels:
        cls_idx = int(cls_idx_tensor.item())
        if cls_idx < 0 or cls_idx >= max_classes:
            continue
        cls_mask = flat_labels == cls_idx
        cls_logits = flat_logits[cls_mask]
        if cls_logits.size(0) == 0:
            continue
        if max_per_class is not None:
            remaining = max_per_class - class_counts[cls_idx]
            if remaining <= 0:
                continue
            if cls_logits.size(0) > remaining:
                perm = torch.randperm(cls_logits.size(0), generator=generator)
                cls_logits = cls_logits[perm[:remaining]]
        class_buffers[cls_idx].append(cls_logits)
        class_counts[cls_idx] += cls_logits.size(0)


def collect_logits(
        dataset,
        model: torch.nn.Module,
        device: torch.device,
        max_images: Optional[int],
        max_per_class: Optional[int],
        use_predictions: bool,
        seed: Optional[int],
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """Iterate over the target dataset and gather per-class logits."""

    num_classes = len(dataset.CLASSES)
    buffers: List[List[torch.Tensor]] = [[] for _ in range(num_classes)]
    counts: List[int] = [0 for _ in range(num_classes)]
    generator = torch.Generator(device='cpu')
    if seed is not None:
        generator.manual_seed(seed)
    processed = 0
    total = len(dataset) if max_images is None else min(max_images, len(dataset))
    print(f'Collecting logits from {total} target samples...')

    for idx in range(len(dataset)):
        if max_images is not None and processed >= max_images:
            break
        _extract_sample(
            dataset,
            model,
            device,
            idx,
            use_predictions,
            generator,
            max_per_class,
            buffers,
            counts,
            num_classes,
        )
        processed += 1
        if total:
            print(f'  Processed {processed}/{total} images', end='\r')
        if max_per_class is not None and all(c >= max_per_class for c in counts):
            break

    print(f'  Processed {processed}/{total} images')

    collected_features: List[torch.Tensor] = []
    collected_labels: List[torch.Tensor] = []
    for cls_idx, chunks in enumerate(buffers):
        if not chunks:
            continue
        feats = torch.cat(chunks, dim=0)
        collected_features.append(feats)
        collected_labels.append(
            torch.full((feats.size(0),), cls_idx, dtype=torch.long))

    if not collected_features:
        raise RuntimeError('No logits were collected. Verify dataset and model outputs.')

    features = torch.cat(collected_features, dim=0)
    labels = torch.cat(collected_labels, dim=0)
    return features, labels, counts


def run_tsne(features: np.ndarray, perplexity: float, seed: Optional[int]) -> np.ndarray:
    """Compute the 2-D t-SNE embedding for the collected features."""

    num_samples = features.shape[0]
    if num_samples < 2:
        raise RuntimeError('At least two feature vectors are required for t-SNE.')
    max_perplexity = max(1.0, float(num_samples - 1))
    effective_perplexity = min(perplexity, max_perplexity)
    if effective_perplexity >= num_samples:
        effective_perplexity = max(1.0, num_samples - 1.0)
    tsne = TSNE(
        n_components=2,
        init='pca',
        random_state=seed,
        perplexity=effective_perplexity,
        learning_rate='auto',
    )
    return tsne.fit_transform(features)


def plot_embedding(
        embedding: np.ndarray,
        labels: np.ndarray,
        class_names: Sequence[str],
        palette: Sequence[Sequence[int]],
        counts: Sequence[int],
        output_path: Path,
        point_size: float,
        alpha: float,
        figsize: Tuple[float, float],
        title: str,
) -> None:
    """Render and store the final scatter plot."""

    fig, ax = plt.subplots(figsize=figsize)
    unique_labels = np.unique(labels)
    colors = np.array(palette, dtype=np.float32) / 255.0
    for cls_idx in unique_labels:
        cls_idx = int(cls_idx)
        mask = labels == cls_idx
        color = colors[cls_idx] if cls_idx < len(colors) else None
        class_name = class_names[cls_idx] if cls_idx < len(class_names) else str(cls_idx)
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=point_size,
            c=[color],
            alpha=alpha,
            edgecolors='none',
            label=f'{class_name} ({counts[cls_idx]})',
        )
    ax.set_title(title)
    ax.set_xlabel('t-SNE dimension 1')
    ax.set_ylabel('t-SNE dimension 2')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout(rect=[0, 0, 0.8, 1])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def maybe_save_array(path: Optional[str], array: np.ndarray) -> None:
    """Persist ``array`` to ``path`` when provided."""

    if path is None:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, array)
    print(f'Saved array to {output_path}')


def maybe_save_npz(path: Optional[str], **tensors: np.ndarray) -> None:
    """Store multiple arrays in an ``.npz`` archive when ``path`` is provided."""

    if path is None:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **tensors)
    print(f'Saved features to {output_path}')


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA device requested but CUDA is not available.')

    set_random_seed(args.seed)

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    dataset = build_target_dataset(cfg)
    print(f'Target dataset: {dataset.__class__.__name__} with {len(dataset)} samples')

    model = prepare_model(cfg, args.checkpoint, device)

    max_images = args.max_images if args.max_images > 0 else None
    max_per_class = args.max_per_class if args.max_per_class > 0 else None

    features, labels, counts = collect_logits(
        dataset,
        model,
        device,
        max_images=max_images,
        max_per_class=max_per_class,
        use_predictions=args.use_predictions,
        seed=args.seed,
    )

    features_np = features.numpy().astype(np.float32)
    labels_np = labels.numpy().astype(np.int32)

    print('Collected samples per class:')
    for idx, (name, count) in enumerate(zip(dataset.CLASSES, counts)):
        status = ''
        if max_per_class is not None and count < max_per_class:
            status = ' (insufficient)'  # Not enough samples collected for this class
        print(f'  [{idx:02d}] {name:<16}: {count}{status}')

    embedding = run_tsne(features_np, args.perplexity, args.seed)
    title = f'Target t-SNE ({Path(args.config).stem})'
    output_path = Path(args.output)
    plot_embedding(
        embedding,
        labels_np,
        dataset.CLASSES,
        dataset.PALETTE,
        counts,
        output_path,
        point_size=args.point_size,
        alpha=args.alpha,
        figsize=tuple(args.figsize),
        title=title,
    )
    print(f'Saved t-SNE visualization to {output_path}')

    maybe_save_npz(args.save_data, features=features_np, labels=labels_np)
    maybe_save_array(args.save_embedding, embedding)


if __name__ == '__main__':
    main()
