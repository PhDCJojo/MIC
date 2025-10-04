"""Utility script to visualize feature distributions with t-SNE.

This helper mirrors the analysis phase implemented in
``examples/cdan_mcc_sdat.py`` but exposes it as a light-weight command line
tool.  Given a trained classification checkpoint, the script extracts
features for the specified source and target domains and stores a t-SNE plot
on disk.
"""

import argparse
import os
import os.path as osp
import sys
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def _extend_sys_path() -> None:
    """Register project modules so that the script can be executed in-place."""

    current_dir = osp.dirname(osp.abspath(__file__))
    project_root = osp.abspath(osp.join(current_dir, os.pardir))

    if project_root not in sys.path:
        sys.path.append(project_root)
    if current_dir not in sys.path:
        sys.path.append(current_dir)


_extend_sys_path()

from dalib.adaptation.cdan import ImageClassifier  # noqa: E402
from common.utils.analysis import collect_feature, tsne  # noqa: E402

import utils  # noqa: E402  pylint: disable=wrong-import-position


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the t-SNE visualization script."""

    default_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parser = argparse.ArgumentParser(
        description="Plot a t-SNE visualization for source and target features",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--data', default='OfficeHome',
                        choices=utils.get_dataset_names(),
                        help='dataset name')
    parser.add_argument('--root', default='examples/data',
                        help='root directory for datasets')
    parser.add_argument('--source', nargs='+', required=True,
                        help='source domain(s) used during training')
    parser.add_argument('--target', nargs='+', required=True,
                        help='target domain(s) used during training')
    parser.add_argument('--arch', default='resnet50',
                        choices=utils.get_model_names(),
                        help='backbone architecture')
    parser.add_argument('--checkpoint', required=True,
                        help='path to a trained classifier checkpoint')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='mini-batch size for feature extraction')
    parser.add_argument('--workers', type=int, default=4,
                        help='number of data loading workers')
    parser.add_argument('--val-resizing', default='default', choices=['default', 'res.'],
                        help='resizing mode for feature extraction')
    parser.add_argument('--resize-size', type=int, default=224,
                        help='resize size when using "res." mode')
    parser.add_argument('--norm-mean', type=float, nargs=3,
                        default=(0.485, 0.456, 0.406),
                        metavar=('R', 'G', 'B'),
                        help='normalization mean for the transforms')
    parser.add_argument('--norm-std', type=float, nargs=3,
                        default=(0.229, 0.224, 0.225),
                        metavar=('R', 'G', 'B'),
                        help='normalization std for the transforms')
    parser.add_argument('--bottleneck-dim', type=int, default=256,
                        help='feature dimension of the bottleneck layer')
    parser.add_argument('--no-pool', action='store_true',
                        help='disable the global pooling layer')
    parser.add_argument('--scratch', action='store_true',
                        help='initialize the backbone from scratch')
    parser.add_argument('--max-batches', type=int, default=None,
                        help='number of mini-batches per domain to extract features from')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='limit the number of features per domain before running t-SNE')
    parser.add_argument('--output', default=None,
                        help='where to store the resulting visualization')
    parser.add_argument('--source-color', default='r',
                        help='color used for source domain samples')
    parser.add_argument('--target-color', default='b',
                        help='color used for target domain samples')
    parser.add_argument('--random-state', type=int, default=33,
                        help='random seed for sklearn.manifold.TSNE')
    parser.add_argument('--device', default=default_device,
                        help='device used for feature extraction')
    return parser.parse_args()


def _prepare_transforms(args: argparse.Namespace) -> Tuple[object, object]:
    """Construct the transforms used to extract features."""

    transform = utils.get_val_transform(
        args.val_resizing,
        resize_size=args.resize_size,
        norm_mean=tuple(args.norm_mean),
        norm_std=tuple(args.norm_std),
    )
    return transform, transform


def _create_dataloaders(args: argparse.Namespace, device: torch.device):
    """Instantiate the dataloaders for source and target domains."""

    train_transform, val_transform = _prepare_transforms(args)
    source_dataset, target_dataset, _, _, num_classes, _ = utils.get_dataset(
        args.data,
        args.root,
        args.source,
        args.target,
        train_transform,
        val_transform,
        train_target_transform=train_transform,
    )
    common_loader_args = dict(
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == 'cuda',
    )
    source_loader = DataLoader(source_dataset, **common_loader_args)
    target_loader = DataLoader(target_dataset, **common_loader_args)
    return source_loader, target_loader, num_classes


def _build_classifier(args: argparse.Namespace, num_classes: int, device: torch.device) -> nn.Module:
    """Restore a classifier from the checkpoint provided by the user."""

    backbone = utils.get_model(args.arch, pretrain=not args.scratch)
    pool_layer = nn.Identity() if args.no_pool else None
    classifier = ImageClassifier(
        backbone,
        num_classes,
        bottleneck_dim=args.bottleneck_dim,
        pool_layer=pool_layer,
        finetune=not args.scratch,
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    classifier.load_state_dict(checkpoint)
    classifier.eval()

    feature_extractor = nn.Sequential(
        classifier.backbone,
        classifier.pool_layer,
        classifier.bottleneck,
    )
    feature_extractor.to(device)
    return feature_extractor


def _limit_samples(feature: torch.Tensor, limit: int) -> torch.Tensor:
    """Restrict the number of features returned by ``collect_feature``."""

    if limit is None:
        return feature
    if feature.size(0) <= limit:
        return feature
    return feature[:limit]


def main() -> None:
    args = parse_args()

    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA device requested but not available on this machine.')

    source_loader, target_loader, num_classes = _create_dataloaders(args, device)
    feature_extractor = _build_classifier(args, num_classes, device)

    source_features = collect_feature(
        source_loader, feature_extractor, device, max_num_features=args.max_batches)
    target_features = collect_feature(
        target_loader, feature_extractor, device, max_num_features=args.max_batches)

    if args.max_samples is not None:
        source_features = _limit_samples(source_features, args.max_samples)
        target_features = _limit_samples(target_features, args.max_samples)

    if args.output is None:
        source_name = '-'.join(args.source)
        target_name = '-'.join(args.target)
        output_path = f'tsne_{source_name}_to_{target_name}.pdf'
    else:
        output_path = args.output

    output_dir = osp.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    tsne.visualize(
        source_features,
        target_features,
        output_path,
        source_color=args.source_color,
        target_color=args.target_color,
        random_state=args.random_state,
    )

    print(f'Saved t-SNE visualization to {output_path}')


if __name__ == '__main__':
    main()

