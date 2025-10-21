# ---------------------------------------------------------------
# Copyright (c) 2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

import random

import torch
from torch.nn import Module

from mmseg.models.uda.teacher_module import EMATeacher
from mmseg.models.utils.dacs_transforms import get_mean_std, strong_transform
from mmseg.models.utils.masking_transforms import build_mask_generator
from mmseg.models.utils import MaskCollator as IJEPAMaskCollator
from mmseg.models.utils import VisionTransformerPredictor


class MaskingConsistencyModule(Module):

    def __init__(self, require_teacher, cfg):
        super(MaskingConsistencyModule, self).__init__()

        self.source_only = cfg.get('source_only', False)
        self.max_iters = cfg['max_iters']
        self.color_jitter_s = cfg['color_jitter_strength']
        self.color_jitter_p = cfg['color_jitter_probability']

        self.mask_mode = cfg['mask_mode']
        self.mask_alpha = cfg['mask_alpha']
        self.mask_pseudo_threshold = cfg['mask_pseudo_threshold']
        self.mask_lambda = cfg['mask_lambda']
        self.mask_gen = build_mask_generator(cfg['mask_generator'])

        self.ijepa_cfg = cfg.get('ijepa')
        self.ijepa_predictor = None
        self.ijepa_collator = None
        self.ijepa_patch_size = None
        self.ijepa_trainable = bool(self.ijepa_cfg.get('trainable', False)) \
            if self.ijepa_cfg is not None else False

        assert self.mask_mode in [
            'separate', 'separatesrc', 'separatetrg', 'separateaug',
            'separatesrcaug', 'separatetrgaug'
        ]

        self.teacher = None
        if require_teacher or \
                self.mask_alpha != 'same' or \
                self.mask_pseudo_threshold != 'same':
            self.teacher = EMATeacher(use_mask_params=True, cfg=cfg)

        self.debug = False
        self.debug_output = {}

    def update_weights(self, model, iter):
        if self.teacher is not None:
            self.teacher.update_weights(model, iter)

    def update_debug_state(self):
        if self.teacher is not None:
            self.teacher.debug = self.debug

    def __call__(self,
                 model,
                 img,
                 img_metas,
                 gt_semantic_seg,
                 target_img,
                 target_img_metas,
                 valid_pseudo_mask,
                 pseudo_label=None,
                 pseudo_weight=None):
        self.update_debug_state()
        self.debug_output = {}
        model.debug_output = {}
        dev = img.device
        means, stds = get_mean_std(img_metas, dev)

        if self.ijepa_cfg is not None and target_img is not None:
            if self.ijepa_predictor is None:
                predictor_cfg = dict(self.ijepa_cfg.get('predictor', {}))
                predictor_cfg.setdefault('img_size', tuple(target_img.shape[-2:]))
                patch_size = predictor_cfg.get(
                    'patch_size', self.ijepa_cfg.get('patch_size', 16))
                predictor_cfg['patch_size'] = patch_size
                self.ijepa_predictor = VisionTransformerPredictor(**predictor_cfg).to(dev)
                if not self.ijepa_trainable:
                    self.ijepa_predictor.eval()
                if isinstance(patch_size, int):
                    self.ijepa_patch_size = (patch_size, patch_size)
                else:
                    self.ijepa_patch_size = tuple(patch_size)

            if self.ijepa_collator is None:
                collator_cfg = dict(self.ijepa_cfg.get('mask_collator', {}))
                collator_cfg.setdefault('input_size', tuple(target_img.shape[-2:]))
                collator_cfg.setdefault('patch_size', self.ijepa_patch_size)
                self.ijepa_collator = IJEPAMaskCollator(**collator_cfg)

            mask_dict = self.ijepa_collator(target_img)
            ijepa_mask = mask_dict['mask']
            if self.ijepa_trainable:
                target_img = self.ijepa_predictor(target_img, ijepa_mask)
            else:
                with torch.no_grad():
                    target_img = self.ijepa_predictor(target_img, ijepa_mask)

            if self.debug:
                upsampled_mask = self.ijepa_collator.upsample(
                    ijepa_mask, (target_img.shape[-2], target_img.shape[-1])
                )
                prediction = self.ijepa_predictor.last_prediction
                self.debug_output['IJepa'] = {
                    'Mask': upsampled_mask.detach().cpu().numpy(),
                    'Prediction': None if prediction is None else
                    prediction.cpu().numpy(),
                }

        if not self.source_only:
            # Share the pseudo labels with the host UDA method
            if self.teacher is None:
                assert self.mask_alpha == 'same'
                assert self.mask_pseudo_threshold == 'same'
                assert pseudo_label is not None
                assert pseudo_weight is not None
                masked_plabel = pseudo_label
                masked_pweight = pseudo_weight
            # Use a separate EMA teacher for MIC
            else:
                masked_plabel, masked_pweight = \
                    self.teacher(
                        target_img, target_img_metas, valid_pseudo_mask)
                if self.debug:
                    self.debug_output['Mask Teacher'] = {
                        'Img': target_img.detach(),
                        'Pseudo Label': masked_plabel.cpu().numpy(),
                        'Pseudo Weight': masked_pweight.cpu().numpy(),
                    }
        # Don't use target images at all
        if self.source_only:
            masked_img = img
            masked_lbl = gt_semantic_seg
            b, _, h, w = gt_semantic_seg.shape
            masked_seg_weight = None
        # Use 1x source image and 1x target image for MIC
        elif self.mask_mode in ['separate', 'separateaug']:
            assert img.shape[0] == 2
            masked_img = torch.stack([img[0], target_img[0]])
            masked_lbl = torch.stack(
                [gt_semantic_seg[0], masked_plabel[0].unsqueeze(0)])
            gt_pixel_weight = torch.ones(masked_pweight[0].shape, device=dev)
            masked_seg_weight = torch.stack(
                [gt_pixel_weight, masked_pweight[0]])
        # Use only source images for MIC
        elif self.mask_mode in ['separatesrc', 'separatesrcaug']:
            masked_img = img
            masked_lbl = gt_semantic_seg
            masked_seg_weight = None
        # Use only target images for MIC
        elif self.mask_mode in ['separatetrg', 'separatetrgaug']:
            masked_img = target_img
            masked_lbl = masked_plabel.unsqueeze(1)
            masked_seg_weight = masked_pweight
        else:
            raise NotImplementedError(self.mask_mode)

        # Apply color augmentation
        if 'aug' in self.mask_mode:
            strong_parameters = {
                'mix': None,
                'color_jitter': random.uniform(0, 1),
                'color_jitter_s': self.color_jitter_s,
                'color_jitter_p': self.color_jitter_p,
                'blur': random.uniform(0, 1),
                'mean': means[0].unsqueeze(0),
                'std': stds[0].unsqueeze(0)
            }
            masked_img, _ = strong_transform(
                strong_parameters, data=masked_img.clone())

        # Apply masking to image
        masked_img = self.mask_gen.mask_image(masked_img)

        # Train on masked images
        masked_loss = model.forward_train(
            masked_img,
            img_metas,
            masked_lbl,
            seg_weight=masked_seg_weight,
        )
        if self.mask_lambda != 1:
            masked_loss['decode.loss_seg'] *= self.mask_lambda

        if self.debug:
            self.debug_output['Masked'] = model.debug_output
            if masked_seg_weight is not None:
                self.debug_output['Masked']['PL Weight'] = \
                    masked_seg_weight.cpu().numpy()

        return masked_loss
