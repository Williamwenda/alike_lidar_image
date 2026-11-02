"""Keypoint extractors for inference: one interface for every method evaluated.

    make_model(name, weights) -> extractor; extract(extractor, img) -> keypoints, descriptors, scores, score map

name:
    'orb'                          OpenCV ORB (binary descriptors as +-1 vectors, see OrbExtractor)
    'alike-t' / '-s' / '-n' / '-l' ALIKE, pretrained (weights None) or fine-tuned (.pth from train_lidar.py)
    'xfeat'                        XFeat, pretrained (xfeat_lidar_image/weights/xfeat.pt) or a XFeatModel .pth
    'xfeat-lidar'                  XFeat with keypoint uncertainty heads (xfeat_lidar_image, IntensityFeatureModel),
                                   fine-tuned .pth from xfeat_lidar_image/modules/training/train_lidar.py
Descriptors are L2-normalised, so every method is matched the same way (lidar_tools.metrics.mutual_nn_match).
"""
import sys

import numpy as np
import cv2
import torch

from lidar_tools import REPO_ROOT

from alike import ALike, configs

XFEAT_ROOT = REPO_ROOT.parent / 'xfeat_lidar_image'

# ORB settings for 128 x 1024 lidar images, chosen on the val split (evaluation/eval_orb.py): the best
# repeatability @3 px (0.54) with 8 pyramid levels and a 15 px patch / border (the default 31 px leaves
# out the top and bottom 31 of the 128 rows); the FAST threshold hardly matters with top_k 500
ORB_PARAMS = dict(scaleFactor=1.2, nlevels=8, edgeThreshold=15, patchSize=15, fastThreshold=10)


def _path(weights):
    return str(weights if str(weights).startswith('/') else REPO_ROOT / weights)


class AlikeExtractor:
    def __init__(self, name, weights=None, top_k=500, scores_th=0.2, n_limit=5000, device='cuda'):
        cfg = dict(configs[name])
        if weights is not None:
            cfg['model_path'] = _path(weights)
        self.net = ALike(**cfg, device=device, top_k=top_k, scores_th=scores_th, n_limit=n_limit)
        self.device = device

    def extract(self, img_rgb, sub_pixel=True):
        pred = self.net(img_rgb, sub_pixel=sub_pixel)
        return pred['keypoints'], pred['descriptors'], pred['scores'], pred['scores_map'][0, 0]


class OrbExtractor:
    """ORB keypoints; the 256-bit descriptors become +-1 / 16 vectors, whose dot product is
    1 - 2 * Hamming / 256: the nearest neighbour by dot product is the nearest by Hamming distance."""

    def __init__(self, top_k=500, **params):
        self.orb = cv2.ORB_create(nfeatures=top_k, **{**ORB_PARAMS, **params})
        self.device = 'cpu'

    def extract(self, img_rgb, sub_pixel=True):
        kps, des = self.orb.detectAndCompute(np.ascontiguousarray(img_rgb[..., 0]), None)
        if des is None or len(kps) == 0:
            return np.zeros((0, 2), np.float32), np.zeros((0, 256), np.float32), np.zeros(0, np.float32), None
        bits = np.unpackbits(des, axis=1).astype(np.float32) * 2 - 1
        kpts = np.array([k.pt for k in kps], np.float32)
        return kpts, bits / np.sqrt(bits.shape[1]), np.array([k.response for k in kps], np.float32), None


class XFeatExtractor:
    """XFeat (sparse keypoints, 64-d descriptors) or its lidar version with keypoint uncertainty
    ('xfeat-lidar'; sub-pixel keypoints from the mu head, covariances in self.last_covariances)."""

    def __init__(self, name='xfeat', weights=None, top_k=500, detection_threshold=0.05, device='cuda'):
        if str(XFEAT_ROOT) not in sys.path:
            sys.path.insert(0, str(XFEAT_ROOT))
        if name == 'xfeat':
            from modules.xfeat import XFeat
            self.net = XFeat(weights=_path(weights) if weights else str(XFEAT_ROOT / 'weights' / 'xfeat.pt'),
                             top_k=top_k, detection_threshold=detection_threshold)
        else:
            from modules.intensity_xfeat import IntensityXFeat
            if weights is None:
                raise ValueError("'xfeat-lidar' needs fine-tuned weights (its uncertainty heads are not pretrained)")
            # one keypoint per 8x8 cell without suppressing neighbouring cells: the default 3x3 cell NMS
            # leaves ~90 keypoints on a 128 x 1024 image (val: rep@3 0.21 -> 0.52, as in train_lidar.py)
            self.net = IntensityXFeat(weights=_path(weights), top_k=top_k, detection_threshold=detection_threshold,
                                      nms_cells=1)
        self.net.dev = torch.device(device)
        self.net.net.to(self.net.dev).eval()
        self.device = device
        self.last_covariances = None

    def extract(self, img_rgb, sub_pixel=True):
        out = self.net.detectAndCompute(img_rgb[..., :1])[0]  # gray (H, W, 1); XFeat normalises the input itself
        self.last_covariances = out['covariances'].cpu().numpy() if 'covariances' in out else None
        return (out['keypoints'].cpu().numpy(), out['descriptors'].cpu().numpy(), out['scores'].cpu().numpy(), None)


def make_model(name, weights=None, top_k=500, scores_th=0.2, n_limit=5000, device='cuda'):
    """:param name: 'orb', 'alike-t', 'alike-s', 'alike-n', 'alike-l', 'xfeat' or 'xfeat-lidar'
    :param weights: .pth state dict (e.g. exported from a fine-tuning run), None = pretrained
    :param top_k: the top_k strongest keypoints (ALIKE: -1 = all with score > scores_th)
    """
    if name == 'orb':
        return OrbExtractor(top_k)
    if name.startswith('alike'):
        return AlikeExtractor(name, weights, top_k, scores_th, n_limit, device)
    if name in ('xfeat', 'xfeat-lidar'):
        return XFeatExtractor(name, weights, top_k, device=device)
    raise ValueError(f'unknown method {name}')


def extract(model, img_rgb, sub_pixel=True):
    """img_rgb: HxWx3 uint8 -> keypoints (N, 2) (x, y) px, descriptors (N, D) L2-normalised, scores (N,),
    score map (H, W) or None (ORB, XFeat)."""
    return model.extract(img_rgb, sub_pixel)
