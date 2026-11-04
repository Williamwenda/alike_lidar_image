"""Lidar intensity image pairs with dense ground-truth correspondences (from lidar_image_process).

Returns the dict of the other ALIKE datasets, with warp mode 'dense' (utils.warp_dense):
    image0, image1 : 3 x H x W float in [0, 1] (8-bit image / 255, exactly what ALike gets at inference)
    warp01_params  : {'mode': 'dense', 'width', 'height', 'corr': (H, W, 2) (row, col) in image1 of image0's pixels}
    warp10_params  : the same from image1 to image0
    name, dist_bin, facing : pair name and its viewpoint change (lidar_tools.pairs.DIST_BINS, FACING)
"""
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'lidar_image_nn_evaluation'))  # lidar_tools
from lidar_tools.pairs import split_pairs, load_pair, preprocess  # noqa: E402


class LidarPairDataset(Dataset):
    def __init__(self, split_file, split, channel='signal', preprocess_mode='linear',
                 corr_min_range=3.0, augment=False, max_pairs=None):
        """
        :param split_file: splits/*.json (see splits/make_splits.py)
        :param split: 'train', 'val', 'test'
        :param channel: 'signal' or 'reflec'
        :param corr_min_range: no ground truth for points closer than this [m]
        :param augment: random column roll, photometric jitter, swap of the two images
        :param max_pairs: evenly subsample to at most this many pairs (None = all)
        """
        self.channel, self.preprocess_mode = channel, preprocess_mode
        self.corr_min_range, self.augment = corr_min_range, augment
        self.items = split_pairs(split_file, split)
        if max_pairs is not None and len(self.items) > max_pairs:
            self.items = [self.items[i] for i in np.linspace(0, len(self.items) - 1, max_pairs).round().astype(int)]
        self.rng = np.random.default_rng(0)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        if self.augment:
            # torch seeds every worker differently, and differently every epoch
            self.rng = np.random.default_rng(torch.initial_seed() + idx)
        pair = load_pair(item['path'], self.corr_min_range)

        img0 = preprocess(pair[f'{self.channel}_A'], pair['range_A'] > 0, self.preprocess_mode)
        img1 = preprocess(pair[f'{self.channel}_B'], pair['range_B'] > 0, self.preprocess_mode)
        img0, img1 = img0.astype(np.float32) / 255., img1.astype(np.float32) / 255.
        corr01, corr10 = pair['corr_AB'], pair['corr_BA']

        if self.augment:
            if self.rng.random() < 0.5:
                img0, img1, corr01, corr10 = img1, img0, corr10, corr01
            img0, img1, corr01, corr10 = self.roll_columns(img0, img1, corr01, corr10)
            img0, img1 = self.photometric(img0), self.photometric(img1)

        H, W = img0.shape[:2]
        return {'image0': torch.from_numpy(np.ascontiguousarray(img0.transpose(2, 0, 1))),
                'image1': torch.from_numpy(np.ascontiguousarray(img1.transpose(2, 0, 1))),
                'warp01_params': {'mode': 'dense', 'width': W, 'height': H,
                                  'corr': torch.from_numpy(np.ascontiguousarray(corr01, dtype=np.float32))},
                'warp10_params': {'mode': 'dense', 'width': W, 'height': H,
                                  'corr': torch.from_numpy(np.ascontiguousarray(corr10, dtype=np.float32))},
                'name': item['name'], 'dist_bin': item['dist_bin'], 'facing': item['facing']}

    def roll_columns(self, img0, img1, corr01, corr10):
        """Independent random column shift of each image (the scan is 360 deg, so this is a change of
        the azimuth origin): pixel j of the new image is pixel j - s of the old one."""
        W = img0.shape[1]
        s0, s1 = self.rng.integers(0, W, 2)
        img0, img1 = np.roll(img0, s0, axis=1), np.roll(img1, s1, axis=1)
        corr01, corr10 = np.roll(corr01, s0, axis=1).copy(), np.roll(corr10, s1, axis=1).copy()
        corr01[..., 1] = np.mod(corr01[..., 1] + s1, W)  # NaN stays NaN
        corr10[..., 1] = np.mod(corr10[..., 1] + s0, W)
        return img0, img1, corr01, corr10

    def photometric(self, img):
        """Gain, gamma and noise on a [0, 1] image (same for the 3 identical channels)."""
        g = img[..., 0]
        g = np.clip(g, 0, 1) ** self.rng.uniform(0.75, 1.33)
        g = g * self.rng.uniform(0.8, 1.2) + self.rng.normal(0, self.rng.uniform(0, 0.02), g.shape)
        g = np.clip(g, 0, 1).astype(np.float32)
        return np.repeat(g[..., None], 3, axis=2)
