"""Pair files of lidar_image_process, preprocessing and train / val / test splits.

Pair files: <PAIRS_ROOT>/<seq>/pairs/<A>_<B>.npz (frame ids = rows of trajectory.csv), from
make_pairs.py: pairs chosen by the odometry positions, 1-15 m apart, any heading (revisits too).
<PAIRS_ROOT>/<seq>/pairs/index.csv lists every pair tried, with the positions of A and B, their
rotation and the share of A with a correspondence; read_index keeps the pairs that were saved.

A split file (JSON, see splits/make_splits.py) lists the pairs of every split by name 'seq/AAAAAA_BBBBBB'.
"""
import csv
import json
from pathlib import Path

import cv2
import numpy as np

PAIRS_ROOT = Path('/home/wenda/repositories/vf_learning/lidar_image_process/outputs')

# viewpoint change of a pair, to report results per group
DIST_BINS = (1, 3, 6, 10, 15)                    # [m] horizontal distance between A and B (make_pairs.dist_bins)
FACING = (('same', 0, 45), ('side', 45, 135), ('opposite', 135, 181))   # [deg] rotation between A and B


# ======================================== pair files

def pair_dir(seq, pairs_root=PAIRS_ROOT):
    return Path(pairs_root) / seq / 'pairs'


def depth_edges(rng, tol):
    """(H, W) bool: pixels with a 4-neighbour (columns wrap) whose range differs by more than tol
    (relative); pixels without a return (range 0) do not count."""
    edge = np.zeros(rng.shape, dtype=bool)
    for axis in (0, 1):
        a = rng if axis == 1 else rng[:-1]
        b = np.roll(rng, -1, axis=1) if axis == 1 else rng[1:]
        jump = (a > 0) & (b > 0) & (np.maximum(a, b) > (1 + tol) * np.minimum(a, b))
        if axis == 1:
            edge |= jump | np.roll(jump, 1, axis=1)
        else:
            edge[:-1] |= jump
            edge[1:] |= jump
    return edge


def load_pair(path, corr_min_range=0., edge_tol=0.05, max_cycle_err=1.0):
    """Load a pair file; correspondences of points closer than corr_min_range [m], of pixels next
    to a depth edge (a neighbour > edge_tol nearer or farther, None = keep), and with an A -> B -> A
    round-trip error above max_cycle_err [px] (None = keep all saved, up to make_pairs' cycle_tol) are set to NaN.

    The edge rule matters for keypoints between pixels: their ground truth interpolates the
    correspondences of the 4 neighbouring pixels, which is wrong if they lie on different surfaces
    (with scans metres apart the parallax between them is large); with one neighbour NaN they get none.
    The round-trip error grows with the scale change between the scans: 1 px keeps about half of the
    correspondences of pairs 6-15 m apart, 2 px most of them, less precisely.
    """
    z = np.load(path)
    pair = {k: z[k] for k in z.files}
    for ab, a in (('AB', 'A'), ('BA', 'B')):
        drop = ~(pair[f'corr_range_{ab}'] >= corr_min_range) if corr_min_range > 0 else np.zeros(pair[f'range_{a}'].shape, bool)
        if edge_tol is not None:
            drop |= depth_edges(pair[f'range_{a}'].astype(np.float64), edge_tol)
        if max_cycle_err is not None and f'cycle_err_{ab}' in pair:  # older pair files: cycle-checked at 1 px
            drop |= ~(pair[f'cycle_err_{ab}'] <= max_cycle_err)
        pair[f'corr_{ab}'] = np.where(drop[..., None], np.nan, pair[f'corr_{ab}'])
    return pair


def dist_bin(dist):
    """Label of the distance bin, e.g. '3-6m' ('>15m' beyond the last edge)."""
    for lo, hi in zip(DIST_BINS[:-1], DIST_BINS[1:]):
        if dist < hi:
            return f'{lo}-{hi}m'
    return f'>{DIST_BINS[-1]}m'


def facing(rot_deg):
    return next(name for name, lo, hi in FACING if lo <= rot_deg < hi)


def read_index(seq, pairs_root=PAIRS_ROOT):
    """The saved pairs of a sequence (index.csv, result 'ok'): list of dicts with seq, name, path,
    frame_A, frame_B, xy_A, xy_B, dist (horizontal [m]), rot_deg, match_ratio, dist_bin, facing."""
    pairs = []
    with open(pair_dir(seq, pairs_root) / 'index.csv') as f:
        for r in csv.DictReader(f):
            if r['result'] != 'ok':
                continue
            a, b = int(r['frame_A']), int(r['frame_B'])
            xy_a = np.array([float(r['x_A']), float(r['y_A'])])
            xy_b = np.array([float(r['x_B']), float(r['y_B'])])
            dist, rot = float(np.linalg.norm(xy_a - xy_b)), float(r['rot_deg'])
            pairs.append(dict(seq=seq, name=f'{seq}/{a:06d}_{b:06d}', path=pair_dir(seq, pairs_root) / f'{a:06d}_{b:06d}.npz',
                              frame_A=a, frame_B=b, xy_A=xy_a, xy_B=xy_b, dist=dist, rot_deg=rot,
                              match_ratio=float(r['match_ratio']), dist_bin=dist_bin(dist), facing=facing(rot)))
    return pairs


# ======================================== images

def preprocess(img16, valid=None, mode='linear', gamma=1.0, clahe_clip=2.0):
    """16-bit intensity image -> HxWx3 uint8 RGB, the input of ALIKE (it divides by 255).

    :param img16: (H, W) uint16 signal or reflectivity image
    :param valid: (H, W) bool, pixels with a lidar return (used for the percentile stretch)
    :param mode: 'linear'     : x / 65535 (best zero-shot, see evaluation/outputs/baseline_sweep.csv)
                 'percentile' : stretch the 1st..99th percentile of the valid pixels to [0, 1]
                 'clahe'      : percentile stretch, then CLAHE (local contrast equalisation)
    :param gamma: applied after the stretch, x ** gamma (< 1 brightens dark regions)
    """
    x = img16.astype(np.float32)
    if mode == 'linear':
        x = x / 65535.
    elif mode in ('percentile', 'clahe'):
        v = x[valid] if valid is not None and valid.any() else x.ravel()
        lo, hi = np.percentile(v, [1, 99])
        x = np.clip((x - lo) / max(hi - lo, 1.), 0, 1)
    else:
        raise ValueError(f'unknown preprocess mode: {mode}')
    if gamma != 1.0:
        x = x ** gamma
    img8 = (x * 255).round().astype(np.uint8)
    if mode == 'clahe':
        img8 = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(4, 16)).apply(img8)
    return np.repeat(img8[:, :, None], 3, axis=2)


# ======================================== splits

def load_split(split_file):
    return json.loads(Path(split_file).read_text())


def split_pairs(split_file, split):
    """The pairs of `split` (read_index dicts), in the order of the split file."""
    spec = load_split(split_file)
    names = spec['splits'][split]
    index = {p['name']: p for seq in sorted({n.split('/')[0] for n in names})
             for p in read_index(seq, spec['pairs_root'])}
    return [index[n] for n in names]
