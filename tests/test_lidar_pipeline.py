"""Checks of the lidar fine-tuning pipeline on real pairs (needs the lidar_image_process outputs).

    uv run python tests/test_lidar_pipeline.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'training_code'))

from lidar_tools.pairs import split_pairs, load_pair  # noqa: E402
from lidar_tools.geometry import gt_warp, wrap_col  # noqa: E402
from utils import warp_dense  # noqa: E402  (training_code/utils)
from datasets.lidar_pairs import LidarPairDataset  # noqa: E402
from lidar_tools.pose import pose_metrics  # noqa: E402

SPLIT = ROOT / 'splits' / 'default.json'


def test_warp_dense_matches_numpy():
    """torch warp_dense (training) == numpy gt_warp (evaluation), also next to the seam and for scans
    facing opposite ways (column flow near +-W/2); warped keypoints land next to the correspondence of
    their nearest pixel; grad = identity."""
    rng = np.random.default_rng(0)
    val = split_pairs(SPLIT, 'val')
    opposite = [p for p in val if p['facing'] == 'opposite']
    for it in (val[10], max(val, key=lambda p: p['dist']), max(opposite, key=lambda p: p['rot_deg'])):
        corr = load_pair(it['path'], 3.0)['corr_AB']
        H, W = corr.shape[:2]
        k = np.concatenate([np.stack([rng.uniform(0, W - 1, 20000), rng.uniform(0, H - 1, 20000)], 1),
                            np.stack([rng.uniform(W - 3, W - 1, 2000), rng.uniform(0, H - 1, 2000)], 1),
                            np.stack([rng.integers(0, W, 2000), rng.integers(0, H, 2000)], 1).astype(float)])
        ref = gt_warp(k, corr)
        near = corr[np.clip(np.rint(k[:, 1]).astype(int), 0, H - 1), np.rint(k[:, 0]).astype(int) % W]  # (row, col)
        ok = np.isfinite(ref).all(1)
        jump = np.hypot(wrap_col(ref[ok, 0] - near[ok, 1], W), ref[ok, 1] - near[ok, 0]).max()
        # (a few px from magnification; ~W/2 if the column wrap is interpolated wrongly, tens of px across a
        # depth edge, which load_pair removes)
        assert jump < 20, f'a warped keypoint is {jump:.1f} px from the correspondence of its nearest pixel'
        kt = torch.tensor(k, dtype=torch.float32, requires_grad=True)
        k0, k01, ids, ids_out = warp_dense(kt, {'corr': torch.from_numpy(corr)})
        ids = ids.numpy()
        # training also drops points warped beyond the pixel centres of the first / last row and
        # between the last and the first column (ALIKE samples the other image's maps in [0, W-1] x [0, H-1])
        ref_ok = np.isfinite(ref).all(1) & (ref[:, 0] <= W - 1) & (ref[:, 1] >= 0) & (ref[:, 1] <= H - 1)
        assert set(ids) == set(np.nonzero(ref_ok)[0]), 'valid keypoints differ'
        d = np.abs(np.stack([wrap_col(k01.detach().numpy()[:, 0] - ref[ids, 0], W),
                             k01.detach().numpy()[:, 1] - ref[ids, 1]], 1)).max()
        assert d < 1e-3, f'warped keypoints differ by {d} px'
        k01.sum().backward()
        g = kt.grad.numpy()
        assert np.allclose(g[ids], 1) and np.allclose(np.delete(g, ids, 0), 0), 'gradient is not the identity'
        assert len(ids_out) == 0
        print(f'warp_dense == gt_warp ({it["name"]}, {it["dist"]:.1f} m, {it["rot_deg"]:.0f} deg; '
              f'max {jump:.1f} px from the nearest pixel): {len(ids)} valid keypoints, max diff {d:.1e} px, grad ok')


def test_roll_keeps_correspondences():
    """After the column roll, every rolled correspondence still points at the same image content."""
    ds = LidarPairDataset(SPLIT, 'val', augment=True)
    pair = load_pair(ds.items[5]['path'], 3.0)
    img0 = np.repeat(pair['signal_A'][..., None].astype(np.float32), 3, 2)
    img1 = np.repeat(pair['signal_B'][..., None].astype(np.float32), 3, 2)
    ds.rng = np.random.default_rng(1)
    r0, r1, c01, c10 = ds.roll_columns(img0, img1, pair['corr_AB'], pair['corr_BA'])
    H, W = c01.shape[:2]
    for (a, b, c), (a_orig, b_orig, c_orig) in (((r0, r1, c01), (img0, img1, pair['corr_AB'])),
                                                ((r1, r0, c10), (img1, img0, pair['corr_BA']))):
        rows, cols = np.nonzero(np.isfinite(c).all(-1))
        ri, ci = np.clip(np.rint(c[rows, cols, 0]).astype(int), 0, H - 1), np.rint(c[rows, cols, 1]).astype(int) % W
        # content of the rolled source pixel and its rolled match, vs the same in the original pair
        src_val, dst_val = a[rows, cols, 0], b[ri, ci, 0]
        rows_o, cols_o = np.nonzero(np.isfinite(c_orig).all(-1))
        ri_o = np.clip(np.rint(c_orig[rows_o, cols_o, 0]).astype(int), 0, H - 1)
        ci_o = np.rint(c_orig[rows_o, cols_o, 1]).astype(int) % W
        assert len(rows) == len(rows_o)
        assert np.array_equal(np.sort(src_val), np.sort(a_orig[rows_o, cols_o, 0]))
        assert np.array_equal(np.sort(src_val * 1e5 + dst_val),
                              np.sort(a_orig[rows_o, cols_o, 0] * 1e5 + b_orig[ri_o, ci_o, 0]))
    print(f'column roll keeps every correspondence ({len(rows)} matched pixels)')


def test_dataset_item():
    ds = LidarPairDataset(SPLIT, 'train', augment=True)
    item = ds[100]
    assert item['image0'].shape == (3, 128, 1024) and item['image0'].dtype == torch.float32
    assert 0 <= item['image0'].min() and item['image0'].max() <= 1
    assert item['warp01_params']['corr'].shape == (128, 1024, 2)
    assert item['dist_bin'] in ('1-3m', '3-6m', '6-10m', '10-15m') and item['facing'] in ('same', 'side', 'opposite')
    ev = LidarPairDataset(SPLIT, 'val')
    print(f'dataset: {len(ds)} training pairs, {len(ev)} val pairs; item {item["name"]} ok')


def test_pose_from_ground_truth():
    """Relative pose from ground-truth correspondences recovers the ICP pose when the within-scan motion is
    compensated (conventions, 180 deg rotation, re-distortion), and is clearly worse without."""
    rng = np.random.default_rng(0)
    val = split_pairs(SPLIT, 'val')
    for it in (val[10], max(val, key=lambda p: p['dist'])):
        pair = load_pair(it['path'], 3.0)
        rows, cols = np.nonzero(np.isfinite(pair['corr_AB'][..., 0]))
        sel = rng.choice(len(rows), min(300, len(rows)), replace=False)
        k0 = np.stack([cols[sel], rows[sel]], 1).astype(float)
        k1 = pair['corr_AB'][rows[sel], cols[sel]][:, ::-1].astype(float)  # (row, col) -> (x, y)
        matches = np.stack([np.arange(len(sel))] * 2, 1)
        m = pose_metrics(it, pair, k0, k1, matches)
        raw = pose_metrics(it, pair, k0, k1, matches, {'compensate': False})
        assert m['t_err'] < 0.02 and m['R_err'] < 0.1, f'pose from ground truth off by {m["t_err"]:.3f} m / {m["R_err"]:.3f} deg'
        print(f'pose from ground truth ({it["name"]}, {it["dist"]:.1f} m): {100 * m["t_err"]:.1f} cm / {m["R_err"]:.3f} deg '
              f'({m["pose_inliers"]} inliers); without motion compensation {100 * raw["t_err"]:.1f} cm / {raw["R_err"]:.3f} deg')


if __name__ == '__main__':
    test_pose_from_ground_truth()
    test_warp_dense_matches_numpy()
    test_roll_keeps_correspondences()
    test_dataset_item()
    print('all tests passed')
