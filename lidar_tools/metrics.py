"""Detection and matching metrics of keypoints on a lidar image pair against the ground truth.

Keypoints are (N, 2) (x=col, y=row) pixel coordinates. A keypoint is covisible if it has a
ground-truth location in the other image (geometry.gt_warp). Definitions (as in ALIKE / D2-Net):

detection (independent of the descriptors)
    rep@t       repeatability: fraction of the covisible keypoints with a keypoint of the other
                image within t px of their ground-truth location (mean of A->B and B->A)
    num_rep@3   number of repeated keypoints (A->B and B->A averaged)
    loc_err     median distance [px] of the repeated keypoints (@3 px) to their ground-truth
                location: how precisely a detected point is re-detected (sub-pixel accuracy)
    cov_ratio   covisible / detected keypoints: how many detections land on points with ground truth
matching (mutual nearest neighbours of the descriptors)
    MMA@t       fraction of the verifiable matches (A keypoint covisible) within t px (precision)
    num_correct@3, matching_score@3 = correct matches / mean number of covisible keypoints
"""
import numpy as np

from lidar_tools.geometry import gt_warp, pixel_dist, wrap_col

THRESHOLDS = tuple(range(1, 11))  # [px]
LOC_TH = 3.0                      # [px] a keypoint counts as repeated within this distance


def mutual_nn_match(desc0, desc1):
    """Mutual nearest neighbours of L2-normalised descriptors -> (K, 2) index pairs, similarity."""
    if len(desc0) == 0 or len(desc1) == 0:
        return np.zeros((0, 2), int), np.zeros(0)
    sim = desc0 @ desc1.T
    nn01 = sim.argmax(1)
    nn10 = sim.argmax(0)
    ids0 = np.nonzero(nn10[nn01] == np.arange(len(desc0)))[0]
    return np.stack([ids0, nn01[ids0]], 1), sim[ids0, nn01[ids0]]


def nearest_dist(kpts, warped, W):
    """Distance from each warped keypoint (NaN = no ground truth) to the nearest of `kpts`."""
    d = np.full(len(warped), np.nan)
    ok = np.isfinite(warped).all(1)
    if ok.any() and len(kpts):
        d[ok] = pixel_dist(warped[ok], kpts, W).min(1)
    return d


def evaluate_pair(kpts0, desc0, kpts1, desc1, corr01, corr10):
    """:return: (metrics dict, details dict with per-keypoint and per-match arrays for the figures)"""
    W = corr01.shape[1]
    w01 = gt_warp(kpts0, corr01)  # kpts0 -> B
    w10 = gt_warp(kpts1, corr10)  # kpts1 -> A
    d0 = nearest_dist(kpts1, w01, W)  # per A keypoint: distance to the nearest B keypoint, NaN = no GT
    d1 = nearest_dist(kpts0, w10, W)
    cov0, cov1 = np.isfinite(d0), np.isfinite(d1)

    m = {'num_kpts': (len(kpts0) + len(kpts1)) / 2, 'num_cov': (cov0.sum() + cov1.sum()) / 2}
    m['cov_ratio'] = m['num_cov'] / max(m['num_kpts'], 1)

    # ---------- detection
    for t in THRESHOLDS:
        r0 = (d0[cov0] <= t).mean() if cov0.any() else 0.
        r1 = (d1[cov1] <= t).mean() if cov1.any() else 0.
        m[f'rep@{t}'] = (r0 + r1) / 2
    rep_d = np.concatenate([d0[cov0 & (d0 <= LOC_TH)], d1[cov1 & (d1 <= LOC_TH)]])
    m['num_rep@3'] = len(rep_d) / 2
    m['loc_err'] = float(np.median(rep_d)) if len(rep_d) else np.nan

    # ---------- matching
    matches, sim = mutual_nn_match(desc0, desc1)
    gt = w01[matches[:, 0]]
    verifiable = np.isfinite(gt).all(1)
    err = np.full(len(matches), np.nan)
    if verifiable.any():
        p1 = kpts1[matches[verifiable, 1]]
        err[verifiable] = np.hypot(wrap_col(p1[:, 0] - gt[verifiable, 0], W), p1[:, 1] - gt[verifiable, 1])
    e = err[verifiable]
    m['num_matches'] = len(matches)
    m['num_verifiable'] = int(verifiable.sum())
    for t in THRESHOLDS:
        m[f'MMA@{t}'] = (e <= t).mean() if len(e) else 0.
    m['num_correct@3'] = int((e <= 3).sum())
    m['matching_score@3'] = m['num_correct@3'] / max(m['num_cov'], 1)

    details = {'d0': d0, 'd1': d1, 'w01': w01, 'w10': w10,
               'matches': matches, 'sim': sim, 'err': err, 'gt': gt}
    return m, details


def mean_metrics(metrics_list):
    """Mean over pairs of each metric (loc_err: mean of the per-pair medians, ignoring NaN)."""
    keys = metrics_list[0].keys()
    return {k: float(np.nanmean([m[k] for m in metrics_list])) for k in keys}
