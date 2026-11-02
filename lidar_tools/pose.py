"""Relative pose of a lidar scan pair from keypoint matches, with the motion distortion of both scans.

An Ouster scan is swept over 0.1 s while the robot moves: pixel (i, j) is measured at the firing time of
its measurement column c, in the sensor frame s(t_c). The continuous-time lidar odometry gives, for every
saved frame, the sensor pose at each column's time: col_T_sc_s[c] = T_{s(t_c), s(stamp)} (within-scan
motion, lidar_image_process/run_odometry.py). Only the transform between the two scans, T_AB, is unknown,
so no motion model is assumed between scans that are metres and minutes apart.

    lift      keypoint (sub-pixel, in the pair image) -> nearest raw pixel -> its point undistorted to the
              scan stamp, xyz_undist = inv(col_T_sc_s[c]) @ (range * ray of the pixel), and its column c
    predict   where scan B saw A's point:  y_B^ = f(col_T_sc_s_B[c_B] @ T_BA @ p_A)   (f: Ouster projection,
              c_B: column of the measured keypoint), and the same for A:  y_A^ = f(col_T_sc_s_A[c_A] @ T_AB @ p_B)
    RANSAC    3-point Kabsch on (p_A, p_B); a match is an inlier if its pixel error is <= inlier_px in both images
    refine    robust least squares of the pixel errors of the inliers, then the inliers again
Ground truth: T_AB of the pair file (ICP between the undistorted clouds; p_A = T_AB @ p_B at the scan stamps).
"""
import sys
import json

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from lidar_tools.pairs import PAIRS_ROOT, depth_edges

sys.path.append(str(PAIRS_ROOT.parent))  # lidar_image_process: one Ouster model for both repos
from core.ouster_model import OusterModel  # noqa: E402

POSE_PARAMS = dict(
    inlier_px=3.0,           # [px] max reprojection error of an inlier, in both images
    min_inliers=6,           # fewer inliers = failure (infinite pose error)
    max_iters=5000,          # RANSAC hypotheses at most (adaptive: stops at `confidence`)
    confidence=0.999,
    batch=256,               # hypotheses scored together (speed only)
    min_sample_spread=0.5,   # [m] the 3 points of a sample must be this far apart (degenerate otherwise)
    refine_rounds=2,         # least squares on the inliers, then new inliers, this many times
    huber_px=1.0,            # [px] Huber scale of the refinement
    min_range=3.0,           # [m] keypoints closer than this get no 3-D point (robot body; the operator walking next
                             # to the robot moves with the sensor; as the ground truth's 3 m)
    edge_tol=0.05,           # keypoints next to a range jump > 5 % get no 3-D point (None = keep)
    compensate=True,         # False: ablation, ignore the within-scan motion (raw points, one frame per scan)
    seed=0,
)
POSE_THRESHOLDS = ((0.1, 1.0), (0.25, 2.0), (0.5, 5.0))   # (translation [m], rotation [deg]) success thresholds


# ---------------- scans ----------------
_models, _frame_files = {}, {}


def ouster_model(seq):
    if seq not in _models:
        meta = json.loads((PAIRS_ROOT / seq / 'odometry' / 'meta.json').read_text())
        _models[seq] = OusterModel(meta['ouster_metadata'])
    return _models[seq]


def load_scan(seq, frame_id, edge_tol=0.05):
    """Odometry frame (lidar_image_process/run_odometry.py): undistorted points, range, within-scan motion."""
    if seq not in _frame_files:  # frame id = index in the sorted frame files (make_pairs.list_frames)
        _frame_files[seq] = sorted((PAIRS_ROOT / seq / 'odometry' / 'frames').glob('*.npz'), key=lambda f: int(f.stem))
    with np.load(_frame_files[seq][int(frame_id)]) as z:
        scan = {k: z[k] for k in ('xyz_undist', 'range', 'col_T_sc_s')}
    scan['edge'] = depth_edges(scan['range'].astype(np.float64), edge_tol) if edge_tol is not None \
        else np.zeros(scan['range'].shape, bool)
    return scan


def lift(kpts, scan, model, rotated180, params):
    """Keypoints (N, 2) (x, y) of the pair image -> raw pixel coordinates y (N, 2) (row, col), 3-D points p (N, 3)
    in the sensor frame at the scan stamp (motion undistorted), measurement columns c (N,), usable mask (N,)."""
    H, W = model.H, model.W
    x, y = kpts[:, 0].astype(np.float64), kpts[:, 1].astype(np.float64)
    if rotated180:  # pair images are rotated (upside-down mount): pixel (r, c) = raw (H-1-r, W-1-c)
        x, y = W - 1 - x, H - 1 - y
    x = np.mod(x, W)
    r = np.clip(np.rint(y).astype(np.int64), 0, H - 1)
    c = np.mod(np.rint(x).astype(np.int64), W)
    p = scan['xyz_undist'][r, c].astype(np.float64)
    cols = model.meas_col_of_pixel[r, c]
    ok = np.isfinite(p).all(1) & (scan['range'][r, c] > params['min_range']) & ~scan['edge'][r, c]
    if not params['compensate']:  # the measured point, as if the whole scan were taken at one time
        T = scan['col_T_sc_s'][cols]
        p = np.einsum('nij,nj->ni', T[:, :3, :3], p) + T[:, :3, 3]
    return np.stack([y, x], 1), p, cols, ok


# ---------------- geometry ----------------
def kabsch(src, dst):
    """Batched rigid fit dst = R src + t: src, dst (K, M, 3) -> T (K, 4, 4)."""
    cs, cd = src.mean(1, keepdims=True), dst.mean(1, keepdims=True)
    U, _, Vt = np.linalg.svd(np.einsum('kmi,kmj->kij', src - cs, dst - cd))
    D = np.ones(U.shape[:2])
    D[:, -1] = np.sign(np.linalg.det(np.einsum('kij,kjl->kil', U, Vt)))
    R = np.einsum('kji,kj,klj->kil', Vt, D, U)  # V diag(D) U^T
    T = np.tile(np.eye(4), (len(src), 1, 1))
    T[:, :3, :3] = R
    T[:, :3, 3] = cd[:, 0] - np.einsum('kij,kj->ki', R, cs[:, 0])
    return T


def pixel_errors(T, p_src, y_dst, cols_dst, colT_dst, model):
    """Pixel errors (K, N, 2) (row, col) of the points p_src (N, 3) moved by the hypotheses T (K, 4, 4) into the
    other scan's stamp frame, then into the frame of the column each was measured in (colT_dst (N, 4, 4))."""
    q = np.einsum('kij,nj->kni', T[:, :3, :3], p_src) + T[:, None, :3, 3]
    q = np.einsum('nij,knj->kni', colT_dst[:, :3, :3], q) + colT_dst[None, :, :3, 3]
    K, N = q.shape[:2]
    row, col = model.project(q.reshape(-1, 3), np.tile(cols_dst, K))
    W = model.W
    d_col = (col.reshape(K, N) - y_dst[None, :, 1] + W / 2) % W - W / 2
    return np.stack([row.reshape(K, N) - y_dst[None, :, 0], d_col], -1)


def inv(T):
    Ti = np.zeros_like(T)
    R = np.swapaxes(T[..., :3, :3], -1, -2)
    Ti[..., :3, :3] = R
    Ti[..., :3, 3] = -np.einsum('...ij,...j->...i', R, T[..., :3, 3])
    Ti[..., 3, 3] = 1
    return Ti


# ---------------- estimation ----------------
def estimate_pose(kpts0, kpts1, matches, scan_a, scan_b, model, rotated180=True, params=None):
    """T_AB (p_A = T_AB p_B) from mutual-NN matches (K, 2) between keypoints of A (kpts0) and B (kpts1).
    Returns dict: T_AB (4, 4) or None (failure), inliers and usable (bool over matches: a 3-D point in both
    scans), num_usable."""
    P = {**POSE_PARAMS, **(params or {})}
    out = {'T_AB': None, 'inliers': np.zeros(len(matches), bool), 'usable': np.zeros(len(matches), bool), 'num_usable': 0}
    if len(matches) < 3:
        return out
    yA, pA, cA, okA = lift(kpts0[matches[:, 0]], scan_a, model, rotated180, P)
    yB, pB, cB, okB = lift(kpts1[matches[:, 1]], scan_b, model, rotated180, P)
    use = np.nonzero(okA & okB)[0]
    out['usable'][use] = True
    out['num_usable'] = len(use)
    if len(use) < max(3, P['min_inliers']):
        return out
    yA, pA, cA, yB, pB, cB = yA[use], pA[use], cA[use], yB[use], pB[use], cB[use]
    eye = np.tile(np.eye(4), (len(use), 1, 1))
    colTA = scan_a['col_T_sc_s'][cA] if P['compensate'] else eye
    colTB = scan_b['col_T_sc_s'][cB] if P['compensate'] else eye

    def errors(T_AB):  # (K, N) worst of the two images' pixel errors
        eB = np.linalg.norm(pixel_errors(inv(T_AB), pA, yB, cB, colTB, model), axis=-1)
        eA = np.linalg.norm(pixel_errors(T_AB, pB, yA, cA, colTA, model), axis=-1)
        return np.maximum(eA, eB)

    # ---- RANSAC (adaptive number of hypotheses)
    rng = np.random.default_rng(P['seed'])
    N, best, best_T, done, needed = len(use), -1, None, 0, P['max_iters']
    while done < min(needed, P['max_iters']):
        s = rng.integers(0, N, (P['batch'], 3))
        a, b, c = pA[s[:, 0]], pA[s[:, 1]], pA[s[:, 2]]
        spread = np.minimum(np.minimum(np.linalg.norm(a - b, axis=1), np.linalg.norm(b - c, axis=1)),
                            np.linalg.norm(a - c, axis=1))
        s = s[spread >= P['min_sample_spread']]
        done += P['batch']
        if len(s) == 0:
            continue
        T = kabsch(pB[s], pA[s])
        counts = (errors(T) <= P['inlier_px']).sum(1)
        k = int(np.argmax(counts))
        if counts[k] > best:
            best, best_T = int(counts[k]), T[k]
            w = best / N
            p_fail = 1 - w ** 3  # a sample is not all inliers
            needed = 0 if p_fail <= 0 else P['max_iters'] if p_fail >= 1 else \
                min(P['max_iters'], int(np.ceil(np.log(1 - P['confidence']) / np.log(p_fail))))
    if best < P['min_inliers']:
        return out

    # ---- refinement: T_AB = [exp(d_phi), d_rho] @ T0, robust least squares of the pixel errors
    T_AB = best_T
    inl = errors(T_AB[None])[0] <= P['inlier_px']
    for _ in range(P['refine_rounds']):
        idx = np.nonzero(inl)[0]

        def residuals(x, T0=T_AB, idx=idx):
            D = np.eye(4)
            D[:3, :3], D[:3, 3] = Rotation.from_rotvec(x[:3]).as_matrix(), x[3:]
            T = (D @ T0)[None]
            eB = pixel_errors(inv(T), pA[idx], yB[idx], cB[idx], colTB[idx], model)[0]
            eA = pixel_errors(T, pB[idx], yA[idx], cA[idx], colTA[idx], model)[0]
            return np.concatenate([eA.ravel(), eB.ravel()])

        x = least_squares(residuals, np.zeros(6), loss='huber', f_scale=P['huber_px']).x
        D = np.eye(4)
        D[:3, :3], D[:3, 3] = Rotation.from_rotvec(x[:3]).as_matrix(), x[3:]
        T_AB = D @ T_AB
        inl = errors(T_AB[None])[0] <= P['inlier_px']
    if inl.sum() < P['min_inliers']:
        return out
    out['T_AB'] = T_AB
    out['inliers'][use[inl]] = True
    return out


def pose_error(T_est, T_gt):
    """Translation [m] and rotation [deg] error of T_est against T_gt (inf, inf if T_est is None)."""
    if T_est is None:
        return np.inf, np.inf
    D = np.linalg.inv(T_gt) @ T_est
    return float(np.linalg.norm(D[:3, 3])), float(np.rad2deg(np.linalg.norm(Rotation.from_matrix(D[:3, :3]).as_rotvec())))


def pose_metrics(item, pair, kpts0, kpts1, matches, params=None, thresholds=POSE_THRESHOLDS, scans=None):
    """Pose metrics of one pair for evaluate_pair's metrics dict. item: split_pairs dict (seq, frame_A, frame_B),
    pair: load_pair dict (T_AB, rotated180), scans: (scan_a, scan_b) to reuse (load_scan), None = load."""
    P = {**POSE_PARAMS, **(params or {})}
    if scans is None:
        scans = (load_scan(item['seq'], item['frame_A'], P['edge_tol']), load_scan(item['seq'], item['frame_B'], P['edge_tol']))
    res = estimate_pose(kpts0, kpts1, matches, *scans, ouster_model(item['seq']), bool(pair.get('rotated180', True)), P)
    t_err, r_err = pose_error(res['T_AB'], pair['T_AB'])
    m = {'pose_usable': res['num_usable'], 'pose_inliers': int(res['inliers'].sum()), 't_err': t_err, 'R_err': r_err}
    for t, r in thresholds:
        m[f'pose@{t:g}m{r:g}deg'] = float(t_err <= t and r_err <= r)
    return m
