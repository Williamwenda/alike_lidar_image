"""Relative pose on one lidar pair, method by method: the detected keypoints, their 3-D points, and the pose they
give compared with the ground-truth (ICP) pose.

Runs exactly what evaluation/eval_relative_pose.py runs for one pair (same POSE_SETTINGS), for every method in
MODELS: top TOP_K keypoints, mutual-nearest-neighbour matches, each keypoint's 3-D point (its pixel's point
undistorted to the scan stamp; a return beyond min_range, not at a depth edge), RANSAC + refinement with both
scans' within-scan motion (lidar_tools/pose.py).

Prints per method the keypoints, matches, inliers and the translation / rotation error of the pose, and shows
(SHOW; SAVE: also saves to OUT_DIR):
    pose_<pair>_keypoints.png      per method, image A above image B: every detected keypoint (TOP_K)
    pose_<pair>_reliable.png       per method, image A above image B: the reliable features, i.e. the RANSAC inlier
                                   matches (keypoints with a 3-D point whose matches agree on the estimated pose)
    pose_<pair>_feature_clouds.png per method (column), top view of the keypoints' 3-D points in A's frame, B's moved
                                   by the ground-truth pose (top) and by the estimated pose (bottom); blue A, orange B,
                                   black lines: inlier matches
SHOW_3D opens one Open3D window per method (close it to see the next): the keypoints' 3-D points of A (blue),
of B moved by the estimated pose (orange) and by the ground-truth pose (green), inlier matches (grey lines),
and the sensor axes (A; B estimated; B ground truth, small).

Usage: edit the parameter block, then `uv run python evaluation/eval_pose_pair.py`.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lidar_tools import REPO_ROOT  # noqa: E402
from lidar_tools.pairs import PAIRS_ROOT, load_pair, preprocess  # noqa: E402
from lidar_tools.model import make_model, extract  # noqa: E402
from lidar_tools.metrics import mutual_nn_match  # noqa: E402
from lidar_tools.pose import estimate_pose, pose_error, load_scan, ouster_model, lift  # noqa: E402
from eval_test import resolve  # noqa: E402
from eval_relative_pose import POSE_SETTINGS, CHANNEL, PREPROCESS  # noqa: E402  (same settings as the report)

######## Parameters (edit by hand) ########
SEQ = 'utias_parking'        # utias_dome, utias_grassy (test routes), utias_parking, utias_woody (training routes;
                             # their 'val' pairs in splits/default.json are places held out from training)
FRAME_A = 500                # a pair file <FRAME_A>_<FRAME_B>.npz must exist (see <seq>/<PAIRS_DIR>/index.csv)
FRAME_B = 3386               # None: the first pair of FRAME_A
PAIRS_DIR = 'pairs'          # folder of the pair files in lidar_image_process/outputs/<seq>/ ('pairs_custom': hand-made)
# Example pairs (PAIRS_DIR 'pairs' unless noted), all three methods good at TOP_K 1000:
#   parking 3527_3539   consecutive, 1.7 m (12 frames apart)       | test route: dome 954_967 (1.6 m, 14 deg)
#   parking 500_3386    loop revisit, 1.6 m, facing the other way   | test route: grassy 10_2502 (1.8 m, 176 deg)
# Harder: parking 1714_2151 (loop, 6.9 m, val); dome 440_505 (9.3 m, repetitive dome panels, all fail);
#   parking 552_3346 / 552_3306 ('pairs_custom', right-turn arrow; loop 1.1 m / 6.1 m), 620_3260 ('pairs_custom', ORB fails)
# name -> (method, weights), as in eval_relative_pose.MODELS
MODELS = {
    'ORB': ('orb', None),
    'ALIKE fine-tuned': ('alike-n', 'latest'),
    'XFeat fine-tuned': ('xfeat-lidar', 'latest'),
}
TOP_K = 1000                  # keypoints per image, the strongest ones (as eval_test / eval_relative_pose: 500).
                             # ORB: nfeatures; ALIKE: -1 = every local maximum with score > 0.2 (up to 5000);
                             # XFeat: at most one per 8 x 8 cell, 2048 on a 128 x 1024 image
VIEW_RANGE = 40.0            # [m] top view: 3-D points within this horizontal distance of sensor A
SHOW = True                  # open the matplotlib figures
SAVE = False                 # also save them to OUT_DIR
SHOW_3D = True               # Open3D window per method
POINT_SIZE = 6.0             # Open3D point size [px]
OUT_DIR = REPO_ROOT / 'evaluation' / 'outputs'
###########################################

BLUE, ORANGE, GREEN, RED, CYAN, GREY = '#1f77b4', '#ff7f0e', (0.1, 0.85, 0.2), (0.95, 0.1, 0.1), (0.0, 0.85, 1.0), (0.6, 0.6, 0.6)


def find_pair():
    folder = PAIRS_ROOT / SEQ / PAIRS_DIR
    pattern = f'{FRAME_A:06d}_' + (f'{FRAME_B:06d}.npz' if FRAME_B is not None else '*.npz')
    files = sorted(folder.glob(pattern))
    if not files:
        raise FileNotFoundError(f'no pair {pattern} in {folder} (see {folder / "index.csv"})')
    return files[0]


def transform(T, p):
    return p @ T[:3, :3].T + T[:3, 3]


def run_method(net, img0, img1, scans, model, rotated, T_gt):
    """Keypoints, matches, pose and the keypoints' 3-D points of one method on the pair."""
    k0, d0, _, _ = extract(net, img0)
    k1, d1, _, _ = extract(net, img1)
    matches, _ = mutual_nn_match(d0, d1)
    res = estimate_pose(k0, k1, matches, *scans, model, rotated, POSE_SETTINGS)
    _, p0, _, ok0 = lift(k0, scans[0], model, rotated, POSE_SETTINGS)   # 3-D point of every keypoint
    _, p1, _, ok1 = lift(k1, scans[1], model, rotated, POSE_SETTINGS)
    inl = matches[res['inliers']]
    t_err, r_err = pose_error(res['T_AB'], T_gt)
    return dict(k0=k0, k1=k1, ok0=ok0, ok1=ok1, matches=matches, res=res, cloud0=p0[ok0], cloud1=p1[ok1],
                inl0=p0[inl[:, 0]], inl1=p1[inl[:, 1]], t_err=t_err, r_err=r_err)


def result_text(r):
    if r['res']['T_AB'] is None:
        return 'pose FAILED'
    return f'error {r["t_err"]:.3f} m / {r["r_err"]:.2f} deg'


def stacked(img0, img1, results, title):
    """Figure with one panel per method showing image A above image B; returns fig, axes, row offset of B."""
    H, W = img0.shape[:2]
    gap = 40
    canvas = np.full((2 * H + gap, W, 3), 255, np.uint8)
    canvas[:H], canvas[H + gap:] = img0, img1
    fig, axes = plt.subplots(len(results), 1, figsize=(18, 3.4 * len(results)), squeeze=False, constrained_layout=True)
    for ax in axes[:, 0]:
        ax.imshow(canvas, interpolation='none')
        ax.text(-8, H / 2, 'A', ha='right', va='center')
        ax.text(-8, H + gap + H / 2, 'B', ha='right', va='center')
        ax.set_axis_off()
    fig.suptitle(title, fontsize=10)
    return fig, axes[:, 0], H + gap


def draw_keypoints(img0, img1, results, title):
    """Every detected keypoint of each method."""
    fig, axes, off = stacked(img0, img1, results, title)
    for ax, (label, r) in zip(axes, results.items()):
        ax.scatter(r['k0'][:, 0], r['k0'][:, 1], s=5, color=CYAN, lw=0)
        ax.scatter(r['k1'][:, 0], r['k1'][:, 1] + off, s=5, color=CYAN, lw=0)
        ax.set_title(f'[{label}] detected keypoints: A {len(r["k0"])}, B {len(r["k1"])}', fontsize=9, loc='left')
    return fig


def draw_reliable(img0, img1, results, title):
    """The RANSAC inlier matches of each method: the features that agree on the estimated pose."""
    W = img0.shape[1]
    fig, axes, off = stacked(img0, img1, results, title)
    for ax, (label, r) in zip(axes, results.items()):
        inl = r['matches'][r['res']['inliers']]
        p0, p1 = r['k0'][inl[:, 0]], r['k1'][inl[:, 1]]
        straight = np.abs(p0[:, 0] - p1[:, 0]) < W / 2  # across the azimuth seam: end points only
        ax.plot(np.stack([p0[straight, 0], p1[straight, 0]]), np.stack([p0[straight, 1], p1[straight, 1] + off]),
                color=GREEN, lw=0.9, alpha=0.9)
        ax.scatter(np.r_[p0[:, 0], p1[:, 0]], np.r_[p0[:, 1], p1[:, 1] + off], s=12, color=GREEN, zorder=3)
        ax.set_title(f'[{label}] reliable features: {len(inl)} RANSAC inliers (of {len(r["matches"])} matches, '
                     f'{r["res"]["num_usable"]} with 3-D points)   {result_text(r)}', fontsize=9, loc='left')
    return fig


def draw_feature_clouds(results, T_gt, title):
    fig, axes = plt.subplots(2, len(results), figsize=(6 * len(results), 11), squeeze=False,
                             constrained_layout=True, sharex=True, sharey=True)
    near = lambda p: np.hypot(p[:, 0], p[:, 1]) < VIEW_RANGE
    for col, (label, r) in enumerate(results.items()):
        for ax, T, name in zip(axes[:, col], (T_gt, r['res']['T_AB']), ('ground-truth (ICP) pose', 'estimated pose')):
            a = r['cloud0']
            ax.scatter(*a[near(a), :2].T, s=6, color=BLUE, label='A keypoints')
            ax.set(aspect='equal', xlabel='x [m] (sensor A)', ylabel='y [m]')
            ax.grid(alpha=0.3)
            if T is None:
                ax.set_title(f'[{label}] {name}: failed', fontsize=10)
                continue
            b = transform(T, r['cloud1'])
            ax.scatter(*b[near(b), :2].T, s=6, color=ORANGE, label='B keypoints, moved into A')
            if len(r['inl0']):
                q = transform(T, r['inl1'])
                v = near(r['inl0']) & near(q)
                ax.plot(np.stack([r['inl0'][v, 0], q[v, 0]]), np.stack([r['inl0'][v, 1], q[v, 1]]), color='k', lw=0.8)
            for Ts, s in ((np.eye(4), 'A'), (T, 'B')):  # sensor positions and headings
                o, x = Ts[:3, 3], Ts[:3, 0]
                ax.arrow(o[0], o[1], 2 * x[0], 2 * x[1], width=0.15, color='k')
                ax.text(o[0], o[1], f' {s}', fontsize=10, weight='bold')
            ax.set_title(f'[{label}] {name}' + (f': {result_text(r)}' if T is not T_gt else ''), fontsize=10)
    axes[0, 0].legend(fontsize=8, loc='upper right')
    fig.suptitle(title, fontsize=10)
    return fig


def show_3d(r, T_gt, name):
    import open3d as o3d

    def points(p, color):
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p))
        pc.paint_uniform_color(color)
        return pc

    T = r['res']['T_AB']
    geoms = [points(r['cloud0'], [0.12, 0.47, 0.71]), points(transform(T_gt, r['cloud1']), [0.1, 0.75, 0.2]),
             o3d.geometry.TriangleMesh.create_coordinate_frame(1.0),
             o3d.geometry.TriangleMesh.create_coordinate_frame(0.5).transform(T_gt)]
    if T is not None:
        geoms += [points(transform(T, r['cloud1']), [1.0, 0.5, 0.05]),
                  o3d.geometry.TriangleMesh.create_coordinate_frame(1.0).transform(T)]
        if len(r['inl0']):
            pts = np.concatenate([r['inl0'], transform(T, r['inl1'])])
            n = len(r['inl0'])
            ls = o3d.geometry.LineSet(o3d.utility.Vector3dVector(pts),
                                      o3d.utility.Vector2iVector(np.stack([np.arange(n), np.arange(n) + n], 1)))
            ls.paint_uniform_color([0.5, 0.5, 0.5])
            geoms.append(ls)
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=name, width=1400, height=900)
    for g in geoms:
        vis.add_geometry(g)
    vis.get_render_option().point_size = POINT_SIZE
    vis.run()
    vis.destroy_window()


def main():
    path = find_pair()
    pair = load_pair(path)
    fa, fb, rotated = int(pair['frame_A']), int(pair['frame_B']), bool(pair.get('rotated180', True))
    T_gt = pair['T_AB']
    dist, rot = np.linalg.norm(T_gt[:2, 3]), pose_error(T_gt, np.eye(4))[1]
    t_odo, r_odo = pose_error(pair['T_AB_odom'], T_gt)
    title = f'{SEQ} {path.stem}: {dist:.1f} m apart, rotated {rot:.0f} deg'

    img0 = preprocess(pair[f'{CHANNEL}_A'], pair['range_A'] > 0, PREPROCESS)
    img1 = preprocess(pair[f'{CHANNEL}_B'], pair['range_B'] > 0, PREPROCESS)
    scans = (load_scan(SEQ, fa, POSE_SETTINGS['edge_tol']), load_scan(SEQ, fb, POSE_SETTINGS['edge_tol']))
    model = ouster_model(SEQ)

    results = {}
    for label, (arch, weights) in MODELS.items():
        net = make_model(arch, resolve(arch, weights), top_k=TOP_K)
        results[label] = run_method(net, img0, img1, scans, model, rotated, T_gt)

    print(f'\n{title}  (odometry guess: {t_odo:.3f} m / {r_odo:.2f} deg from the ICP pose)\n')
    print(f'{"method":18s} {"keypoints A/B":>14s} {"3-D A/B":>10s} {"matches":>8s} {"usable":>7s} {"inliers":>8s}'
          f' {"t_err [m]":>10s} {"R_err [deg]":>12s}')
    for label, r in results.items():
        t, R = ('failed', 'failed') if r['res']['T_AB'] is None else (f'{r["t_err"]:.3f}', f'{r["r_err"]:.2f}')
        print(f'{label:18s} {f"{len(r["k0"])}/{len(r["k1"])}":>14s} {f"{r["ok0"].sum()}/{r["ok1"].sum()}":>10s} '
              f'{len(r["matches"]):8d} {r["res"]["num_usable"]:7d} {r["res"]["inliers"].sum():8d} {t:>10s} {R:>12s}')

    figs = {'keypoints': draw_keypoints(img0, img1, results, title), 'reliable': draw_reliable(img0, img1, results, title),
            'feature_clouds': draw_feature_clouds(results, T_gt, title)}
    if SAVE:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        for k, fig in figs.items():
            fig.savefig(OUT_DIR / f'pose_{SEQ}_{path.stem}_{k}.png', dpi=110)
        print(f'\nfigures saved to {OUT_DIR}/pose_{SEQ}_{path.stem}_*.png')
    if SHOW:
        plt.show()
    if SHOW_3D:
        for label, r in results.items():
            show_3d(r, T_gt, f'{path.stem} [{label}]: A blue, B estimated orange, B ground truth green ({result_text(r)})')


if __name__ == '__main__':
    main()
