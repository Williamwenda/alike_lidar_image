"""Relative pose report: how well does each method's matching recover the transform between two lidar scans.

For every method in MODELS and every pair of SPLITS: the top TOP_K keypoints of both images, mutual-nearest-
neighbour matches of their descriptors (as evaluation/eval_test.py), then the relative pose from the matches
whose keypoints have a 3-D point in both scans (lidar_tools/pose.py: RANSAC + refinement, the motion distortion
of both scans handled with the odometry's within-scan motion). No ground truth is used to select matches; the
estimate is compared with the ICP pose of the pair (T_AB).

    summary.md / summary.csv     success rates, median errors, inliers per split, group and model
    per_pair.csv                 every pair and model
    pose_<split>.png             success rates and inliers vs distance and vs facing
    curves_<split>.png           share of pairs vs translation / rotation error threshold (all pairs)

Usage: edit the parameter block, then `uv run python evaluation/eval_relative_pose.py`.
"""
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib

import matplotlib.pyplot as plt  # noqa: E402
from tqdm import tqdm  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lidar_tools import REPO_ROOT  # noqa: E402
from lidar_tools.pairs import split_pairs, load_pair, preprocess, DIST_BINS, FACING, dist_bin  # noqa: E402
from lidar_tools.model import make_model, extract  # noqa: E402
from lidar_tools.metrics import mutual_nn_match  # noqa: E402
from lidar_tools.pose import pose_metrics, load_scan  # noqa: E402
from eval_test import resolve, groups, in_group, get, write_csv  # noqa: E402

######## Parameters (edit by hand) ########
SPLIT_FILE = REPO_ROOT / 'splits' / 'default.json'
SPLITS = ['test']                 # 'test' (unseen routes) and / or 'val'
MAX_PAIRS = None                  # per split, evenly spaced (None = all)

# name -> (method, weights), as in eval_test.py
MODELS = {
    'ORB': ('orb', None),
    'ALIKE': ('alike-n', None),
    'ALIKE fine-tuned': ('alike-n', 'latest'),
    'XFeat': ('xfeat', None),
    'XFeat fine-tuned': ('xfeat-lidar', 'latest'),
}
CHANNEL = 'signal'
PREPROCESS = 'linear'
TOP_K = 500

# lidar_tools/pose.py
POSE_SETTINGS = dict(
    inlier_px=3.0,           # [px] max reprojection error of an inlier, in both images
    min_inliers=6,           # fewer inliers = failure (infinite pose error)
    max_iters=5000,          # RANSAC hypotheses at most (adaptive: stops at `confidence`)
    confidence=0.999,
    batch=256,               # hypotheses scored together (speed only)
    min_sample_spread=0.5,   # [m] the 3 points of a sample must be this far apart
    refine_rounds=2,         # least squares on the inliers, then new inliers, this many times
    huber_px=1.0,            # [px] Huber scale of the refinement
    min_range=3.0,           # [m] keypoints closer than this get no 3-D point (the operator walking next to the
                             # robot, ~1.5-3 m away, moves with the sensor and supports 'no motion'; as the GT's 3 m)
    edge_tol=0.05,           # keypoints next to a range jump > 5 % get no 3-D point (None = keep)
    compensate=True,         # False: ablation, ignore the within-scan motion
    seed=0,
)
POSE_SUCCESS = ((0.1, 1.0), (0.25, 2.0), (0.5, 5.0))   # (translation [m], rotation [deg]) success thresholds
OUT_DIR = REPO_ROOT / 'evaluation' / 'outputs' / f'pose_report_{time.strftime("%m%d-%H%M")}'
###########################################

SUCCESS = [f'pose@{t:g}m{r:g}deg' for t, r in POSE_SUCCESS]
TABLE = SUCCESS + ['t_err', 'R_err', 'pose_inliers', 'pose_usable', 'num_matches']
MEDIAN = ('t_err', 'R_err')       # median over the pairs (a failure counts as infinite); the rest: mean
COLORS = ['tab:gray', 'tab:cyan', 'tab:blue', 'tab:olive', 'tab:red', 'tab:purple', 'tab:orange']
BINS = [dist_bin(0.5 * (lo + hi)) for lo, hi in zip(DIST_BINS[:-1], DIST_BINS[1:])]
FACES = [f for f, _, _ in FACING]
GUIDE = """How to read it (pairs of scans 1-15 m apart, top TOP_K keypoints per image, mutual-nearest-neighbour matches):
- `pose@<t>m<r>deg`: share of pairs whose estimated pose is within t metres and r degrees of the ICP pose.
- `t_err` [m], `R_err` [deg]: median translation / rotation error over the pairs (a failure, < min_inliers
  inliers, counts as infinite, so `inf` = more than half of the pairs failed).
- `pose_inliers`: RANSAC inliers per pair; `pose_usable`: matches whose keypoints have a 3-D point in both scans
  (a return beyond min_range, not at a depth edge); `num_matches`: mutual-nearest-neighbour matches.
The ICP poses agree within ~3 cm / 0.2 deg when started from shifted guesses: smaller errors are not resolved.
"""


def run(models):
    rows = []
    for split in SPLITS:
        items = split_pairs(SPLIT_FILE, split)
        if MAX_PAIRS is not None and len(items) > MAX_PAIRS:
            items = [items[i] for i in np.linspace(0, len(items) - 1, MAX_PAIRS).round().astype(int)]
        for it in tqdm(items, desc=split):
            pair = load_pair(it['path'])
            img0 = preprocess(pair[f'{CHANNEL}_A'], pair['range_A'] > 0, PREPROCESS)
            img1 = preprocess(pair[f'{CHANNEL}_B'], pair['range_B'] > 0, PREPROCESS)
            scans = (load_scan(it['seq'], it['frame_A'], POSE_SETTINGS['edge_tol']),
                     load_scan(it['seq'], it['frame_B'], POSE_SETTINGS['edge_tol']))
            for name, net in models.items():
                k0, d0, _, _ = extract(net, img0)
                k1, d1, _, _ = extract(net, img1)
                matches, _ = mutual_nn_match(d0, d1)
                m = pose_metrics(it, pair, k0, k1, matches, POSE_SETTINGS, POSE_SUCCESS, scans)
                rows.append({'split': split, 'pair': it['name'], 'seq': it['seq'], 'dist_m': it['dist'],
                             'rot_deg': it['rot_deg'], 'dist_bin': it['dist_bin'], 'facing': it['facing'],
                             'model': name, 'num_matches': len(matches), **m})
    return rows


def summarize(rows, names):
    summary = []
    for split in SPLITS:
        for kind, group in groups(rows, split):
            for name in names:
                sel = [r for r in rows if r['split'] == split and r['model'] == name and in_group(r, kind, group)]
                s = {'split': split, 'kind': kind, 'group': group, 'model': name, 'pairs': len(sel)}
                for k in TABLE:
                    v = [r[k] for r in sel]
                    s[k] = float(np.median(v) if k in MEDIAN else np.mean(v))
                summary.append(s)
    return summary


def write_tables(rows, summary, sources):
    write_csv(OUT_DIR / 'per_pair.csv', rows, list(rows[0]))
    write_csv(OUT_DIR / 'summary.csv', summary, ['split', 'kind', 'group', 'model', 'pairs'] + TABLE)
    settings = ', '.join(f'{k}={v}' for k, v in POSE_SETTINGS.items())
    lines = ['# Relative pose from keypoint matches on lidar intensity images', '',
             'Models:', *[f'- **{n}**: {s}' for n, s in sources.items()], '',
             f'Settings: {CHANNEL} channel, {PREPROCESS} scaling, top {TOP_K} keypoints per image. '
             f'Pose (lidar_tools/pose.py): {settings}. Splits: `{SPLIT_FILE.name}`.', '', GUIDE]
    for split in SPLITS:
        lines += [f'## {split}', '', '| group | model | pairs | ' + ' | '.join(TABLE) + ' |',
                  '|---|---|---|' + '---|' * len(TABLE)]
        for r in (r for r in summary if r['split'] == split):
            group = r['group'] if r['kind'] in ('all', 'route') else f'{r["kind"]} {r["group"]}'
            lines.append(f'| {group} | {r["model"]} | {r["pairs"]} | ' +
                         ' | '.join(f'{r[k]:.3f}' if k in SUCCESS + list(MEDIAN) else f'{r[k]:.1f}' for k in TABLE) + ' |')
        lines.append('')
    (OUT_DIR / 'summary.md').write_text('\n'.join(lines))
    return '\n'.join(lines)


def plot_groups(summary, names, split):
    """Success rates and inliers vs distance (lines) and vs facing (bars), one colour per model."""
    metrics = [(k, f'success (<= {t:g} m, {r:g} deg)') for k, (t, r) in zip(SUCCESS, POSE_SUCCESS)] + \
              [('pose_inliers', 'inliers / pair')]
    fig, axes = plt.subplots(2, len(metrics), figsize=(4.3 * len(metrics), 7.4), constrained_layout=True)
    width = 0.8 / len(names)
    for j, (k, label) in enumerate(metrics):
        for c, name in enumerate(names):
            color = COLORS[c % len(COLORS)]
            axes[0, j].plot(BINS, [get(summary, split, 'distance', b, name, k) for b in BINS], 'o-', color=color, label=name)
            x = np.arange(len(FACES)) + (c - (len(names) - 1) / 2) * width
            bars = axes[1, j].bar(x, [get(summary, split, 'facing', f, name, k) for f in FACES], width, color=color)
            axes[1, j].bar_label(bars, fmt='%.0f' if k == 'pose_inliers' else '%.2f', fontsize=6)
        n_pairs = [int(get(summary, split, 'facing', f, names[0], 'pairs')) for f in FACES]
        axes[1, j].set_xticks(np.arange(len(FACES)), [f'{f}\n({n} pairs)' for f, n in zip(FACES, n_pairs)])
        axes[0, j].set(title=f'{split}: {label}', xlabel='distance between the scans')
        axes[1, j].set(title=f'{split}: {label}', xlabel='facing (rotation between the scans)')
        for ax in axes[:, j]:
            ax.grid(axis='y', alpha=0.3)
            if k != 'pose_inliers':
                ax.set_ylim(0, 1)
    axes[0, 0].legend(fontsize=7)
    fig.savefig(OUT_DIR / f'pose_{split}.png', dpi=130)
    plt.close(fig)


def plot_curves(rows, names, split):
    """Share of pairs with an error below the threshold (failures never count), translation and rotation."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    for ax, (k, ths, label) in zip(axes, (('t_err', np.linspace(0, 1, 101), 'translation error [m]'),
                                          ('R_err', np.linspace(0, 10, 101), 'rotation error [deg]'))):
        for c, name in enumerate(names):
            e = np.array([r[k] for r in rows if r['split'] == split and r['model'] == name])
            ax.plot(ths, [(e <= t).mean() for t in ths], color=COLORS[c % len(COLORS)], label=name)
        ax.set(xlabel=label, ylabel='share of pairs', title=f'{split}: pose error below threshold (all pairs)',
               ylim=(0, 1), xlim=(0, ths[-1]))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.savefig(OUT_DIR / f'curves_{split}.png', dpi=130)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    models, sources = {}, {}
    for name, (arch, weights) in MODELS.items():
        weights = resolve(arch, weights)
        models[name] = make_model(arch, weights, top_k=TOP_K)
        sources[name] = f'{arch}, ' + (str(weights) if weights else ('OpenCV' if arch == 'orb' else 'pretrained weights'))
    names = list(models)
    rows = run(models)
    summary = summarize(rows, names)
    print('\n' + write_tables(rows, summary, sources))
    for split in SPLITS:
        plot_groups(summary, names, split)
        plot_curves(rows, names, split)
    print(f'\nreport saved to {OUT_DIR}')


if __name__ == '__main__':
    matplotlib.use('Agg')  # files only (not on import: eval_pair / eval_pose_pair show figures)
    main()
