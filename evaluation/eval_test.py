"""Test report: how good and how fast are the methods on held-out lidar pairs, against the ground truth.

For every method in MODELS (ORB, ALIKE, XFeat, pretrained or fine-tuned; lidar_tools.model) on the splits in SPLITS ('test': whole routes never used in training;
'val': the held-out places of the training routes), results for all pairs and per group:
route, distance between the two scans (lidar_tools.pairs.DIST_BINS) and facing (rotation between
them: same / side / opposite, lidar_tools.pairs.FACING).

    summary.md / summary.csv     mean metrics per split, group and model (definitions: lidar_tools/metrics.py)
    per_pair.csv                 every pair and model
    timing.md / speed.png        extraction time per image on the GPU and the CPU, matching time per pair,
                                 and accuracy vs extraction time
    viewpoint_<split>.png        repeatability, localisation error, re-detected keypoints and MMA
                                 vs distance and vs facing: how the models cope with viewpoint change
    curves_<split>.png           repeatability and MMA vs pixel threshold (all pairs)
    loc_err_<split>.png          distance of the re-detected keypoints to their ground-truth location
    examples/*.png               the same pairs for every model, one per distance bin and facing:
                                 detections in A and B (green re-detected / red missed / grey no
                                 ground truth) and matches (green correct / red wrong)

Usage: edit the parameter block, then `uv run python evaluation/eval_test.py`.
"""
import os
import sys
import time
from pathlib import Path

# one BLAS thread for numpy: the 500 x 500 descriptor products are too small to split, and with 24 threads
# competing with torch's the matching time varied 0.3-9 ms per pair
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np  # noqa: E402
import matplotlib
import torch

import matplotlib.pyplot as plt  # noqa: E402
from tqdm import tqdm  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lidar_tools import REPO_ROOT  # noqa: E402
from lidar_tools.pairs import split_pairs, load_pair, preprocess, DIST_BINS, FACING, dist_bin  # noqa: E402
from lidar_tools.model import make_model, extract, XFEAT_ROOT  # noqa: E402
from lidar_tools.metrics import evaluate_pair, mean_metrics, mutual_nn_match, THRESHOLDS, LOC_TH  # noqa: E402
from lidar_tools.plots import pair_figure  # noqa: E402

LOG_LIDAR = REPO_ROOT / 'training_code' / 'training' / 'log_lidar'
LOG_XFEAT = XFEAT_ROOT / 'log_lidar'


def latest_run(name='best.pth', model='alike-n'):
    """Weights of the most recent fine-tuning run: training_code/training/train_lidar.py (ALIKE) or
    xfeat_lidar_image/modules/training/train_lidar.py ('xfeat-lidar')."""
    pattern, root = (f'*/{name}', LOG_XFEAT) if model.startswith('xfeat') else (f'*/checkpoints/{name}', LOG_LIDAR / model)
    runs = sorted(root.glob(pattern))
    if not runs:
        raise FileNotFoundError(f'no {name} in {root}: run the fine-tuning first')
    return runs[-1]


######## Parameters (edit by hand) ########
SPLIT_FILE = REPO_ROOT / 'splits' / 'default.json'
SPLITS = ['test']                 # 'test' (unseen routes) and / or 'val'
MAX_PAIRS = None                  # per split, evenly spaced (None = all)

# name -> (method, weights): method 'orb', 'alike-n', 'xfeat', 'xfeat-lidar' (lidar_tools.model); weights None =
# pretrained, 'latest' = best.pth of the most recent fine-tuning run of that method, or the path of a .pth
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
CORR_MIN_RANGE = 3.0
TIMING_IMAGES = 100               # test images timed per method (after 10 warm-up runs)
TIME_CPU = True                   # also time every method on the CPU (torch threads: all cores)
OUT_DIR = REPO_ROOT / 'evaluation' / 'outputs' / f'test_report_{time.strftime("%m%d-%H%M")}'
###########################################

TABLE = ['rep@1', 'rep@3', 'loc_err', 'num_rep@3', 'cov_ratio', 'MMA@1', 'MMA@3', 'num_correct@3',
         'matching_score@3']
COLORS = ['tab:gray', 'tab:cyan', 'tab:blue', 'tab:olive', 'tab:red', 'tab:purple', 'tab:orange']
BINS = [dist_bin(0.5 * (lo + hi)) for lo, hi in zip(DIST_BINS[:-1], DIST_BINS[1:])]
FACES = [f for f, _, _ in FACING]
GUIDE = """How to read it (pairs of scans 1-15 m apart, top 500 keypoints per image; definitions in lidar_tools/metrics.py):
- `rep@3` repeatability: of the keypoints whose location in the other scan is known, the share that the other scan
  also detects within 3 px. The main detection score; 1 = every keypoint is found again.
- `loc_err` localisation error: median distance [px] between a re-detected keypoint and its ground-truth location
  (only keypoints within 3 px count). Lower = more precise positions; the ground truth itself is ~0.1 px.
- `num_rep@3` re-detected keypoints per image: how many usable keypoints a scan gives (depends on the overlap).
- `cov_ratio`: share of the keypoints that have ground truth (on surfaces seen by both scans, beyond 3 m).
- `MMA@3` match precision: share of the mutual-nearest-neighbour descriptor matches that land within 3 px.
- `num_correct@3`, `matching_score@3`: correct matches per pair, and per keypoint with ground truth.
"""


def resolve(arch, weights):
    return latest_run(model=arch) if weights == 'latest' else weights


def load_models(device='cuda'):
    models, sources = {}, {}
    for name, (arch, weights) in MODELS.items():
        weights = resolve(arch, weights)
        models[name] = make_model(arch, weights, top_k=TOP_K, device=device)
        sources[name] = f'{arch}, ' + (str(weights) if weights else ('OpenCV' if arch == 'orb' else 'pretrained weights'))
    return models, sources


def time_extraction(model, images):
    """Median and mean extraction time [ms] per image (keypoints + descriptors, back as numpy arrays)."""
    cuda = model.device != 'cpu'
    for img in images[:10]:
        extract(model, img)
    times = []
    for img in images:
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        extract(model, img)
        if cuda:
            torch.cuda.synchronize()
        times.append(1000 * (time.perf_counter() - t0))
    return float(np.median(times)), float(np.mean(times))


def run_timing(models):
    """Extraction time on the GPU (ORB: CPU only) and the CPU, and matching time (mutual nearest neighbours of
    the descriptors, as in the evaluation, numpy on the CPU) on TIMING_IMAGES test images."""
    items = split_pairs(SPLIT_FILE, SPLITS[0])
    items = [items[i] for i in np.linspace(0, len(items) - 1, TIMING_IMAGES).round().astype(int)]
    pairs = [load_images(load_pair(it['path'], CORR_MIN_RANGE)) for it in items]
    images = [img for pair in pairs for img in pair][:TIMING_IMAGES]
    cpu_models = load_models('cpu')[0] if TIME_CPU else {}
    timing = []
    for name, model in models.items():
        row = {'model': name, 'gpu_ms': np.nan, 'gpu_ms_mean': np.nan, 'cpu_ms': np.nan, 'cpu_ms_mean': np.nan}
        if model.device != 'cpu':
            row['gpu_ms'], row['gpu_ms_mean'] = time_extraction(model, images)
        if name in cpu_models:
            row['cpu_ms'], row['cpu_ms_mean'] = time_extraction(cpu_models[name], images)
        feats = [(extract(model, a), extract(model, b)) for a, b in pairs[:20]]
        t = []
        for (k0, d0, _, _), (k1, d1, _, _) in feats:
            mutual_nn_match(d0, d1)  # warm-up
            for _ in range(5):
                t0 = time.perf_counter()
                mutual_nn_match(d0, d1)
                t.append(1000 * (time.perf_counter() - t0))
        row.update(match_ms=float(np.median(t)), num_kpts=float(np.mean([len(f[0][0]) for f in feats])),
                   desc_dim=int(feats[0][0][1].shape[1]))
        timing.append(row)
        print(f'{name}: GPU {row["gpu_ms"]:.1f} ms, CPU {row["cpu_ms"]:.1f} ms per image, matching {row["match_ms"]:.2f} ms')
    return timing


def write_timing(timing, summary):
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'
    lines = ['# Inference time', '',
             f'Median over {TIMING_IMAGES} test images ({TOP_K} keypoints requested), extraction = keypoints + descriptors '
             f'from the 8-bit image, results back as numpy arrays. GPU: {gpu}; CPU: {torch.get_num_threads()} torch '
             f'threads (ORB: OpenCV, CPU only). Matching: mutual nearest neighbours of the descriptors (numpy, CPU, 1 BLAS thread), '
             f'the matcher of the evaluation, per pair.', '',
             '| method | GPU [ms / image] | CPU [ms / image] | matching [ms / pair] | keypoints | descriptor dim | '
             'rep@3 | MMA@3 |', '|---|---|---|---|---|---|---|---|']
    for r in timing:
        acc = [get(summary, SPLITS[0], 'all', 'all', r['model'], k) for k in ('rep@3', 'MMA@3')]
        lines.append(f'| {r["model"]} | ' + ' | '.join('-' if np.isnan(v) else f'{v:.1f}' for v in (r['gpu_ms'], r['cpu_ms']))
                     + f' | {r["match_ms"]:.2f} | {r["num_kpts"]:.0f} | {r["desc_dim"]} | {acc[0]:.3f} | {acc[1]:.3f} |')
    (OUT_DIR / 'timing.md').write_text('\n'.join(lines) + '\n')
    write_csv(OUT_DIR / 'timing.csv', timing, list(timing[0]))
    return '\n'.join(lines)


def plot_speed(timing, summary, names):
    """Accuracy (all pairs of the first split) vs extraction time, GPU (filled) and CPU (open)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    for ax, (k, label) in zip(axes, (('rep@3', 'repeatability @3 px'), ('MMA@3', 'MMA @3 px'))):
        for c, name in enumerate(names):
            r = next(r for r in timing if r['model'] == name)
            y = get(summary, SPLITS[0], 'all', 'all', name, k)
            color = COLORS[c % len(COLORS)]
            for dev, face in (('gpu_ms', color), ('cpu_ms', 'none')):
                if not np.isnan(r[dev]):
                    ax.scatter(r[dev], y, s=60, facecolors=face, edgecolors=color,
                               label=f'{name} ({dev[:3].upper()})')
        ax.set(xscale='log', xlabel='extraction time per image [ms] (log)', ylabel=label,
               title=f'{SPLITS[0]}: {label} vs time (filled: GPU, open: CPU)', ylim=(0, 1))
        ax.grid(alpha=0.3, which='both')
    axes[0].legend(fontsize=6, ncol=2)
    fig.savefig(OUT_DIR / 'speed.png', dpi=130)
    plt.close(fig)


def load_images(pair):
    return (preprocess(pair[f'{CHANNEL}_A'], pair['range_A'] > 0, PREPROCESS),
            preprocess(pair[f'{CHANNEL}_B'], pair['range_B'] > 0, PREPROCESS))


def run(models):
    rows = []                                                  # one per pair and model
    rep_dist = {(s, n): [] for s in SPLITS for n in models}    # distances of the re-detected keypoints
    for split in SPLITS:
        items = split_pairs(SPLIT_FILE, split)
        if MAX_PAIRS is not None and len(items) > MAX_PAIRS:
            items = [items[i] for i in np.linspace(0, len(items) - 1, MAX_PAIRS).round().astype(int)]
        for it in tqdm(items, desc=split):
            pair = load_pair(it['path'], CORR_MIN_RANGE)
            img0, img1 = load_images(pair)
            for name, net in models.items():
                k0, d0, _, _ = extract(net, img0)
                k1, d1, _, _ = extract(net, img1)
                m, det = evaluate_pair(k0, d0, k1, d1, pair['corr_AB'], pair['corr_BA'])
                rows.append({'split': split, 'pair': it['name'], 'seq': it['seq'], 'dist_m': it['dist'],
                             'rot_deg': it['rot_deg'], 'dist_bin': it['dist_bin'], 'facing': it['facing'],
                             'model': name, **m})
                d = np.concatenate([det['d0'], det['d1']])
                rep_dist[(split, name)].append(d[np.isfinite(d) & (d <= LOC_TH)])
    return rows, rep_dist


def groups(rows, split):
    """(kind, group) of the summary: all pairs, per route, per distance bin, per facing."""
    sel = [r for r in rows if r['split'] == split]
    out = [('all', 'all')] + [('route', s) for s in sorted({r['seq'] for r in sel})]
    out += [('distance', b) for b in BINS if any(r['dist_bin'] == b for r in sel)]
    out += [('facing', f) for f in FACES if any(r['facing'] == f for r in sel)]
    return out


def in_group(r, kind, group):
    return {'all': True, 'route': r['seq'] == group, 'distance': r['dist_bin'] == group,
            'facing': r['facing'] == group}[kind]


def summarize(rows, names):
    meta = ('split', 'pair', 'seq', 'dist_m', 'rot_deg', 'dist_bin', 'facing', 'model')
    summary = []
    for split in SPLITS:
        for kind, group in groups(rows, split):
            for name in names:
                sel = [r for r in rows if r['split'] == split and r['model'] == name and in_group(r, kind, group)]
                summary.append({'split': split, 'kind': kind, 'group': group, 'model': name, 'pairs': len(sel),
                                **mean_metrics([{k: v for k, v in r.items() if k not in meta} for r in sel])})
    return summary


def get(summary, split, kind, group, name, key):
    return next((r[key] for r in summary if (r['split'], r['kind'], r['group'], r['model']) == (split, kind, group, name)),
                np.nan)


def write_csv(path, rows, keys):
    with open(path, 'w') as f:
        f.write(','.join(keys) + '\n')
        for r in rows:
            f.write(','.join(f'{r[k]:.4f}' if isinstance(r[k], float) else str(r[k]) for k in keys) + '\n')


def write_tables(rows, summary, sources):
    write_csv(OUT_DIR / 'per_pair.csv', rows, list(rows[0]))
    write_csv(OUT_DIR / 'summary.csv', summary, ['split', 'kind', 'group', 'model', 'pairs'] + TABLE)

    lines = ['# ALIKE on lidar intensity images: test report', '',
             'Models:', *[f'- **{n}**: {s}' for n, s in sources.items()], '',
             f'Settings: {CHANNEL} channel, {PREPROCESS} scaling, top {TOP_K} keypoints per image, sub-pixel, '
             f'ground truth beyond {CORR_MIN_RANGE} m. Splits: `{SPLIT_FILE.name}`.', '', GUIDE]
    for split in SPLITS:
        lines += [f'## {split}', '', '| group | model | pairs | ' + ' | '.join(TABLE) + ' |',
                  '|---|---|---|' + '---|' * len(TABLE)]
        for r in (r for r in summary if r['split'] == split):
            group = r['group'] if r['kind'] in ('all', 'route') else f'{r["kind"]} {r["group"]}'
            lines.append(f'| {group} | {r["model"]} | {r["pairs"]} | ' +
                         ' | '.join(f'{r[k]:.3f}' if k not in ('num_rep@3', 'num_correct@3') else f'{r[k]:.1f}'
                                    for k in TABLE) + ' |')
        lines.append('')
    (OUT_DIR / 'summary.md').write_text('\n'.join(lines))
    return '\n'.join(lines)


def plot_viewpoint(summary, names, split):
    """Metrics vs distance (lines) and vs facing (bars), one colour per model."""
    metrics = [('rep@3', 'repeatability @3 px'), ('loc_err', 'loc. error [px], lower = better'),
               ('num_rep@3', 're-detected keypoints / image'), ('MMA@3', 'MMA @3 px')]
    fig, axes = plt.subplots(2, len(metrics), figsize=(4.3 * len(metrics), 7.4), constrained_layout=True)
    width = 0.8 / len(names)
    for j, (k, label) in enumerate(metrics):
        for c, name in enumerate(names):
            color = COLORS[c % len(COLORS)]
            axes[0, j].plot(BINS, [get(summary, split, 'distance', b, name, k) for b in BINS], 'o-', color=color, label=name)
            x = np.arange(len(FACES)) + (c - (len(names) - 1) / 2) * width
            bars = axes[1, j].bar(x, [get(summary, split, 'facing', f, name, k) for f in FACES], width, color=color,
                                  label=name)
            axes[1, j].bar_label(bars, fmt='%.2f' if k != 'num_rep@3' else '%.0f', fontsize=6)
        n_pairs = [int(get(summary, split, 'facing', f, names[0], 'pairs')) for f in FACES]
        axes[1, j].set_xticks(np.arange(len(FACES)), [f'{f}\n({n} pairs)' for f, n in zip(FACES, n_pairs)])
        axes[0, j].set(title=f'{split}: {label}', xlabel='distance between the scans')
        axes[1, j].set(title=f'{split}: {label}', xlabel='facing (rotation between the scans)')
        for ax in axes[:, j]:
            ax.grid(axis='y', alpha=0.3)
            if k in ('rep@3', 'MMA@3'):
                ax.set_ylim(0, 1)
    axes[0, 0].legend(fontsize=7)
    fig.savefig(OUT_DIR / f'viewpoint_{split}.png', dpi=130)
    plt.close(fig)


def plot_curves(summary, names, split):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    for ax, (metric, label) in zip(axes, (('rep', 'repeatability'), ('MMA', 'MMA (match precision)'))):
        for c, name in enumerate(names):
            ax.plot(THRESHOLDS, [get(summary, split, 'all', 'all', name, f'{metric}@{t}') for t in THRESHOLDS],
                    color=COLORS[c % len(COLORS)], marker='.', label=name)
        ax.set(xlabel='threshold [px]', ylabel=label, title=f'{split}: {label} (all pairs)', ylim=(0, 1),
               xticks=THRESHOLDS)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.savefig(OUT_DIR / f'curves_{split}.png', dpi=130)
    plt.close(fig)


def plot_loc_hist(rep_dist, names, split):
    fig, ax = plt.subplots(figsize=(7, 3.6), constrained_layout=True)
    bins = np.linspace(0, LOC_TH, 31)
    for c, name in enumerate(names):
        d = np.concatenate(rep_dist[(split, name)]) if rep_dist[(split, name)] else np.zeros(0)
        ax.hist(d, bins, histtype='step', lw=1.5, color=COLORS[c % len(COLORS)],
                label=f'{name}: {len(d)} keypoints, median {np.median(d):.2f} px')
    ax.set(title=f'{split}: distance of the re-detected keypoints to the ground truth', xlabel='[px]',
           ylabel='keypoints')
    ax.legend(fontsize=7)
    fig.savefig(OUT_DIR / f'loc_err_{split}.png', dpi=130)
    plt.close(fig)


def plot_examples(models):
    """One pair per distance bin and facing (if there is one), the same pairs for every model."""
    (OUT_DIR / 'examples').mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    for split in SPLITS:
        items = split_pairs(SPLIT_FILE, split)
        for b in BINS:
            for f in ('same', 'opposite'):
                cand = [it for it in items if it['dist_bin'] == b and it['facing'] == f]
                if not cand:
                    continue
                it = cand[rng.integers(len(cand))]
                pair = load_pair(it['path'], CORR_MIN_RANGE)
                img0, img1 = load_images(pair)
                results = []
                for name, net in models.items():
                    k0, d0, _, _ = extract(net, img0)
                    k1, d1, _, _ = extract(net, img1)
                    m, det = evaluate_pair(k0, d0, k1, d1, pair['corr_AB'], pair['corr_BA'])
                    results.append((name, k0, k1, det, m))
                title = f'{split}: {it["name"]} ({it["dist"]:.1f} m apart, rotated {it["rot_deg"]:.0f} deg)'
                fig = pair_figure(img0, img1, results, suptitle=title)
                fig.savefig(OUT_DIR / 'examples' / f'{split}_{b}_{f}_{it["name"].replace("/", "_")}.png', dpi=110)
                plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    models, sources = load_models()
    names = list(models)
    rows, rep_dist = run(models)
    summary = summarize(rows, names)
    print('\n' + write_tables(rows, summary, sources))
    for split in SPLITS:
        plot_viewpoint(summary, names, split)
        plot_curves(summary, names, split)
        plot_loc_hist(rep_dist, names, split)
    plot_examples(models)
    timing = run_timing(models)
    print('\n' + write_timing(timing, summary))
    plot_speed(timing, summary, names)
    print(f'\nreport saved to {OUT_DIR}')


if __name__ == '__main__':
    matplotlib.use('Agg')  # files only (not on import: eval_pair / eval_pose_pair show figures)
    main()
