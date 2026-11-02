"""Grid of match figures (as in the XFeat paper, Fig. 1): one column per test pair, one row per method;
green lines = correct matches (<= 3 px from the ground truth), banner = metrics of that pair.
Ground truth covers only part of each scan (surfaces seen by both, no sky, no depth edges, beyond 3 m):
keypoints and matches elsewhere cannot be checked; SHOW_KPTS / SHOW_UNVERIFIABLE / DIM_NO_GT show them.

Pairs: PAIRS by name, or one per GROUPS entry chosen from per_pair.csv of the latest test report
(evaluation/eval_test.py): the pair of that distance bin / facing whose REF_MODEL number of correct
matches is closest to the group median, so the columns are typical pairs, not the best ones.

Saves OUT_DIR/fig_matches.png and .pdf.

Usage: edit the parameter block, then `uv run python evaluation/fig_matches.py`.
"""
import sys
import csv
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.ndimage import binary_closing, binary_opening

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lidar_tools import REPO_ROOT  # noqa: E402
from lidar_tools.pairs import split_pairs, load_pair  # noqa: E402
from lidar_tools.model import make_model, extract  # noqa: E402
from lidar_tools.metrics import evaluate_pair  # noqa: E402
from lidar_tools.plots import match_panel  # noqa: E402
from eval_test import resolve, load_images, SPLIT_FILE, CORR_MIN_RANGE, TOP_K  # noqa: E402

######## Parameters (edit by hand) ########
# rows: name -> (method, weights), as in eval_test.py
MODELS = {
    'ORB': ('orb', None),
    'ALIKE': ('alike-n', None),
    'ALIKE fine-tuned': ('alike-n', 'latest'),
    'XFeat': ('xfeat', None),
    'XFeat fine-tuned': ('xfeat-lidar', 'latest'),
}
PAIRS = []                   # pair names, e.g. ['utias_dome/000500_000560']; empty = chosen by GROUPS
GROUPS = [('1-3m', 'same'), ('6-10m', 'same'), (None, 'side')]   # (distance bin, facing), None = any
REF_MODEL = 'ALIKE fine-tuned'   # typical pair = median number of correct matches of this model
SPLIT = 'test'
SHOW_KPTS = True             # every keypoint as a dot: cyan re-detected (<= 3 px), orange not, white no ground truth
SHOW_WRONG = False           # also draw the wrong matches (red)
SHOW_UNVERIFIABLE = False    # also draw the matches without ground truth (grey, can be right or wrong)
DIM_NO_GT = False             # darken the regions without ground truth (where nothing can be checked)
REPORT_DIR = None            # test report with per_pair.csv; None = the latest evaluation/outputs/test_report_*
OUT_DIR = REPO_ROOT / 'evaluation' / 'outputs'
###########################################


def gt_region(corr):
    """Pixels with ground truth, small holes filled (the per-pixel mask is speckled: dimming it is unreadable)."""
    return binary_opening(binary_closing(np.isfinite(corr).all(-1), np.ones((5, 9))), np.ones((3, 5)))


def choose_pairs():
    if PAIRS:
        return PAIRS
    report = Path(REPORT_DIR) if REPORT_DIR else sorted((REPO_ROOT / 'evaluation' / 'outputs').glob('test_report_*'))[-1]
    with open(report / 'per_pair.csv') as f:
        rows = [r for r in csv.DictReader(f) if r['split'] == SPLIT and r['model'] == REF_MODEL]
    names = []
    for dist, face in GROUPS:
        group = [r for r in rows if dist in (None, r['dist_bin']) and face in (None, r['facing'])]
        correct = np.array([float(r['num_correct@3']) for r in group])
        names.append(group[int(np.argmin(np.abs(correct - np.median(correct))))]['pair'])
    print(f'pairs from {report / "per_pair.csv"}: {names}')
    return names


def main():
    items = {it['name']: it for it in split_pairs(SPLIT_FILE, SPLIT)}
    pairs = [items[n] for n in choose_pairs()]
    data = [(it, load_pair(it['path'], CORR_MIN_RANGE)) for it in pairs]
    data = [(it, p, *load_images(p)) for it, p in data]

    H, W = data[0][2].shape[:2]
    panel_w = 6.0                                     # inches per column
    panel_h = panel_w * (2 * H + 16) / W
    fig, axes = plt.subplots(len(MODELS), len(data), squeeze=False,
                             figsize=(panel_w * len(data), (panel_h + 0.04) * len(MODELS) + 0.35),
                             gridspec_kw=dict(wspace=0.02, hspace=0.04, left=0.005, right=0.995, bottom=0.005,
                                              top=1 - 0.35 / ((panel_h + 0.04) * len(MODELS) + 0.35)))
    for row, (label, (arch, weights)) in enumerate(MODELS.items()):
        net = make_model(arch, resolve(arch, weights), top_k=TOP_K)
        for col, (it, pair, img0, img1) in enumerate(data):
            k0, d0, _, _ = extract(net, img0)
            k1, d1, _, _ = extract(net, img1)
            m, det = evaluate_pair(k0, d0, k1, d1, pair['corr_AB'], pair['corr_BA'])
            banner = (f'matches {m["num_matches"]:3d} ({m["num_verifiable"]:2d} checkable) | correct {m["num_correct@3"]:2d} | '
                      f'MMA@3 {m["MMA@3"]:.2f} | rep@3 {m["rep@3"]:.2f}')
            masks = (gt_region(pair['corr_AB']), gt_region(pair['corr_BA'])) if DIM_NO_GT else None
            match_panel(axes[row, col], img0, img1, k0, k1, det, label, banner, show_wrong=SHOW_WRONG,
                        show_unverifiable=SHOW_UNVERIFIABLE, show_kpts=SHOW_KPTS, gt_masks=masks)
            if row == 0:
                axes[row, col].set_title(f'{it["name"]}: {it["dist"]:.1f} m apart, rotated {it["rot_deg"]:.0f} deg',
                                         fontsize=9)
        print(f'{label} done')

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(OUT_DIR / f'fig_matches.{ext}', dpi=200)
    print(f'saved {OUT_DIR}/fig_matches.png / .pdf')


if __name__ == '__main__':
    matplotlib.use('Agg')
    main()
