"""Inspect one lidar intensity image pair: run one or more ALIKE models and check them against the ground truth.

Prints the metrics of every model and saves to OUT_DIR:
    <pair>.png            per model: detections in A and B (green re-detected within 3 px / red missed /
                          grey no ground truth) and matches (green correct / red wrong, yellow circle =
                          where the wrong match should be)
    <pair>_scores.png     score maps of every model

Usage: edit the parameter block, then `uv run python evaluation/eval_pair.py`.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lidar_tools import REPO_ROOT  # noqa: E402
from lidar_tools.pairs import read_index, load_pair, preprocess  # noqa: E402
from lidar_tools.model import make_model, extract  # noqa: E402
from lidar_tools.metrics import evaluate_pair  # noqa: E402
from lidar_tools.plots import pair_figure  # noqa: E402
from eval_test import resolve  # noqa: E402

######## Parameters (edit by hand) ########
SEQ = 'utias_parking'        # utias_dome, utias_parking, utias_woody, utias_grassy
FRAME_A = 3355               # a pair file <FRAME_A>_<FRAME_B>.npz must exist (see <seq>/pairs/index.csv)
FRAME_B = None               # None: the first pair of FRAME_A
                             # (splits/default.json: grassy and dome are the unseen test routes)

# name -> (method, weights): method 'orb', 'alike-n', 'xfeat', 'xfeat-lidar'; weights None = pretrained,
# 'latest' = best.pth of the most recent fine-tuning run of that method
MODELS = {
    'ORB': ('orb', None),
    'ALIKE': ('alike-n', None),
    # 'ALIKE fine-tuned': ('alike-n', 'latest'),
    'XFeat': ('xfeat', None),
    # 'XFeat fine-tuned': ('xfeat-lidar', 'latest'),
}
CHANNEL = 'signal'           # 'signal' or 'reflec' (signal is better zero-shot)
PREPROCESS = 'linear'        # 'linear', 'percentile', 'clahe' (see lidar_tools.pairs.preprocess)
TOP_K = 500                  # > 0: strongest TOP_K keypoints; -1: all with score > 0.2
CORR_MIN_RANGE = 3.0         # ignore ground truth closer than this [m] (less accurate, see lidar_image_process)

SHOW = True                  # open the figures (they are saved either way)
OUT_DIR = REPO_ROOT / 'evaluation' / 'outputs'
###########################################


def main():
    it = next((p for p in read_index(SEQ) if p['frame_A'] == FRAME_A and FRAME_B in (None, p['frame_B'])), None)
    if it is None:
        raise ValueError(f'no pair file of {SEQ} for frames {FRAME_A}, {FRAME_B} (see {SEQ}/pairs/index.csv)')
    pair = load_pair(it['path'], CORR_MIN_RANGE)
    img0 = preprocess(pair[f'{CHANNEL}_A'], pair['range_A'] > 0, PREPROCESS)
    img1 = preprocess(pair[f'{CHANNEL}_B'], pair['range_B'] > 0, PREPROCESS)
    name = it['name'].replace('/', '_')
    title = f'{it["name"]} ({it["dist"]:.1f} m apart, rotated {it["rot_deg"]:.0f} deg)'
    print(f'\n{title}: ground truth for {np.isfinite(pair["corr_AB"]).all(-1).mean():.0%} of the pixels '
          f'(beyond {CORR_MIN_RANGE} m)\n')

    results, score_maps = [], []
    for label, (arch, weights) in MODELS.items():
        weights = resolve(arch, weights)
        net = make_model(arch, weights, top_k=TOP_K)
        k0, d0, _, s0 = extract(net, img0)
        k1, d1, _, s1 = extract(net, img1)
        m, det = evaluate_pair(k0, d0, k1, d1, pair['corr_AB'], pair['corr_BA'])
        results.append((label, k0, k1, det, m))
        if s0 is not None:  # ORB and XFeat give no dense score map
            score_maps.append((label, s0, s1))
        print(f'[{label}] {arch}, {weights or "pretrained weights"}')
        print(f'  detection  keypoints {m["num_kpts"]:.0f}, with ground truth {m["num_cov"]:.0f}, '
              f'repeatability @1/2/3 px {m["rep@1"]:.3f} / {m["rep@2"]:.3f} / {m["rep@3"]:.3f}, '
              f're-detected {m["num_rep@3"]:.0f}, localisation error {m["loc_err"]:.2f} px')
        print(f'  matching   {m["num_matches"]} mutual-NN matches ({m["num_verifiable"]} with ground truth), '
              f'MMA @1/3/5 px {m["MMA@1"]:.3f} / {m["MMA@3"]:.3f} / {m["MMA@5"]:.3f}, '
              f'correct @3 px {m["num_correct@3"]}\n')

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = pair_figure(img0, img1, results, suptitle=title)
    fig.savefig(OUT_DIR / f'{name}.png', dpi=120)

    if score_maps:
        fig2, axes = plt.subplots(2 * len(score_maps), 1, figsize=(16, 2.4 * len(score_maps)), squeeze=False,
                                  constrained_layout=True)
        for i, (label, s0, s1) in enumerate(score_maps):
            for j, (s, ab) in enumerate(((s0, 'A'), (s1, 'B'))):
                ax = axes[2 * i + j, 0]
                ax.imshow(s, cmap='jet', vmin=0, vmax=1)
                ax.set_title(f'[{label}] score map {ab}', fontsize=8, loc='left')
                ax.set_axis_off()
        fig2.savefig(OUT_DIR / f'{name}_scores.png', dpi=120)
    print(f'figures saved to {OUT_DIR}/{name}*.png')
    if SHOW:
        plt.show()


if __name__ == '__main__':
    main()
