"""Choose the ORB settings for lidar images on the val split (so the ORB baseline is tuned as well as
the learned methods, without looking at the test routes).

Prints one row per setting (metrics averaged over the pairs), sorted by repeatability @3 px; copy the
best into lidar_tools.model.ORB_PARAMS.

Usage: `uv run python evaluation/eval_orb.py`
"""
import sys
import itertools
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lidar_tools import REPO_ROOT  # noqa: E402
from lidar_tools.pairs import split_pairs, load_pair, preprocess  # noqa: E402
from lidar_tools.model import OrbExtractor, extract  # noqa: E402
from lidar_tools.metrics import evaluate_pair, mean_metrics  # noqa: E402

######## Parameters (edit by hand) ########
SPLIT_FILE = REPO_ROOT / 'splits' / 'default.json'
NUM_PAIRS = 100                   # evenly spaced over the val split
TOP_K = 500
CORR_MIN_RANGE = 3.0
GRID = dict(nlevels=[1, 3, 8], scaleFactor=[1.2], patch=[15, 31], fastThreshold=[5, 10, 20])
###########################################

COLUMNS = ['rep@3', 'loc_err', 'num_rep@3', 'MMA@3', 'num_correct@3', 'num_kpts']


def main():
    items = split_pairs(SPLIT_FILE, 'val')
    items = [items[i] for i in np.linspace(0, len(items) - 1, NUM_PAIRS).round().astype(int)]
    images = []
    for it in items:
        p = load_pair(it['path'], CORR_MIN_RANGE)
        images.append((preprocess(p['signal_A'], p['range_A'] > 0), preprocess(p['signal_B'], p['range_B'] > 0),
                       p['corr_AB'], p['corr_BA']))

    rows = []
    for nl, sf, patch, ft in tqdm(list(itertools.product(*GRID.values())), desc='settings'):
        orb = OrbExtractor(TOP_K, nlevels=nl, scaleFactor=sf, edgeThreshold=patch, patchSize=patch, fastThreshold=ft)
        ms = []
        for img0, img1, c01, c10 in images:
            k0, d0, _, _ = extract(orb, img0)
            k1, d1, _, _ = extract(orb, img1)
            ms.append(evaluate_pair(k0, d0, k1, d1, c01, c10)[0])
        rows.append((dict(nlevels=nl, scaleFactor=sf, edgeThreshold=patch, patchSize=patch, fastThreshold=ft),
                     mean_metrics(ms)))

    rows.sort(key=lambda r: -r[1]['rep@3'])
    print(f'\nval: {len(images)} pairs, top {TOP_K} keypoints\n')
    print(f'{"setting":70s}' + ''.join(f'{c:>14s}' for c in COLUMNS))
    for params, m in rows:
        print(f'{str(params):70s}' + ''.join(f'{m[c]:14.3f}' for c in COLUMNS))


if __name__ == '__main__':
    main()
