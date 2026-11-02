"""Sweep: pretrained ALIKE models x image channels x preprocessing over the validation pairs.

Prints one row per configuration (metrics averaged over pairs) and saves the table as CSV.
Runs on the 'val' split, so the choice of input settings never looks at the test pairs.
For the test report of a fine-tuned model, use eval_test.py.

Usage: edit the parameter block, then `uv run python evaluation/eval_pairs.py`.
"""
import sys
import time
import itertools
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lidar_tools import REPO_ROOT  # noqa: E402
from lidar_tools.pairs import split_pairs, load_pair, preprocess  # noqa: E402
from lidar_tools.model import make_model, extract  # noqa: E402
from lidar_tools.metrics import evaluate_pair  # noqa: E402

######## Parameters (edit by hand) ########
SPLIT_FILE = REPO_ROOT / 'splits' / 'default.json'
SPLIT = 'val'
PAIRS_PER_SEQ = 20           # evenly spaced over each sequence (None = all pairs)

MODELS = ['alike-t', 'alike-s', 'alike-n', 'alike-l']
CHANNELS = ['reflec', 'signal']
PREPROCESS = ['linear', 'percentile', 'clahe']
GAMMA = 1.0
TOP_K = 500
SCORES_TH = 0.2
SUB_PIXEL = True
CORR_MIN_RANGE = 3.0         # ignore ground truth closer than this [m]

OUT_CSV = REPO_ROOT / 'evaluation' / 'outputs' / 'sweep_val.csv'
###########################################

COLUMNS = ['rep@3', 'loc_err', 'MMA@1', 'MMA@3', 'MMA@5', 'num_correct@3', 'matching_score@3', 'num_matches',
           'num_cov']


def pair_files():
    items = split_pairs(SPLIT_FILE, SPLIT)
    files = []
    for seq in sorted({it['seq'] for it in items}):
        fs = [it['path'] for it in items if it['seq'] == seq]
        if PAIRS_PER_SEQ is not None and len(fs) > PAIRS_PER_SEQ:
            fs = [fs[i] for i in np.linspace(0, len(fs) - 1, PAIRS_PER_SEQ).round().astype(int)]
        files += [(seq, f) for f in fs]
    return files, sorted({it['seq'] for it in items})


def main():
    files, seqs = pair_files()
    models = {name: make_model(name, top_k=TOP_K, scores_th=SCORES_TH) for name in MODELS}
    configs = list(itertools.product(MODELS, CHANNELS, PREPROCESS))
    results = {cfg: {} for cfg in configs}  # cfg -> seq -> list of metric dicts
    t0 = time.time()

    for seq, f in tqdm(files, desc='pairs'):
        pair = load_pair(f, CORR_MIN_RANGE)
        for model_name, channel, prep in configs:
            img0 = preprocess(pair[f'{channel}_A'], pair['range_A'] > 0, prep, GAMMA)
            img1 = preprocess(pair[f'{channel}_B'], pair['range_B'] > 0, prep, GAMMA)
            kpts0, desc0, _, _ = extract(models[model_name], img0, SUB_PIXEL)
            kpts1, desc1, _, _ = extract(models[model_name], img1, SUB_PIXEL)
            m, _ = evaluate_pair(kpts0, desc0, kpts1, desc1, pair['corr_AB'], pair['corr_BA'])
            results[(model_name, channel, prep)].setdefault(seq, []).append(m)

    # ---------- table: mean over all pairs, plus MMA@3 per sequence
    rows = []
    for cfg in configs:
        all_m = [m for ms in results[cfg].values() for m in ms]
        row = {'model': cfg[0], 'channel': cfg[1], 'preprocess': cfg[2]}
        row.update({c: float(np.nanmean([m[c] for m in all_m])) for c in COLUMNS})
        row.update({f'MMA@3_{seq.replace("utias_", "")}': float(np.mean([m['MMA@3'] for m in results[cfg][seq]]))
                    for seq in seqs})
        rows.append(row)
    rows.sort(key=lambda r: -r['MMA@3'])

    keys = list(rows[0].keys())
    print(f'\n{SPLIT}: {len(files)} pairs ({PAIRS_PER_SEQ} per sequence), top_k {TOP_K}, GT beyond {CORR_MIN_RANGE} m, '
          f'{time.time() - t0:.0f} s\n')
    print(' '.join(f'{k:>10s}' if i > 2 else f'{k:<10s}' for i, k in enumerate(keys)))
    for r in rows:
        print(' '.join(f'{r[k]:>10.3f}' if isinstance(r[k], float) and i > 2 else f'{str(r[k]):<10s}'
                       for i, k in enumerate(keys)))

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, 'w') as fp:
        fp.write(','.join(keys) + '\n')
        for r in rows:
            fp.write(','.join(f'{r[k]:.4f}' if isinstance(r[k], float) else str(r[k]) for k in keys) + '\n')
    print(f'\nsaved {OUT_CSV}')


if __name__ == '__main__':
    main()
