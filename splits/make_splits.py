"""Make the train / val / test split file of the lidar pairs, by place.

Default (splits/default.json):
    TEST_SEQS (grassy, dome)    -> 'test': whole routes never used in training (unseen places)
    every other route (parking, woody):
        VAL_WINDOWS             -> 'val' (checkpoint selection): the frames within REGION_RADIUS of a
                                   section of the route (a fraction of its length), so a place passed
                                   twice (revisit, often facing the other way) is in val with both passes
        BUFFER beyond that      -> no split, so no training frame is near a val place
        the rest                -> 'train'
    A pair belongs to a split only if both of its frames do.

Usage: `uv run python splits/make_splits.py`
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lidar_tools.pairs import PAIRS_ROOT, FACING, DIST_BINS, read_index, dist_bin  # noqa: E402

######## Parameters (edit by hand) ########
TRAIN_SEQS = ['utias_parking', 'utias_woody']
TEST_SEQS = ['utias_grassy', 'utias_dome']
VAL_WINDOWS = {'utias_parking': (0.44, 0.50),    # around the far turnaround, both passes
               'utias_woody': (0.02, 0.08)}      # near the start, passed again at the end
REGION_RADIUS = 15.0   # [m] frames this close to the val section are val (= the largest pair distance)
BUFFER = 15.0          # [m] frames up to REGION_RADIUS + BUFFER from it are in no split
OUT = Path(__file__).resolve().parent / 'default.json'
###########################################


def frame_positions(pairs):
    """Frame ids (sorted) and their (x, y) from the pairs of a route."""
    pos = {}
    for p in pairs:
        pos[p['frame_A']], pos[p['frame_B']] = p['xy_A'], p['xy_B']
    frames = np.array(sorted(pos))
    return frames, np.array([pos[f] for f in frames])


def frame_labels(pairs, window):
    """'val', 'buffer' or 'train' for every frame of a route."""
    frames, xy = frame_positions(pairs)
    along = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    along /= along[-1]  # fraction of the route length, in frame order
    section = xy[(along >= window[0]) & (along < window[1])]
    d = np.min(np.linalg.norm(xy[:, None] - section[None], axis=-1), axis=1)
    labels = np.where(d <= REGION_RADIUS, 'val', np.where(d <= REGION_RADIUS + BUFFER, 'buffer', 'train'))
    return dict(zip(frames.tolist(), labels))


def main():
    splits = {'train': [], 'val': [], 'test': []}
    dropped = 0
    for seq in TEST_SEQS:
        splits['test'] += [p['name'] for p in read_index(seq)]
    for seq in TRAIN_SEQS:
        pairs = read_index(seq)
        labels = frame_labels(pairs, VAL_WINDOWS[seq])
        for p in pairs:
            a, b = labels[p['frame_A']], labels[p['frame_B']]
            if a == b and a != 'buffer':
                splits[a].append(p['name'])
            else:
                dropped += 1

    spec = {'pairs_root': str(PAIRS_ROOT),
            'description': f'test = whole routes {TEST_SEQS}; {TRAIN_SEQS}: val = frames within {REGION_RADIUS} m of '
                           f'the route sections {VAL_WINDOWS} (fractions of the route length), then a {BUFFER} m '
                           f'buffer in no split, the rest train; a pair is in a split if both frames are',
            'splits': splits}
    OUT.write_text(json.dumps(spec, indent=2))
    print(f'saved {OUT} ({dropped} pairs of {TRAIN_SEQS} in no split: buffer, or frames in two splits)\n')

    index = {p['name']: p for seq in TRAIN_SEQS + TEST_SEQS for p in read_index(seq)}
    bins = [dist_bin(0.5 * (lo + hi)) for lo, hi in zip(DIST_BINS[:-1], DIST_BINS[1:])]
    print(f'{"split":8s}{"pairs":>7s}' + ''.join(f'{b:>8s}' for b in bins) + ''.join(f'{f:>10s}' for f, _, _ in FACING))
    for split, names in splits.items():
        ps = [index[n] for n in names]
        print(f'{split:8s}{len(ps):7d}' + ''.join(f'{sum(p["dist_bin"] == b for p in ps):8d}' for b in bins)
              + ''.join(f'{sum(p["facing"] == f for p in ps):10d}' for f, _, _ in FACING))


if __name__ == '__main__':
    main()
