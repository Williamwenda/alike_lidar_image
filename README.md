# alike_lidar_image

Fine-tuning [ALIKE](https://github.com/Shiaoming/ALIKE) (Zhao et al., IEEE TMM 2022) keypoint detection and
description on low-resolution lidar intensity images (Ouster, 128 x 1024).

Based on the official ALIKE inference code and its released training code (`assets/ALIKE_code.zip` upstream),
ported to Python 3.12 / PyTorch 2.x / PyTorch Lightning 2.x. Original code is BSD-3 licensed, see `LICENSE`.

## Layout

```
alike.py, alnet.py, soft_detect.py   inference model (ALike class, network, differentiable keypoint detection)
demo.py                              run detection + mutual-NN matching on an image folder / video / camera
models/alike-{t,s,n,l}.pth           pretrained weights
lidar_tools/                         lidar pairs, shared by training and evaluation
  geometry.py                        ground-truth lookup in the dense correspondences
  pairs.py                           pair files and their index (distance, facing), preprocessing, splits
  metrics.py                         detection (repeatability, localisation) and matching metrics
  model.py, export.py, plots.py      inference models, checkpoint -> .pth, figures
splits/                              make_splits.py -> default.json (train / val / test, by place)
training_code/
  training/train_lidar.py            fine-tuning on lidar pairs (parameter block at the top)
  training/lidar_wrapper.py          TrainWrapper with lidar validation (same metrics as the test report)
  datasets/lidar_pairs.py            lidar pair dataset (warp mode 'dense'), augmentation
  utils/utils.py                     warp(): 'homo', 'se3' and 'dense' (lidar) modes
  training/train.py, train_wrapper.py, nets/, datasets/{megadepth,hpatches}.py   original ALIKE training
evaluation/
  eval_test.py                       test report: methods vs ground truth on the held-out pairs, per distance /
                                     facing, and inference time (ORB, ALIKE, XFeat: lidar_tools/model.py)
  eval_orb.py                        ORB settings for lidar images, on the val split
  eval_pair.py                       one pair, one or more models: metrics + figures
  eval_pairs.py                      sweep models x channels x preprocessing on the val pairs
tests/test_lidar_pipeline.py         checks of the lidar pipeline on real pairs
tests/smoke_test.py                  end-to-end training check on synthetic homography pairs
```

## Setup

This repo lives in `lidar_image_learning/` next to `xfeat_lidar_image/`; both use the environment of the
parent folder (`lidar_image_learning/pyproject.toml`, `uv.lock`). Requires [uv](https://docs.astral.sh/uv/)
and an NVIDIA GPU.

```shell
cd .. && uv sync && cd -                       # creates lidar_image_learning/.venv; `uv run` finds it from here
cd training_code/utils && uv run bash build_cython.sh && cd -   # optional, faster depth projection
```

## Lidar fine-tuning workflow

Ground truth: `lidar_image_process` (ICP-refined poses, motion distortion removed and re-applied per
column, Ouster beam model, occlusion and cycle checks) gives for every pixel of frame A its sub-pixel
location in frame B: `outputs/<seq>/pairs/<A>_<B>.npz`, `corr_AB` / `corr_BA`, and `index.csv` (distance,
rotation, overlap of every pair). Pairs are chosen by position: 1-15 m apart, any heading, revisits
included. The dense maps are the projection model evaluated at every pixel; a keypoint between pixels
is warped by interpolating them (`lidar_tools/geometry.py`); pixels next to a depth edge get no ground
truth (`lidar_tools.pairs.load_pair`), since interpolating across surfaces is wrong.

```shell
uv run python splits/make_splits.py                      # 1. splits/default.json (after new pairs)
uv run python tests/test_lidar_pipeline.py               # 2. checks of the pipeline
uv run python evaluation/eval_orb.py                     #    ORB settings for lidar images (val split)
uv run python evaluation/eval_test.py                    # 3. baseline report -> evaluation/outputs/test_report_*/
cd training_code/training && uv run python train_lidar.py && cd -   # 4. fine-tune
uv run tensorboard --logdir training_code/training/log_lidar        #    progress, val per distance / facing, figures
uv run python evaluation/eval_test.py                    # 5. report: ORB, ALIKE, XFeat, pretrained / fine-tuned, accuracy + time
uv run python evaluation/eval_pair.py                    # 6. look at any pair, any models
```

Splits (`splits/make_splits.py`), by place: `test` = whole routes never used in training (grassy, dome);
on parking and woody, `val` (checkpoint selection) = the frames within 15 m of one section of the
route, with both passes of a revisit, then a 15 m buffer in no split; the rest is `train`. A pair is in
a split only if both frames are. Results are reported per route, per distance between the scans and per
facing (same / side / opposite).

Metrics (`lidar_tools/metrics.py`; a keypoint has ground truth if its location in the other image is
known, beyond `CORR_MIN_RANGE` = 3 m):

| | metric | meaning |
|---|---|---|
| detection | `rep@t` | repeatability: share of the keypoints with ground truth that are re-detected within t px |
| | `loc_err` | median distance [px] of the re-detected keypoints to their ground-truth location |
| | `num_rep@3` | re-detected keypoints per image |
| | `cov_ratio` | share of the detected keypoints that have ground truth |
| matching | `MMA@t` | precision of the mutual-nearest-neighbour matches within t px |
| | `num_correct@3`, `matching_score@3` | correct matches, correct matches / keypoints with ground truth |

## Other usage

```shell
uv run python demo.py assets/kitti --model alike-n     # inference demo
uv run python tests/smoke_test.py                       # training pipeline check (synthetic)
```

## Changes from upstream training code

- Lightning 2.x API (`Trainer(accelerator, devices)`, `reload_dataloaders_every_n_epochs`, hook signatures).
- `torch.load(weights_only=...)` for torch >= 2.6; `.pth` pretrained weights can be loaded via `pretrained_model`.
- `PeakyLoss` counted keypoints of the first sample for the whole batch, which crashed with `batch_size > 1`.
- matplotlib / numpy 2 / Python 3.12 (`distutils`) compatibility fixes.
- `warp()` mode `'dense'` for lidar pairs (keypoints without a match are ignored, not treated as outside).
