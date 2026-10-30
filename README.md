# alike_lidar_image

Fine-tuning [ALIKE](https://github.com/Shiaoming/ALIKE) (Zhao et al., IEEE TMM 2022) keypoint detection and
description on low-resolution lidar intensity images.

Based on the official ALIKE inference code and its released training code (`assets/ALIKE_code.zip` upstream),
ported to Python 3.12 / PyTorch 2.x / PyTorch Lightning 2.x. Original code is BSD-3 licensed, see `LICENSE`.

## Layout

```
alike.py, alnet.py, soft_detect.py   inference model (ALike class, network, differentiable keypoint detection)
demo.py                              run detection + mutual-NN matching on an image folder / video / camera
models/alike-{t,s,n,l}.pth           pretrained weights
assets/kitti/                        sample frames for the demo and smoke test
training_code/
  training/train.py                  training entry point (configs at the top of __main__)
  training/train_wrapper.py          LightningModule: correspondences, losses, validation metrics
  nets/                              training copy of the network + losses
  datasets/                          MegaDepth (train), HPatches (val) datasets
  utils/                             geometry / warping utilities, Cython depth projection
tests/smoke_test.py                  end-to-end training check on synthetic homography pairs
```

## Setup

Requires [uv](https://docs.astral.sh/uv/) and an NVIDIA GPU.

```shell
uv sync                                        # creates .venv from uv.lock
cd training_code/utils && uv run bash build_cython.sh && cd -   # optional, faster depth projection
```

## Usage

```shell
uv run python demo.py assets/kitti --model alike-n     # inference demo
uv run python tests/smoke_test.py                       # training pipeline check
cd training_code/training && uv run python train.py     # training (set dataset paths in train.py)
```

## Changes from upstream training code

- Lightning 2.x API (`Trainer(accelerator, devices)`, `reload_dataloaders_every_n_epochs`, hook signatures).
- `torch.load(weights_only=...)` for torch >= 2.6; `.pth` pretrained weights can be loaded via `pretrained_model`.
- `PeakyLoss` counted keypoints of the first sample for the whole batch, which crashed with `batch_size > 1`.
- matplotlib / numpy 2 / Python 3.12 (`distutils`) compatibility fixes.
