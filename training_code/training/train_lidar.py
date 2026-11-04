"""Fine-tune pretrained ALIKE on lidar intensity image pairs.

    cd training_code/training && uv run python train_lidar.py

1. validates the pretrained model (step 0 in TensorBoard = the baseline),
2. fine-tunes on the 'train' split, validating every epoch on 'val' (also per distance bin and facing),
3. exports the best and the last checkpoint to .pth for lidar_image_nn_evaluation/eval_features.py.

TensorBoard: `uv run tensorboard --logdir training_code/training/log_lidar`
"""
import os
import sys
import time
import logging
import functools
from pathlib import Path

import matplotlib

matplotlib.use('Agg')

sys.path.append('../')

import torch  # noqa: E402
import pytorch_lightning as pl  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from pytorch_lightning.loggers import TensorBoardLogger  # noqa: E402
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint  # noqa: E402

from datasets.lidar_pairs import LidarPairDataset  # noqa: E402
from training.lidar_wrapper import LidarTrainWrapper  # noqa: E402
from training.scheduler import WarmupConstantSchedule  # noqa: E402
from training.export import ckpt_to_pth  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]

######## Parameters (edit by hand) ########
# ---------- model: start from the pretrained weights
MODEL = 'alike-n'                  # architecture of the pretrained weights below
PRETRAINED = REPO_ROOT / 'models' / f'{MODEL}.pth'
ARCH = {'alike-t': dict(c1=8, c2=16, c3=32, c4=64, dim=64, single_head=True),
        'alike-s': dict(c1=8, c2=16, c3=48, c4=96, dim=96, single_head=True),
        'alike-n': dict(c1=16, c2=32, c3=64, c4=128, dim=128, single_head=True),
        'alike-l': dict(c1=32, c2=64, c3=128, c4=128, dim=128, single_head=False)}[MODEL]
FREEZE = ()                        # e.g. ('block1', 'block2') to keep the first layers fixed

# ---------- data
SPLIT_FILE = REPO_ROOT.parent / 'lidar_image_nn_evaluation' / 'splits' / 'default.json'
CHANNEL = 'signal'                 # 'signal' or 'reflec'
CORR_MIN_RANGE = 3.0               # [m] no ground truth closer than this

# ---------- detection / losses (ALIKE defaults unless noted)
TOP_K = 400                        # keypoints per image in training (+ as many random points)
EVAL_TOP_K = 500                   # keypoints per image in validation (as lidar_image_nn_evaluation/eval_features.py)
TRAIN_GT_TH = 3                    # [px] ALIKE: 5 (480 px images); a lidar row is ~0.35 deg
W_PK, W_RP, W_SP, W_DS = 0.5, 1.0, 1.0, 5.0   # peaky, reprojection (keypoint position), score map, descriptor
SC_TH, NORM, TEMP_SP, TEMP_DS = 0.1, 1, 0.1, 0.1

# ---------- optimisation
LR = 1e-4                          # ALIKE from scratch: 3e-4
WARMUP_STEPS = 200
BATCH_SIZE = 2
ACCUMULATE = 4                     # effective batch of 8 pairs
MAX_EPOCHS = 30
NUM_WORKERS = 8
MONITOR = 'val/rep@3'              # checkpoint selection: detection repeatability (see lidar_tools.metrics)
GPUS = [0]
LOG_DIR = Path(__file__).resolve().parent / 'log_lidar'
###########################################


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    torch.set_float32_matmul_precision('high')

    model = LidarTrainWrapper(
        **ARCH, 
        agg_mode='cat', 
        pe=False,
        radius=2, 
        top_k=TOP_K, 
        scores_th=0, 
        n_limit=0, 
        scores_th_eval=0.2, 
        n_limit_eval=5000,
        train_gt_th=TRAIN_GT_TH, 
        eval_gt_th=3,
        w_pk=W_PK, 
        w_rp=W_RP, 
        w_sp=W_SP, 
        w_ds=W_DS, 
        w_triplet=0,
        sc_th=SC_TH, 
        norm=NORM, 
        temp_sp=TEMP_SP, 
        temp_ds=TEMP_DS,
        lr=LR, 
        log_freq_img=500, 
        pretrained_model=str(PRETRAINED),
        lr_scheduler=functools.partial(WarmupConstantSchedule, warmup_steps=WARMUP_STEPS),
        eval_top_k=EVAL_TOP_K, 
        freeze=FREEZE)

    # training and validation datasets
    train_set = LidarPairDataset(SPLIT_FILE, 'train', channel=CHANNEL, corr_min_range=CORR_MIN_RANGE, augment=True)
    val_set = LidarPairDataset(SPLIT_FILE, 'val', channel=CHANNEL, corr_min_range=CORR_MIN_RANGE)
    # training and validation data loaders
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                              pin_memory=True, drop_last=True, persistent_workers=True)
    val_loader = DataLoader(val_set, batch_size=1, num_workers=NUM_WORKERS, persistent_workers=True)
    logging.info(f'{len(train_set)} training pairs, {len(val_set)} validation pairs')

    version = time.strftime('%m%d-%H%M%S')
    logger = TensorBoardLogger(save_dir=str(LOG_DIR), name=MODEL, version=version, default_hp_metric=False)
    # save the checkpoint:
    # the checkpoint are selected based on the monitored metric (MONITOR) and saved in ckpt_dir
    ckpt_dir = Path(logger.log_dir) / 'checkpoints'
    checkpoint = ModelCheckpoint(
        monitor=MONITOR,                 # metric to monitor for saving the best checkpoint
        mode='max',                      # mode for the monitored metric ('max' means higher is better)
        save_top_k=2,                    # number of top checkpoints to save
        save_last=True,                  # whether to save the last checkpoint
        dirpath=ckpt_dir,                # directory to save the checkpoints
        auto_insert_metric_name=False,   # whether to automatically insert the metric name into the filename
        filename='epoch={epoch}-' + MONITOR.replace('/', '_') + '={' + MONITOR + ':.4f}'  # filename format for the checkpoint
        )
    
    # initialize the PyTorch Lightning trainer 
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=GPUS, 
        max_epochs=MAX_EPOCHS,
        accumulate_grad_batches=ACCUMULATE, 
        num_sanity_val_steps=0, 
        logger=logger,
        callbacks=[checkpoint, LearningRateMonitor(logging_interval='step')]
    )

    trainer.validate(model, val_loader)  # pretrained baseline at step 0 (all val pairs, also in a quick check)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    best = ckpt_to_pth(checkpoint.best_model_path, MODEL, ckpt_dir / 'best.pth')
    last = ckpt_to_pth(checkpoint.last_model_path, MODEL, ckpt_dir / 'last.pth')
    logging.info(f'best ({MONITOR} = {checkpoint.best_model_score:.4f}): {best}\nlast: {last}')
