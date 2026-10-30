"""Smoke test for the Lightning 2.x port of ALIKE training_code.

Synthetic homography pairs from assets/kitti, pretrained alike-n weights,
a few train steps + validation on two loaders (exercises the HPatches MMA path).
"""
import sys
from pathlib import Path
import glob
import functools
import logging

ALIKE_ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, ALIKE_ROOT + '/training_code')

import cv2
import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from datasets.cat_datasets import ConcatDatasets
from training.train_wrapper import TrainWrapper
from training.scheduler import WarmupConstantSchedule
from training.train import RebuildDatasetCallback


class SyntheticHomography(Dataset):
    def __init__(self, h=256, w=512, n=8, seed=0):
        self.files = sorted(glob.glob(ALIKE_ROOT + '/assets/kitti/*.png'))
        self.h, self.w, self.n = h, w, n
        self.rng = np.random.default_rng(seed)
        self.built = 0

    def build_dataset(self):
        self.built += 1

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        img = cv2.cvtColor(cv2.imread(self.files[i % len(self.files)]), cv2.COLOR_BGR2RGB)
        img0 = img[:self.h, :self.w]
        src = np.float32([[0, 0], [self.w, 0], [self.w, self.h], [0, self.h]])
        dst = src + self.rng.uniform(-25, 25, src.shape).astype(np.float32)
        H = cv2.getPerspectiveTransform(src, dst).astype(np.float32)
        img1 = cv2.warpPerspective(img0, H, (self.w, self.h))
        to_t = lambda x: x.astype(np.float32).transpose(2, 0, 1) / 255.
        return {'image0': to_t(img0), 'image1': to_t(img1),
                'warp01_params': {'mode': 'homo', 'width': self.w, 'height': self.h, 'homography_matrix': H},
                'warp10_params': {'mode': 'homo', 'width': self.w, 'height': self.h,
                                  'homography_matrix': np.linalg.inv(H).astype(np.float32)}}


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    log_dir = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).parent / 'smoke_logs')
    model = TrainWrapper(c1=16, c2=32, c3=64, c4=128, dim=128, agg_mode='cat', single_head=True, pe=False,
                         radius=2, top_k=200, scores_th=0, n_limit=0, scores_th_eval=0.2, n_limit_eval=5000,
                         train_gt_th=5, eval_gt_th=3, w_pk=0.5, w_rp=1, w_sp=1, w_ds=5, w_triplet=0,
                         sc_th=0.1, norm=1, temp_sp=0.1, temp_ds=0.1, lr=1e-4, log_freq_img=2,
                         pretrained_model=ALIKE_ROOT + '/models/alike-n.pth',
                         lr_scheduler=functools.partial(WarmupConstantSchedule, warmup_steps=2), debug=False)

    train_ds = ConcatDatasets(SyntheticHomography(seed=0), SyntheticHomography(seed=1))
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=2)
    val_loaders = [DataLoader(SyntheticHomography(n=3, seed=s), batch_size=1) for s in (2, 3)]

    logger = TensorBoardLogger(save_dir=log_dir, name='smoke', default_hp_metric=False)
    trainer = pl.Trainer(accelerator='gpu', devices=[0], accumulate_grad_batches=2, num_sanity_val_steps=1,
                         limit_train_batches=6, max_epochs=2, logger=logger, reload_dataloaders_every_n_epochs=1,
                         callbacks=[ModelCheckpoint(monitor='val_metrics/mean', save_top_k=1, mode='max',
                                                    save_last=True, dirpath=logger.log_dir + '/checkpoints',
                                                    auto_insert_metric_name=False,
                                                    filename='epoch={epoch}-mean_metric={val_metrics/mean:.4f}'),
                                    LearningRateMonitor(logging_interval='step'), RebuildDatasetCallback()])
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loaders)
    print('REBUILD CALLS:', [d.built for d in train_ds.datasets])

    # resume path + ckpt reload through the pretrained_model branch
    ckpt = logger.log_dir + '/checkpoints/last.ckpt'
    TrainWrapper.load_from_checkpoint(ckpt, map_location='cpu', weights_only=False)
    m2 = TrainWrapper(c1=16, c2=32, c3=64, c4=128, dim=128, single_head=True,
                      pretrained_model=ckpt, lr_scheduler=model.lr_scheduler)
    print('CKPT RELOAD OK', trainer.callback_metrics.get('val_metrics/mean'))
