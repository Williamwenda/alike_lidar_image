"""ALIKE fine-tuning on lidar intensity image pairs: TrainWrapper with lidar validation.

Training is ALIKE's (losses on correspondences from utils.warp_dense). Validation runs the same
inference path as ALike.forward (8-bit image, dense maps, DKD with a fixed top_k, sub-pixel) and the
same metrics as evaluation/eval_test.py (lidar_tools.metrics), so validation and test numbers of a
checkpoint are directly comparable.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
from lidar_tools.metrics import evaluate_pair, mean_metrics  # noqa: E402
from lidar_tools.plots import pair_figure, figure_to_array  # noqa: E402
from soft_detect import DKD  # noqa: E402  (inference detector, repo root)
from training.train_wrapper import TrainWrapper  # noqa: E402

LOGGED = ['rep@1', 'rep@2', 'rep@3', 'loc_err', 'num_rep@3', 'cov_ratio',
          'MMA@1', 'MMA@3', 'num_correct@3', 'matching_score@3']


class LidarTrainWrapper(TrainWrapper):
    def __init__(self, *args, eval_top_k: int = 500, freeze: tuple = (), num_val_figures: int = 4, **kwargs):
        """
        :param eval_top_k: keypoints per image in validation (same as evaluation/eval_test.py)
        :param freeze: parameter name prefixes not to train, e.g. ('block1', 'block2')
        :param num_val_figures: validation pairs drawn to TensorBoard every epoch
        """
        super().__init__(*args, **kwargs)
        self.eval_top_k = eval_top_k
        self.num_val_figures = num_val_figures
        self.dkd_eval = DKD(radius=self.radius, top_k=eval_top_k, scores_th=self.scores_th_eval,
                            n_limit=self.n_limit_eval)
        for name, p in self.named_parameters():
            if any(name.startswith(f) for f in freeze):
                p.requires_grad = False
        self.val_results = []

    @torch.no_grad()
    def infer(self, image):
        """ALike.forward on a 1x3xHxW image: keypoints (N, 2) px, descriptors (N, D), as numpy."""
        h, w = image.shape[2:]
        descriptor_map, scores_map = self.extract_dense_map(image)
        kpts, desc, _, _ = self.dkd_eval(scores_map, descriptor_map, sub_pixel=True)
        kpts = (kpts[0] + 1) / 2 * kpts[0].new_tensor([[w - 1, h - 1]])
        return kpts.cpu().numpy(), desc[0].cpu().numpy()

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        kpts0, desc0 = self.infer(batch['image0'])
        kpts1, desc1 = self.infer(batch['image1'])
        corr01 = batch['warp01_params']['corr'][0].cpu().numpy()
        corr10 = batch['warp10_params']['corr'][0].cpu().numpy()
        m, details = evaluate_pair(kpts0, desc0, kpts1, desc1, corr01, corr10)

        name = batch['name'][0]
        self.val_results.append(({'dist': batch['dist_bin'][0], 'facing': batch['facing'][0]}, m))

        if batch_idx < self.num_val_figures and not self.trainer.sanity_checking:
            to_img = lambda t: (t[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
            fig = pair_figure(to_img(batch['image0']), to_img(batch['image1']),
                              [('this checkpoint', kpts0, kpts1, details, m)], suptitle=name)
            self.logger.experiment.add_image(f'val_pairs/{batch_idx}', figure_to_array(fig),
                                             global_step=self.global_step, dataformats='HWC')

    def on_validation_epoch_start(self):
        self.val_results = []

    def on_validation_epoch_end(self):
        if not self.val_results:
            return
        mean = mean_metrics([m for _, m in self.val_results])
        for k in LOGGED:
            self.log(f'val/{k}', mean[k])
        # per viewpoint change: val_dist_<bin>/..., val_facing_<same|side|opposite>/...
        for key in ('dist', 'facing'):
            for group in sorted({g[key] for g, _ in self.val_results}):
                mean = mean_metrics([m for g, m in self.val_results if g[key] == group])
                for k in ('rep@3', 'loc_err', 'MMA@3'):
                    self.log(f'val_{key}_{group}/{k}', mean[k])
