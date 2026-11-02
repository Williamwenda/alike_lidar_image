"""Lidar intensity image tools shared by training and evaluation.

geometry : ground-truth lookup in the dense correspondences of lidar_image_process
pairs    : pair files and their index (distance, rotation), preprocessing, train / val / test splits
metrics  : detection (repeatability, localisation) and matching metrics against the ground truth
model    : pretrained or fine-tuned ALIKE for inference
plots    : keypoint and match figures
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))  # inference ALIKE (alike.py, alnet.py, soft_detect.py)
