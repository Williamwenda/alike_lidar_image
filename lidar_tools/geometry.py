"""Ground-truth correspondences of lidar intensity image pairs.

lidar_image_process (make_pairs.py) gives, for every pixel of A, its sub-pixel (row, col) in B:
the undistorted 3-D point of the pixel, moved by the ICP-refined relative pose, re-distorted with
B's per-column motion and projected with the Ouster beam model, kept if it passes the range
(occlusion) check and the A -> B -> A cycle check. So corr_AB is the projection model evaluated at
every pixel; a keypoint between pixels is warped by interpolating it (validated: A -> B -> A of
random sub-pixel points comes back within 0.06 px median).

Conventions:
    - integer coordinates are pixel centres (same as ALIKE's keypoints).
    - keypoints are (x, y) = (col, row); corr is (row, col).
    - columns wrap around 360 deg: col is in [0, W), and columns W-1 and 0 are neighbours.
"""
import numpy as np


def wrap_col(dcol, W):
    """Column difference wrapped to [-W/2, W/2)."""
    return (dcol + W / 2) % W - W / 2


def flow_from_corr(corr):
    """(H, W, 2) correspondences (row, col) -> flow (drow, dcol) with the column flow wrapped."""
    H, W = corr.shape[:2]
    rows, cols = np.mgrid[0:H, 0:W]
    return np.stack([corr[..., 0] - rows, wrap_col(corr[..., 1] - cols, W)], axis=-1)


def gt_warp(kpts_xy, corr):
    """Ground-truth location in the other image of sub-pixel keypoints.

    Bilinear interpolation of the flow (corr - pixel), with the column flow wrapped, so a point next
    to the azimuth seam is not interpolated between col W-1 and col 0. The column flows of the four
    neighbours are unwrapped relative to the nearest one before interpolating: for scans rotated by
    ~180 deg the flow is near +-W/2, and neighbours on both sides of the wrap must not average to 0.
    A keypoint gets NaN if one of its neighbouring pixels (with non-zero weight) is unmatched: no
    interpolation across object edges or missing returns.

    :param kpts_xy: (N, 2) keypoints (x=col, y=row) in this image
    :param corr: (H, W, 2) (row, col) in the other image for every pixel of this image
    :return: (N, 2) warped keypoints (x=col, y=row) in the other image, NaN = no ground truth
    """
    H, W = corr.shape[:2]
    flow = flow_from_corr(corr)

    x = np.mod(kpts_xy[:, 0], W)
    y = np.clip(kpts_xy[:, 1], 0, H - 1)
    x0 = np.floor(x).astype(int)
    y0 = np.minimum(np.floor(y).astype(int), H - 2)
    x1 = (x0 + 1) % W  # columns wrap
    y1 = y0 + 1
    ax, ay = (x - x0)[:, None], (y - y0)[:, None]
    ref = flow[np.where(ay[:, 0] < 0.5, y0, y1), np.where(ax[:, 0] < 0.5, x0, x1), 1]  # nearest neighbour

    # a neighbour with zero weight (keypoint exactly on a pixel row / column) does not count
    f = np.zeros((len(x), 2))
    for yy, xx, wgt in ((y0, x0, (1 - ay) * (1 - ax)), (y0, x1, (1 - ay) * ax),
                        (y1, x0, ay * (1 - ax)), (y1, x1, ay * ax)):
        fl = flow[yy, xx].copy()
        fl[:, 1] = ref + wrap_col(fl[:, 1] - ref, W)
        f = f + np.where(wgt > 0, wgt * fl, 0.)
        f[(wgt[:, 0] > 0) & ~np.isfinite(fl).all(1)] = np.nan  # unmatched neighbour

    return np.stack([np.mod(x + f[:, 1], W), y + f[:, 0]], axis=1)


def pixel_dist(a_xy, b_xy, W):
    """Pairwise distance [px] between (N, 2) and (M, 2) keypoints, columns wrapping around."""
    dx = wrap_col(a_xy[:, None, 0] - b_xy[None, :, 0], W)
    dy = a_xy[:, None, 1] - b_xy[None, :, 1]
    return np.hypot(dx, dy)
