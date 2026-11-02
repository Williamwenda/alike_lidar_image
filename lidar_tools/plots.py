"""Keypoint and match figures of a lidar image pair (128 x 1024 images: stacked vertically)."""
import numpy as np
import matplotlib.pyplot as plt

from lidar_tools.metrics import LOC_TH

GREEN, RED, GREY, YELLOW = (0.1, 0.85, 0.2), (0.95, 0.1, 0.1), (0.6, 0.6, 0.6), 'yellow'


def draw_detections(ax, img, kpts, d, title=''):
    """Keypoints of one image, coloured by detection quality.

    d: per keypoint, distance [px] from its ground-truth location in the other image to the nearest
    keypoint detected there (metrics.evaluate_pair details 'd0' / 'd1'), NaN = no ground truth.
    green: re-detected within LOC_TH px; red: not re-detected; grey: no ground truth
    """
    ax.imshow(img)
    cov = np.isfinite(d)
    rep = cov & (d <= LOC_TH)
    miss = cov & ~rep
    ax.scatter(kpts[~cov, 0], kpts[~cov, 1], s=4, color=GREY)
    ax.scatter(kpts[miss, 0], kpts[miss, 1], s=7, color=RED)
    ax.scatter(kpts[rep, 0], kpts[rep, 1], s=7, color=GREEN)
    ax.set_title(f'{title}   re-detected (<= {LOC_TH:g} px): {rep.sum()}   missed: {miss.sum()}   '
                 f'no ground truth: {(~cov).sum()}', fontsize=8, loc='left')
    ax.set_axis_off()


def draw_matches(ax, img0, img1, kpts0, kpts1, details, err_th=3.0, show_unverifiable=False,
                 show_gt_of_wrong=True, title=''):
    """A on top, B below; a line joins the two keypoints of every mutual-NN match.

    green: the B keypoint is within err_th px of the ground-truth location of the A keypoint
    red:   it is further away (yellow circle = where it should be, if show_gt_of_wrong)
    grey:  the A keypoint has no ground truth (only if show_unverifiable)
    """
    H, W = img0.shape[:2]
    gap = 40  # px between the images, so the direction of the lines is visible
    off = H + gap
    canvas = np.full((2 * H + gap, W, 3), 255, np.uint8)
    canvas[:H], canvas[off:] = img0, img1
    ax.imshow(canvas)

    m, err, gt = details['matches'], details['err'], details['gt']
    p0, p1 = kpts0[m[:, 0]], kpts1[m[:, 1]]
    unverifiable, wrong, correct = ~np.isfinite(err), err > err_th, err <= err_th
    for sel, color, draw in ((unverifiable, GREY, show_unverifiable), (wrong, RED, True), (correct, GREEN, True)):
        if not draw:
            continue
        # matches across the azimuth seam would cross the whole image: draw their end points only
        straight = sel & (np.abs(p0[:, 0] - p1[:, 0]) < W / 2)
        ax.plot(np.stack([p0[straight, 0], p1[straight, 0]]), np.stack([p0[straight, 1], p1[straight, 1] + off]),
                color=color, lw=1.0, alpha=0.85)
        ax.scatter(p0[sel, 0], p0[sel, 1], s=6, color=color, zorder=3)
        ax.scatter(p1[sel, 0], p1[sel, 1] + off, s=6, color=color, zorder=3)
    if show_gt_of_wrong:
        ax.scatter(gt[wrong, 0], gt[wrong, 1] + off, s=22, facecolors='none', edgecolors=YELLOW, lw=0.8, zorder=4)
    ax.text(-8, H / 2, 'A', ha='right', va='center', fontsize=10)
    ax.text(-8, off + H / 2, 'B', ha='right', va='center', fontsize=10)
    ax.set_title(f'{title}   matches: correct (<= {err_th:g} px) {correct.sum()}   wrong {wrong.sum()}   '
                 f'no ground truth {unverifiable.sum()}' + ('' if show_unverifiable else ' (not drawn)'),
                 fontsize=8, loc='left')
    ax.set_axis_off()


def match_panel(ax, img0, img1, kpts0, kpts1, details, label, banner, err_th=3.0, show_wrong=False,
                show_unverifiable=False, show_kpts=False, gt_masks=None, gap=16):
    """Compact match panel for figure grids (as in the XFeat paper): A on top, B below, a green line per
    correct match (<= err_th px from the ground truth); banner text on a dark strip at the top left, the
    method name boxed at the top right.
    show_wrong: red lines for the wrong matches; show_unverifiable: grey lines for the matches without
    ground truth; show_kpts: every keypoint as a dot (cyan: re-detected in the other image within LOC_TH px,
    orange: not re-detected, white: no ground truth); gt_masks: (mask A, mask B) of the pixels with ground
    truth, the others are dimmed."""
    H, W = img0.shape[:2]
    off = H + gap
    canvas = np.full((2 * H + gap, W, 3), 255, np.uint8)
    canvas[:H], canvas[off:] = img0, img1
    if gt_masks is not None:
        canvas[:H][~gt_masks[0]] //= 3
        canvas[off:][~gt_masks[1]] //= 3
    ax.imshow(canvas, interpolation='none')

    if show_kpts:
        for k, d, dy in ((kpts0, details['d0'], 0), (kpts1, details['d1'], off)):
            cov = np.isfinite(d)
            rep = cov & (d <= LOC_TH)
            for sel, color in ((~cov, 'white'), (cov & ~rep, (1.0, 0.55, 0.0)), (rep, (0.0, 0.85, 1.0))):
                ax.scatter(k[sel, 0], k[sel, 1] + dy, s=5, color=color, edgecolors='black', lw=0.2, zorder=2)

    m, err = details['matches'], details['err']
    p0, p1 = kpts0[m[:, 0]], kpts1[m[:, 1]]
    layers = (([(~np.isfinite(err), GREY, 0.35)] if show_unverifiable else [])
              + ([(err > err_th, RED, 0.5)] if show_wrong else []) + [(err <= err_th, GREEN, 0.8)])
    for sel, color, alpha in layers:
        straight = sel & (np.abs(p0[:, 0] - p1[:, 0]) < W / 2)  # across the azimuth seam: end points only
        ax.plot(np.stack([p0[straight, 0], p1[straight, 0]]), np.stack([p0[straight, 1], p1[straight, 1] + off]),
                color=color, lw=0.6, alpha=alpha)
        ax.scatter(np.r_[p0[sel, 0], p1[sel, 0]], np.r_[p0[sel, 1], p1[sel, 1] + off], s=3, color=color, zorder=3)
    ax.text(0.004, 0.985, banner, transform=ax.transAxes, ha='left', va='top', fontsize=6.5, color='white',
            family='monospace', bbox=dict(facecolor='#333333', alpha=0.85, edgecolor='none', pad=1.5))
    ax.text(0.995, 0.97, label, transform=ax.transAxes, ha='right', va='top', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='black', lw=0.8))
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(2 * H + gap - 0.5, -0.5)
    ax.set_axis_off()


def pair_figure(img0, img1, results, suptitle=''):
    """One block per model: detections in A, detections in B, matches.

    results: list of (name, kpts0, kpts1, details, metrics)
    """
    n = len(results)
    fig, axes = plt.subplots(3 * n, 1, figsize=(16, 7.6 * n),
                             gridspec_kw={'height_ratios': [1, 1, 2.3] * n}, constrained_layout=True)
    for i, (name, kpts0, kpts1, details, m) in enumerate(results):
        tag = f'[{name}] rep@3 {m["rep@3"]:.2f}, loc_err {m["loc_err"]:.2f} px, MMA@3 {m["MMA@3"]:.2f}'
        draw_detections(axes[3 * i], img0, kpts0, details['d0'], f'{tag}\nA')
        draw_detections(axes[3 * i + 1], img1, kpts1, details['d1'], 'B')
        draw_matches(axes[3 * i + 2], img0, img1, kpts0, kpts1, details)
    if suptitle:
        fig.suptitle(suptitle, fontsize=10)
    return fig


def figure_to_array(fig):
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return img
