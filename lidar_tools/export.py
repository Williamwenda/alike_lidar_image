"""Export the network weights of a Lightning checkpoint to a .pth that the inference ALike loads.

    uv run python -m lidar_tools.export <checkpoint.ckpt> [alike-n] [out.pth]
"""
import sys
from pathlib import Path

import torch

from lidar_tools import REPO_ROOT  # noqa: F401  (puts the inference ALIKE on the path)
from alike import ALike, configs


def ckpt_to_pth(ckpt_path, name='alike-n', out_path=None):
    """Keeps the weights of the inference network (strict load check) and saves them as out_path."""
    state = torch.load(ckpt_path, map_location='cpu', weights_only=False)['state_dict']
    cfg = {k: v for k, v in configs[name].items() if k != 'model_path'}
    net = ALike(**cfg, device='cpu')
    net.load_state_dict({k: v for k, v in state.items() if k in net.state_dict()}, strict=True)
    out_path = Path(out_path) if out_path else Path(ckpt_path).with_suffix('.pth')
    torch.save(net.state_dict(), out_path)
    return out_path


if __name__ == '__main__':
    print(ckpt_to_pth(sys.argv[1], *sys.argv[2:]))
