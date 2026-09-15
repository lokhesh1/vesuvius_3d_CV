import os
import random
import numpy as np
import torch


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_positions(dim_size, patch_size, stride):
    positions = list(range(0, max(1, dim_size - patch_size + 1), stride))
    last = dim_size - patch_size
    if last > 0 and positions[-1] != last:
        positions.append(last)
    return positions
