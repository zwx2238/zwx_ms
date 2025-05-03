import random

import mindspore as ms
import numpy as np
import torch


def set_random_seed(seed: int = 42):
    """设置所有随机种子以确保可重现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    ms.set_seed(seed)
