import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Sets the seeds to guarantee reproducibility in the
    random, numpy, and torch modules. Additionally, adjusts
    cuDNN settings to ensure deterministic results.

    Args:
        seed: Seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False