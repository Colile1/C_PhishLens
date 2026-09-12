# seeding.py
# Purpose: seed every source of randomness from one value.
# Author: Colile Sibanda

from __future__ import annotations

import random

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """Seed random, numpy and torch (CPU and CUDA) and set deterministic flags.

    Inputs: seed, the single integer driving every source of randomness.
    Output: None. Mutates global RNG state in random, numpy and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_rng(seed: int) -> np.random.Generator:
    """Build a numpy random Generator for code that should not rely on global state.

    Inputs: seed, the integer seed.
    Output: a numpy.random.Generator seeded with the given value.
    """
    return np.random.default_rng(seed)
