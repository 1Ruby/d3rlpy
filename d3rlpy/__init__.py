import random

import numpy as np
import torch

from . import datasets
from ._version import __version__
from .algos import *
from .base import load_learnable
from .dataset import *
from .envs import *
from .metrics import *
from .models import *
from .ope import *
from .preprocessing import *
from .wrappers import *


def seed(n: int) -> None:
    """Sets random seed value.

    Args:
        n (int): seed value.

    """
    random.seed(n)
    np.random.seed(n)
    torch.manual_seed(n)
    torch.cuda.manual_seed(n)
    torch.backends.cudnn.deterministic = True
