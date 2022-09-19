import numpy as np
from torch import Tensor
import torch


class ToLongTensor(object):
    def __init__(self):
        pass

    def __call__(self, x: np.array) -> Tensor:
        return torch.LongTensor(np.array(x)).permute((2, 0, 1))
