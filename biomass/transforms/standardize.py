from typing import Union, Tuple
from torch import Tensor
import torchvision.transforms.functional as TF


class Standardize(object):
    def __init__(self, dim: Union[int, Tuple[int, ...]]):
        self.dim = tuple(dim)

    def __call__(self, x: Tensor) -> Tensor:
        mean = x.mean(dim=self.dim)
        std = x.std(dim=self.dim)
        std[std == 0.0] = 1.0
        return TF.normalize(x, mean, std)

