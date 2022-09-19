import torch
from torch import Tensor
import torchvision.transforms.functional as TF


class RandomRotate(object):
    def __init__(self):
        self.angles = [0, 90, 180, 270]

    def __call__(self, x: Tensor) -> Tensor:
        angle = self.angles[torch.randint(low=0, high=len(self.angles), size=(1,))]
        if angle > 0:
            x = TF.rotate(x, angle)
        return x
