from torch import Tensor
import numpy as np
import torch


class Quartering(object):
    def __init__(self, p: float):
        self.p = p

    def __call__(self, x: Tensor) -> Tensor:
        W, H = x.shape[-2:]
        if np.random.random() < self.p:
            quarters = [
                x[..., : W // 2, : H // 2],
                x[..., : W // 2, H // 2 :],
                x[..., W // 2 :, : H // 2],
                x[..., W // 2 :, H // 2 :],
            ]
            np.random.shuffle(quarters)
            x = torch.cat(
                [
                    torch.cat([quarters[0], quarters[1]], axis=-1),
                    torch.cat([quarters[2], quarters[3]], axis=-1),
                ],
                axis=-2,
            )
        return x
