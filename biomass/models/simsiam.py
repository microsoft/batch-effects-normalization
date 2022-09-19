from typing import Tuple

import torch
from torch import nn
from torch.nn import Module
from torch import Tensor


"""Adapted from https://github.com/facebookresearch/simsiam/blob/main/simsiam/builder.py"""


class SimSiam(Module):
    def __init__(self, encoder: Module, dim: int = 2048, pred_dim: int = 512):
        super().__init__()
        self.encoder = encoder

        # build a 3-layer projector
        self.projector = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),  # first layer
            nn.Linear(dim, dim, bias=False),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),  # second layer
            nn.Linear(dim, dim, bias=False),
            nn.BatchNorm1d(dim, affine=False),
        )  # output layer

        # build a 2-layer predictor
        self.predictor = nn.Sequential(
            nn.Linear(dim, pred_dim, bias=False),
            nn.BatchNorm1d(pred_dim),
            nn.ReLU(inplace=True),  # hidden layer
            nn.Linear(pred_dim, dim),
        )  # output layer

    def encode(self, data: Tensor) -> Tensor:
        return self.encoder(data)

    def forward(self, data: Tensor) -> Tuple[Tensor, Tensor]:
        z = self.projector(self.encoder(data))
        p = self.predictor(z)
        return p, z.detach()
