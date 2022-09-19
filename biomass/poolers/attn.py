from typing import Tuple
from torch.nn import Module, Linear, Sequential, Sigmoid
import torch


class AttentionPooler(Module):
    def __init__(self, dims: Tuple[int]):
        super().__init__()
        self.dims = dims
        modules = []
        for i in range(len(dims) - 1):
            modules.append(Linear(dims[i], dims[i + 1]))
            modules.append(Sigmoid())
        modules.append(Linear(dims[-1], 1))
        self.seq = Sequential(*modules)

    def forward(self, xs):
        scores = torch.softmax(self.seq(xs), dim=0)
        return torch.sum(scores * xs, dim=0)
