from torch.nn import Module
import torch


class MeanPooler(Module):
    def __init__(self):
        super().__init__()

    def forward(self, xs):
        return torch.mean(xs, dim=0)
