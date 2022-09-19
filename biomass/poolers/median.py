from torch.nn import Module
import torch


class MedianPooler(Module):
    def __init__(self):
        super().__init__()

    def forward(self, xs):
        return torch.median(xs, dim=0).values
