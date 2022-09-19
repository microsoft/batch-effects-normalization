import torch
from torch.nn import Module, Sequential, Linear, ReLU
from torch import Tensor


class ContrastiveLearner(Module):
    def __init__(
        self,
        encoder: Module,
        classifier: Module,
        proj_nonlinear: bool,
        proj_in: int,
        proj_out: int,
    ):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier
        if proj_nonlinear:
            self.projection = Sequential(
                Linear(proj_in, proj_in), ReLU(inplace=True), Linear(proj_in, proj_out)
            )
        else:
            self.projection = Linear(proj_in, proj_out)

    def encode(self, data: Tensor) -> Tensor:
        return self.encoder(data)

    def project(self, data: Tensor) -> Tensor:
        features = self.encode(data)
        return self.projection(features)

    def classify(self, data: Tensor) -> Tensor:
        features = self.encode(data)
        return self.classifier(features)

    def freeze_encoder(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = True

    def forward(self, data: Tensor) -> Tensor:
        return self.classify(data)

