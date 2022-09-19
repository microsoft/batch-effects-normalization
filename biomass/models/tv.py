from typing import Optional

from torch import Tensor
from torch.nn import Module, Identity, Linear
import torchvision


class TorchVisionClassifier(Module):
    def __init__(self, key: str, classifier: str, d_out: Optional[int], **kwargs):
        super().__init__()
        # construct the default model, which has the default last layer
        constructor = getattr(torchvision.models, key)
        self.model = constructor(**kwargs)
        # adjust the last layer
        d_features = getattr(self.model, classifier).in_features
        if d_out is None:  # want to initialize a featurizer model
            last_layer = Identity(d_features)
            self.d_out = d_features
        else:  # want to initialize a classifier for a particular num_classes
            last_layer = Linear(d_features, d_out)
            self.d_out = d_out
        setattr(self.model, classifier, last_layer)

    def forward(self, x: Tensor):
        return self.model(x)


class TorchVisionEncoder(Module):
    def __init__(self, name: str, classifier_key: str, **kwargs):
        super().__init__()
        # construct the default model, which has the default last layer
        constructor = getattr(torchvision.models, name)
        self.tv_encoder = constructor(**kwargs)
        # replace last layer classifier with identity
        last_layer = Identity()
        setattr(self.tv_encoder, classifier_key, last_layer)

    def forward(self, x: Tensor):
        return self.tv_encoder(x)
