import torch
from torch import Tensor
from torch.nn import Module, Conv2d, BatchNorm2d, MaxPool2d, ReLU, Sequential, Upsample


class PairedCellInpainter(Module):
    def __init__(self):

        super().__init__()

        # Encoding layers
        self.conv1 = Sequential(
            Conv2d(2, 96, (3, 3), padding="same"),
            ReLU(),
            BatchNorm2d(96),
            MaxPool2d((2, 2)),
        )
        self.conv2 = Sequential(
            Conv2d(96, 256, (3, 3), padding="same"),
            ReLU(),
            BatchNorm2d(256),
            MaxPool2d((2, 2)),
        )
        self.conv3 = Sequential(
            Conv2d(256, 384, (3, 3), padding="same"), ReLU(), BatchNorm2d(384),
        )
        self.conv4 = Sequential(
            Conv2d(384, 384, (3, 3), padding="same"), ReLU(), BatchNorm2d(384),
        )
        self.conv5 = Sequential(
            Conv2d(384, 256, (3, 3), padding="same"), ReLU(), BatchNorm2d(256),
        )

        self.rfp_conv1 = Sequential(
            Conv2d(1, 16, (3, 3), padding="same"),
            ReLU(),
            BatchNorm2d(16),
            MaxPool2d((2, 2)),
        )
        self.rfp_conv2 = Sequential(
            Conv2d(16, 32, (3, 3), padding="same"),
            ReLU(),
            BatchNorm2d(32),
            MaxPool2d((2, 2)),
        )
        self.rfp_conv3 = Sequential(
            Conv2d(32, 32, (3, 3), padding="same"), ReLU(), BatchNorm2d(32),
        )

        # Decoding layers
        self.conv6 = Sequential(Conv2d(288, 256, (3, 3), padding="same"), ReLU())
        self.conv7 = Sequential(Conv2d(256, 384, (3, 3), padding="same"), ReLU())
        self.conv8 = Sequential(Conv2d(384, 384, (3, 3), padding="same"), ReLU())
        self.conv9 = Sequential(
            Upsample(scale_factor=(2, 2)),
            Conv2d(384, 256, (3, 3), padding="same"),
            ReLU(),
        )
        self.conv10 = Sequential(
            Upsample(scale_factor=(2, 2)),
            Conv2d(256, 96, (3, 3), padding="same"),
            ReLU(),
            Conv2d(96, 1, (1, 1)),
        )

    def encode(self, nucleus: Tensor, protein: Tensor) -> Tensor:
        data = torch.stack([nucleus, protein], axis=-3)
        encoding = self.conv1(data)
        encoding = self.conv2(encoding)
        encoding = self.conv3(encoding)
        encoding = self.conv4(encoding)
        encoding = self.conv5(encoding)
        return encoding

    def encode_nucleus(self, nucleus: Tensor) -> Tensor:
        encoding = self.rfp_conv3(
            self.rfp_conv2(self.rfp_conv1(nucleus.unsqueeze(dim=-3)))
        )
        return encoding

    def decode(self, encoding1: Tensor, encoding2: Tensor) -> Tensor:
        encoding = torch.cat([encoding1, encoding2], axis=-3)
        decoding = self.conv10(self.conv9(self.conv8(self.conv7(self.conv6(encoding)))))
        return decoding

    def forward(self, nucleus1: Tensor, protein1: Tensor, nucleus2: Tensor) -> Tensor:
        return self.decode(
            encoding1=self.encode(nucleus1, protein1),
            encoding2=self.encode_nucleus(nucleus2),
        ).squeeze(dim=-3)
