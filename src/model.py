import torch.nn as nn


def InputBlock(in_channels=3, out_channels=64):
  # convolution /2
  yield nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3)
  yield nn.ReLU(inplace=True)
  # max pooling /2
  yield nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


def ConvBlock(in_channels, out_channels=None, stride=1):
  if out_channels is None:
    out_channels = in_channels
  # first convolution
  yield nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
  yield nn.ReLU(inplace=True)
  # second convolution
  yield nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
  yield nn.ReLU(inplace=True)


def ConvBlock2(in_channels, mid_channels, out_channels=None, stride=1):
  if out_channels is None:
    out_channels = in_channels
  # first convolution
  yield nn.Conv2d(in_channels, mid_channels, kernel_size=1)
  yield nn.ReLU(inplace=True)
  # second convolution
  yield nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1)
  yield nn.ReLU(inplace=True)
  # third convolution
  yield nn.Conv2d(mid_channels, out_channels, kernel_size=1)
  yield nn.ReLU(inplace=True)


def OutputBlock(in_features, out_features):
  # gap
  yield nn.AdaptiveAvgPool2d((1, 1))
  # fc
  yield nn.Flatten()
  yield nn.Linear(in_features, out_features)


class F34(nn.Sequential):
  def __init__(self, out_features=1000):
    super().__init__(
        # input
        *InputBlock(3, 64),
        # ch=64
        *ConvBlock(64),
        *ConvBlock(64),
        *ConvBlock(64),
        # ch=128
        *ConvBlock(64, 128, 2),
        *ConvBlock(128),
        *ConvBlock(128),
        *ConvBlock(128),
        # ch=256
        *ConvBlock(128, 256, 2),
        *ConvBlock(256),
        *ConvBlock(256),
        *ConvBlock(256),
        *ConvBlock(256),
        *ConvBlock(256),
        # ch=512
        *ConvBlock(256, 512, 2),
        *ConvBlock(512),
        *ConvBlock(512),
        # output
        *OutputBlock(1 * 1 * 512, out_features)
    )


class F50(nn.Sequential):
  def __init__(self, out_features=1000):
    super().__init__(
        # input
        *InputBlock(3, 64),
        # ch=256
        *ConvBlock2(64, 64, 256),
        *ConvBlock2(256, 64),
        *ConvBlock2(256, 64),
        # ch=512
        *ConvBlock2(256, 128, 512, 2),
        *ConvBlock2(512, 128),
        *ConvBlock2(512, 128),
        *ConvBlock2(512, 128),
        # ch=1024
        *ConvBlock2(512, 256, 1024, 2),
        *ConvBlock2(1024, 256),
        *ConvBlock2(1024, 256),
        *ConvBlock2(1024, 256),
        *ConvBlock2(1024, 256),
        *ConvBlock2(1024, 256),
        # ch=2048
        *ConvBlock2(1024, 512, 2048, 2),
        *ConvBlock2(2048, 512),
        *ConvBlock2(2048, 512),
        # output
        *OutputBlock(1 * 1 * 2048, out_features)
    )
