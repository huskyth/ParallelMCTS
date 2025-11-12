import torch.nn as nn


def conv3x3(in_channels, out_channels, stride=1):
    # 3x3 convolution
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)


class ResidualBlock(nn.Module):
    # Residual block
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = False
        if in_channels != out_channels or stride != 1:
            self.downsample = True
            self.downsample_conv = conv3x3(in_channels, out_channels, stride=stride)
            self.downsample_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample:
            residual = self.downsample_conv(residual)
            residual = self.downsample_bn(residual)

        out += residual
        out = self.relu(out)
        return out


class ChessNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.channel = 256
        self.feature = nn.Sequential(
            *[ResidualBlock(1 if idx == 0 else self.channel, self.channel) for idx in range(5)])

        self.value = nn.Linear(self.channel * 49, 1)
        self.probability = nn.Linear(self.channel * 49, 72)
        self.tanh = nn.Tanh()
        self.log_softmax = nn.LogSoftmax()

    def forward(self, state):
        assert len(state.shape) == 4
        batch, _, _, _ = state.shape
        state = self.feature(state)
        state = state.view(batch, -1)
        v = self.value(state)
        v = self.tanh(v)
        p = self.probability(state)
        p = self.log_softmax(p)
        return v, p
