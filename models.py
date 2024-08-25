import torch
import torch.nn as nn

from constants import ACTION_SIZE


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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.channel = 256
        self.feature = nn.Sequential(
            *[ResidualBlock(3 if idx == 0 else self.channel, self.channel) for idx in range(5)])

        self.value = nn.Linear(2 * 49, 1)

        self.tanh = nn.Tanh()
        self.log_softmax = nn.LogSoftmax(dim=1)

        # policy head
        self.p_conv = nn.Conv2d(self.channel, 4, kernel_size=1, padding=0, bias=False)
        self.p_bn = nn.BatchNorm2d(num_features=4)
        self.relu = nn.ReLU(inplace=True)
        self.probability = nn.Linear(4 * 49, ACTION_SIZE)

        # value head
        self.v_conv = nn.Conv2d(self.channel, 2, kernel_size=1, padding=0, bias=False)
        self.v_bn = nn.BatchNorm2d(num_features=2)

    def forward(self, state):
        assert len(state.shape) == 4
        batch, _, _, _ = state.shape
        state = self.feature(state)

        p = self.p_conv(state)
        p = self.p_bn(p)
        p = self.relu(p).view(batch, -1)
        p = self.probability(p)
        p = self.log_softmax(p)

        v = self.v_conv(state).view(batch, -1)
        v = self.value(v)
        v = self.tanh(v)

        return v, p


if __name__ == '__main__':
    state_ = torch.randn(2, 1, 7, 7)
    m = ChessNet()
    v, p = m(state_)
    print(v.shape)
    print(p.shape)
