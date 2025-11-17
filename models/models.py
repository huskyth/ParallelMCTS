import torch
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
        self.feature = nn.Sequential(ResidualBlock(2, self.channel))

        self.value = nn.Linear(self.channel * 49, 1)
        self.probability = nn.Linear(self.channel * 49, 72)
        self.tanh = nn.Tanh()
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, state):
        for _ in range(4 - len(state.shape)):
            state = state.unsqueeze(0)

        if len(state.shape) != 4:
            raise Exception(f'state must be 4 dim, but {len(state.shape)} dim')

        batch, _, _, end = state.shape

        if end == 2:
            state = torch.permute(state, (0, 3, 1, 2))

        state = self.feature(state)
        state = state.reshape(batch, -1)

        v = self.value(state)
        v = self.tanh(v)
        p = self.probability(state)
        p = self.log_softmax(p)

        return v, p


if __name__ == '__main__':
    # md = ChessNet().cuda()
    # tens = torch.randn(1, 7, 7, 2)
    # tens = tens.cuda()
    #
    # y = md(tens)
    # print(y[1].shape, (y[1] / y[1].sum(keepdims=True,dim=1))[0])
    # import torch.nn.functional as F
    #
    input = torch.randn(3, 5, requires_grad=True)
    target = torch.randint(5, (3,), dtype=torch.int64)
    print(input.shape, target.shape)
    # loss = F.cross_entropy(input, target)
    # print(loss)

    p = torch.randn(5, 72)
    print(p.shape)

    print(torch.argmax(p, dim=1).shape == (5, ))
