import torch
import torch.nn as nn
from torch.nn import init


def grad_hook(grad):
    # 计算梯度范数（L2范数）
    grad_norm = grad.norm().item()
    # 统计非零梯度比例
    non_zero_ratio = torch.count_nonzero(grad).item() / grad.numel()
    print(f"梯度范数: {grad_norm:.6f} | 非零比例: {non_zero_ratio:.2%}")


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


class GameNet(nn.Module):
    def __init__(self, input_channel, input_size, action_size):
        super().__init__()
        self.action_size = action_size
        self.input_size = input_size
        self.channel = 256
        self.cnn_layer_num = 2
        self.feature = nn.Sequential(
            *([ResidualBlock(input_channel, self.channel)] + [ResidualBlock(self.channel, self.channel) for i in
                                                              range(self.cnn_layer_num - 1)]))

        self.value = nn.Linear(self.channel * input_size * input_size, 1)
        self.probability = nn.Linear(self.channel * input_size * input_size, action_size)
        self.tanh = nn.Tanh()
        self.log_softmax = nn.LogSoftmax(dim=1)

        # 为每个参数注册钩子
        for name, param in self.named_parameters():
            if param.requires_grad:
                if "conv" in name:
                    init.kaiming_normal_(param.data)
                if "bias" in name:
                    init.zeros_(param.data)
                # param.register_hook(grad_hook)

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
