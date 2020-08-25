import torch
import torch.nn
import torch.nn.functional as F

class CenteredBatchNorm2d(torch.nn.BatchNorm2d):
    """It appears the only way to get a trainable model with beta (bias) but not scale (weight
     is by keeping the weight data, even though it's not used"""

    def __init__(self, channels):
        super().__init__(channels, affine=True)
        self.weight.data.fill_(1)
        self.weight.requires_grad = False

class ConvBlock(torch.nn.Module):
    def __init__(self, kernel_size, input_channels, output_channels=None):
        super().__init__()
        if output_channels is None:
            output_channels = input_channels
        padding = kernel_size // 2
        self.conv1 = torch.nn.Conv2d(input_channels, output_channels, kernel_size, stride=1, padding=padding, bias=False)
        self.conv1_bn = CenteredBatchNorm2d(output_channels)

    def forward(self, x):
        out = self.conv1_bn(self.conv1(x))
        out = F.relu(out, inplace=True)
        return out

class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(channels, channels, 3, stride=1, padding=1, bias=False)
        self.conv1_bn = CenteredBatchNorm2d(channels)
        self.conv2 = torch.nn.Conv2d(channels, channels, 3, stride=1, padding=1, bias=False)
        self.conv2_bn = CenteredBatchNorm2d(channels)

    def forward(self, x):
        out = self.conv1_bn(self.conv1(x))
        out = F.relu(out, inplace=True)
        out = self.conv2_bn(self.conv2(out))
        out += x
        out = F.relu(out, inplace=True)
        return out
