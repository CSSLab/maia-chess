import torch
import torch.nn
import torch.nn.functional

import collections

class ScalarToChannel(torch.nn.Module):
    def __init__(self, multiplier = 1.0):
        super().__init__()
        self.expander = torch.nn.Linear(1, 8*8, bias = False)
        self.expander.weight.requires_grad = False
        self.expander.weight.data.fill_(multiplier)

    def forward(self, x):
        return self.expander(x.unsqueeze(1)).reshape(-1, 1, 8, 8)

class Flatten(torch.nn.Module):
    #https://stackoverflow.com/a/56771143
    def forward(self, input_x):
        return input_x.view(input_x.size(0), -1)

class No_op(torch.nn.Module):
    def forward(self, input_x):
        return input_x


class CenteredBatchNorm2d(torch.nn.BatchNorm2d):
    """Only apply bias, no scale like:
        tf.layers.batch_normalization(
            center=True, scale=False,
            )
    """

    def __init__(self, channels):
        super().__init__(channels, affine = True, eps=1e-5)
        #self.weight = 1 by default
        self.weight.requires_grad = False

class ConvBlock(torch.nn.Module):
    def __init__(self, filter_size, input_channels, output_channels):
        super().__init__()
        layers = [
            ('conv2d', torch.nn.Conv2d(
                        input_channels,
                        output_channels,
                        filter_size,
                        stride = 1,
                        padding = filter_size // 2,
                        bias = False,
                        )),
            ('norm2d', CenteredBatchNorm2d(output_channels)),
            ('ReLU', torch.nn.ReLU()),
            ]
        self.seq = torch.nn.Sequential(collections.OrderedDict(layers))

    def forward(self, x):
        return self.seq(x)

class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()

        layers = [
            ('conv2d_1', torch.nn.Conv2d(
                        channels,
                        channels,
                        3,
                        stride = 1,
                        padding = 1,
                        bias = False,
                        )),
            ('norm2d_1', CenteredBatchNorm2d(channels)),
            ('ReLU', torch.nn.ReLU()),
            ('conv2d_2', torch.nn.Conv2d(
                        channels,
                        channels,
                        3,
                        stride = 1,
                        padding = 1,
                        bias = False,
                        )),
            ('norm2d_2', CenteredBatchNorm2d(channels)),
            ]
        self.seq = torch.nn.Sequential(collections.OrderedDict(layers))

    def forward(self, x):
        y = self.seq(x)
        y += x
        y = torch.nn.functional.relu(y, inplace = True)
        return y
