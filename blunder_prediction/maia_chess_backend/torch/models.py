import torch
import torch.nn
import torch.nn.functional as F

import numpy as np

from .blocks import ResidualBlock, ConvBlock
from ..utils import printWithDate

from .resnet import ResNet, BasicBlock, Bottleneck

num_planes = 17

def norm_extras(name, vals_arr):
    if name == 'active_elo':
        return vals_arr / 4000
    else:
        return vals_arr

def NetFromConfig(config):
    printWithDate(f"loading modele:{config}")
    if config['type'] == 'fully_connected':
        return FullyConnectedLower(
                        config['hidden_size'],
                        config.get('num_layers', 2),
                        dropout = config.get('dropout'),
                        activation_func = config.get('activation_func'),
                        batch_norm = config.get('batch_norm', None),
                        extra_inputs = config['extra_inputs'],
                        output_reg = config['ouputs_reg'],
                        output_cls = config['ouputs_cls'],
                        )
    elif config['type'] == 'leela':
        return LeelaLower(
                        config['channels'],
                        config['blocks'],
                        extra_inputs = config['extra_inputs'],
                        output_reg = config['ouputs_reg'],
                        output_cls = config['ouputs_cls'],
                        )
    elif config['type'] == 'CNN':
        return SimpleCNNLower(
                        config['channels'],
                        config['kernel_size'],
                        extra_inputs = config['extra_inputs'],
                        output_reg = config['ouputs_reg'],
                        output_cls = config['ouputs_cls'],
                        )
    elif config['type'] == 'fully_connected_simple':
        return FullyConnectedSimple(
                        config['hidden_size'],
                        output_cls = config['ouputs_cls'],
                        )
    elif config['type'] == 'resnet':
        return ResNet(Bottleneck, [3, 4, 6, 3], num_classes = 2)
    else:
        raise RuntimeError(f"{config['type']} is not a valid network type")

class FullyConnectedSimple(torch.nn.Module):
    def __init__(self, hidden_size, output_cls):
        super().__init__()
        self.hidden_size = hidden_size

        self.output_cls = list(output_cls)

        self.fc1 = torch.nn.Linear(num_planes*8*8, self.hidden_size)
        self.fc2 = torch.nn.Linear(self.hidden_size, self.hidden_size // 2)
        self.fc3 = torch.nn.Linear(self.hidden_size // 2, 2)

        self.act = torch.nn.Tanh()

    def forward(self, input_val):
        out_val = input_val.view(-1, num_planes*8*8)
        out_val = self.act(self.fc1(out_val))
        out_val = self.act(self.fc2(out_val))
        out_val = self.fc3(out_val).softmax(1)
        return out_val

    def save(self, fname):
        with open(fname, 'wb') as f:
            torch.save(self, f)

    def train_batch(self, input_x, real_y, optimizer, loss_reg = None, loss_cls = None):
        optimizer.zero_grad()
        pred_y = self(input_x)
        y_val = real_y[self.output_cls[0]]
        loss_n = loss_cls(pred_y, y_val)
        loss_n.backward()
        optimizer.step()
        return {self.output_cls[0] : float(loss_n)}

    def test_batch(self, input_x, real_y, loss_reg = None, loss_cls = None):
        n = self.output_cls[0]
        pred_y = self(input_x)
        y_val = real_y[n]
        loss_n = loss_cls(pred_y, y_val)
        losses = {}
        errors = {}
        losses[n] = float(loss_n)

        errors[n] = 1 - float((pred_y.argmax(1) == real_y[n]).sum()) / real_y[n].shape[0]
        return losses, errors

class BaseNet(torch.nn.Module):
    """This lets us specify the outputs of the net at creation time. All the different nets just need to define their core structure and it does the rest.

    The dynamic inputs stuff isn't finished, but it should be just a matter of adding stuff to the final layer.

    We're also only considering regression and classification as the two types of output to make things simpler.
    """
    #@profile
    def __init__(self, final_layer_size, extra_inputs, output_reg, output_cls, hidden_size = 128):
        super().__init__()

        if output_reg is None:
            output_reg = []
        if output_cls is None:
            output_cls = []
        if extra_inputs is None:
            extra_inputs = []
            self.has_extra_inputs = False
        else:
            self.has_extra_inputs = True

        self.final_layer_size = final_layer_size
        self.hidden_size = hidden_size

        self.output_reg = list(output_reg)
        self.output_cls = list(output_cls)
        self.extra_inputs = list(extra_inputs)

        self.reg_layers = {}
        self.cls_layers = {}
        self._on_cuda = None

        if self.has_extra_inputs:
            self.extras_fc = torch.nn.Linear(len(extra_inputs), 8 * len(extra_inputs))

            final_layer_size = final_layer_size + 8 * len(extra_inputs)

        for n in self.output_reg:
            self.reg_layers[n] = (torch.nn.Linear(final_layer_size, hidden_size), torch.nn.Linear(hidden_size, 1))
            #A bit of a hack to add theme to the parameters list
            setattr(self, f"{n}_reg_lower", self.reg_layers[n][0])
            setattr(self, f"{n}_reg_upper", self.reg_layers[n][1])

        for n in self.output_cls:
            if n in ['move', 'top_nonblunder', 'top_blunder']:
                self.cls_layers[n] = (torch.nn.Linear(final_layer_size, hidden_size), torch.nn.Linear(hidden_size, 2**12))
            else:
                self.cls_layers[n] = (torch.nn.Linear(final_layer_size, hidden_size), torch.nn.Linear(hidden_size, 2))
            #A bit of a hack to add theme to the parameters list
            setattr(self, f"{n}_cls_lower", self.cls_layers[n][0])
            setattr(self, f"{n}_cls_upper", self.cls_layers[n][1])

    @property
    def on_cuda(self):
        if self._on_cuda is None:
            self._on_cuda = bool(next(self.parameters()).is_cuda)
        return self._on_cuda

    def forward_lower(self, input_val):
        raise NotImplementedError
    #@profile
    def forward(self, input_x, extra_ins = None):
        if isinstance(input_x, np.ndarray):
            input_x = torch.from_numpy(input_x.astype(np.float32))
        if self.on_cuda:
            input_x = input_x.cuda()

        y_lower = self.forward_lower(input_x).view(input_x.shape[0], -1)

        if extra_ins is not None:
            if isinstance(extra_ins, np.ndarray):
                extra_ins = torch.from_numpy(extra_ins.astype(np.float32))
            if self.on_cuda:
                extra_ins = extra_ins.cuda()

            extras_y = self.extras_fc(extra_ins).tanh()
            y_lower = torch.cat([y_lower, extras_y], axis = 1)

        y_vals = {}

        for n in self.output_reg:
            fc_l, fc_o = self.reg_layers[n]
            y_working = fc_l(y_lower).tanh()
            y_vals[n] = fc_o(y_working)

        for n in self.output_cls:
            fc_l, fc_o = self.cls_layers[n]
            y_working = fc_l(y_lower).tanh()
            y_vals[n] = fc_o(y_working).softmax(1)

        return y_vals

    def save(self, fname):
        with open(fname, 'wb') as f:
            torch.save(self, f)
    #@profile
    def train_batch(self, input_x, real_y, optimizer, loss_reg = None, loss_cls = None):
        optimizer.zero_grad()
        self.train(mode = True)
        #import pdb; pdb.set_trace()
        if self.has_extra_inputs:
            inputs = torch.cat([norm_extras(n, real_y[n]) for n in self.extra_inputs], axis = 1)
            pred_y = self(input_x, extra_ins = inputs)
        else:
            pred_y = self(input_x)
        losses = {}

        for n in self.output_reg:
            loss_n = loss_reg(pred_y[n], real_y[n].view(-1, 1))
            loss_n.backward(retain_graph=True)
            losses[n] = loss_n.detach()#.detach().cpu()

        for n in self.output_cls:
            loss_n = loss_cls(pred_y[n], real_y[n])
            loss_n.backward(retain_graph=True)
            losses[n] = loss_n.detach()#.detach().cpu()

        self.train(mode = False)
        optimizer.step()

        return losses
    #@profile
    def test_batch(self, input_x, real_y, loss_reg = None, loss_cls = None, extra_cls_loss = None):
        if self.has_extra_inputs:
            inputs_extra = torch.cat([real_y[n] for n in self.extra_inputs], axis = 1)
            pred_y = self(input_x, extra_ins = inputs_extra)
        else:
            pred_y = self(input_x)
        losses = {}
        errors = {}

        for n in self.output_reg:
            loss_n = loss_reg(pred_y[n], real_y[n].view(-1, 1))
            losses[n] = loss_n.detach()

        for n in self.output_cls:
            loss_n = loss_cls(pred_y[n], real_y[n])
            losses[n] = loss_n.detach()
            errors[n] = (pred_y[n].argmax(1) != real_y[n]).float().mean().item()
        if extra_cls_loss is not None:
            for n in self.output_cls:
                loss_n = extra_cls_loss(pred_y[n], real_y[n])
                losses[n+'_nll'] = loss_n.detach()
        return losses, errors

class LeelaLower(BaseNet):
    def __init__(self, channels, blocks, extra_inputs = None, output_reg = None, output_cls = None):

        super().__init__(channels*8*8, extra_inputs, output_reg, output_cls)
        self.conv_in = ConvBlock(kernel_size=3,
                               input_channels=num_planes,
                               output_channels=channels)
        self.residual_blocks = []
        for idx in range(blocks):
            block = ResidualBlock(channels)
            self.residual_blocks.append(block)
            self.add_module('residual_block{}'.format(idx+1), block)
        self.conv_out = ConvBlock(kernel_size=1,
                                   input_channels=channels,
                                   output_channels=channels)
    #@profile
    def forward_lower(self, lower_x):
        out = self.conv_in(lower_x)
        for block in self.residual_blocks:
            out = block(out)
        return self.conv_out(out)

class FullyConnectedLower(BaseNet):
    def __init__(self, h_size, num_layers, dropout = None, activation_func = 'tanh', batch_norm = None, extra_inputs = None, output_reg = None, output_cls = None):

        super().__init__(h_size, extra_inputs, output_reg, output_cls)

        self.num_layers = num_layers
        if dropout is None:
            dropout = 0.0
        self.dropout = torch.nn.Dropout(dropout)

        self.batch_norm = batch_norm

        self.fc1 = torch.nn.Linear(num_planes*8*8, h_size)
        for i in range(num_layers - 1):
            setattr(self, f"fc{i + 2}", torch.nn.Linear(h_size, h_size))

        if self.batch_norm is not None:
            self.fc1_norm = torch.nn.BatchNorm1d(h_size)
            for i in range(num_layers - 1):
                setattr(self, f"fc1_norm{i + 2}", torch.nn.BatchNorm1d(h_size))

        if activation_func == 'tanh':
            self.act = torch.nn.Tanh()
        elif activation_func == 'sigmoid':
            self.act = torch.nn.Sigmoid()
        elif activation_func == 'relu':
            self.act = torch.nn.ReLU()
        else:
            raise RuntimeError(f"{activation_func} is not a known activation func")
    #@profile
    def forward_lower(self, lower_x):
        out_val = lower_x.view(-1, num_planes*8*8)
        if self.batch_norm is not None:
            out_val = self.act(self.dropout(self.fc1(out_val)))
            out_val = self.fc1_norm(out_val)
            for i in range(self.num_layers - 1):
                out_val = self.act(self.dropout(getattr(self, f"fc{i + 2}")(out_val)))
                out_val = getattr(self, f"fc1_norm{i + 2}")(out_val)
        else:
            out_val = self.act(self.dropout(self.fc1(out_val)))
            for i in range(self.num_layers - 1):
                out_val = self.act(self.dropout(getattr(self, f"fc{i + 2}")(out_val)))
        return out_val

class AttentionLower(BaseNet):
    def __init__(self, h_size, extra_inputs = None, output_reg = None, output_cls = None):

        super().__init__(h_size, extra_inputs, output_reg, output_cls)

        self.fc1 = torch.nn.Linear(num_planes*8*8, h_size)
        self.fc2 = torch.nn.Linear(h_size, h_size)

    def forward_lower(self, lower_x):
        out_val = lower_x.view(-1, num_planes*8*8)
        out_val = self.fc1(out_val).sigmoid()
        return self.fc2(out_val).sigmoid()

class SimpleCNNLower(BaseNet):
    def __init__(self, channels, kernel_size, stride_size = 1, extra_inputs = None, output_reg = None, output_cls = None):
        super().__init__(channels*6*6, extra_inputs, output_reg, output_cls)

        self.cnn1 = torch.nn.Conv2d(num_planes, channels, kernel_size, stride = stride_size, padding = 1)
        #self.pool = torch.nn.MaxPool2d(2, 2)
        self.cnn2 = torch.nn.Conv2d(channels, channels, kernel_size, stride = stride_size, padding = 1)

    def forward_lower(self, lower_x):
        l_y = self.cnn1(lower_x).relu()
        #l_y = self.pool(l_y)
        l_y = self.cnn2(l_y).relu()
        #l_y = self.pool(l_y)
        return l_y

class LeelaNet(torch.nn.Module):
    def __init__(self, channels, blocks):
        super().__init__()
        self.conv_in = ConvBlock(kernel_size=3,
                               input_channels=num_planes,
                               output_channels=channels)
        self.residual_blocks = []
        for idx in range(blocks):
            block = ResidualBlock(channels)
            self.residual_blocks.append(block)
            self.add_module('residual_block{}'.format(idx+1), block)
        self.conv_out = ConvBlock(kernel_size=1,
                                   input_channels=channels,
                                   output_channels=32)
        self.fc_1 = torch.nn.Linear(32*8*8, 128)
        self.fc_2 = torch.nn.Linear(128, 2)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x.astype(np.float32))
            if self.on_cuda:
                x = x.cuda()
        x = x.view(-1, num_planes, 8, 8)
        out = self.conv_in(x)
        for block in self.residual_blocks:
            out = block(out)
        out_val = self.conv_out(out).view(-1, 32*8*8)
        out_val = self.fc_1(out_val).relu()
        out_val = self.fc_2(out_val).softmax(1)
        return out_val

    def save(self, fname):
        with open(fname, 'wb') as f:
            torch.save(self, f)

class FullyConnectedNet(torch.nn.Module):
    def __init__(self, h_size):
        super().__init__()

        self.fc1 = torch.nn.Linear(num_planes*8*8, h_size)
        self.fc2 = torch.nn.Linear(h_size, h_size)
        self.fc3 = torch.nn.Linear(h_size, 2)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x.astype(np.float32))
            if self.on_cuda:
                x = x.cuda()

        x = x.view(-1, num_planes*8*8)
        out_val = self.fc1(x).sigmoid()
        out_val = self.fc2(out_val).sigmoid()
        return self.fc3(out_val).softmax(1)

    def save(self, fname):
        with open(fname, 'wb') as f:
            torch.save(self, f)

class SimpleCNN(torch.nn.Module):
    def __init__(self, h_size):
        pass
