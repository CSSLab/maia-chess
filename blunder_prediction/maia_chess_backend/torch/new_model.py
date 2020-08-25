import torch
import torch.nn
import torch.nn.functional

import collections

from ..utils import profile_helper
from .new_blocks import ConvBlock, ResidualBlock, ScalarToChannel, No_op, Flatten

num_planes = 17

move_types = {'move', 'top_nonblunder', 'top_blunder'}
unbounded_types = {'cp_rel', 'active_elo', 'opponent_elo'}

input_scaling_factors = {
    'cp_rel' : .01,
    'clock_percent' : 1.0,
    'move_ply' : .01,
    'opponent_elo' : 1 / 3000,
    'active_elo' : 1 / 3000,
}

validation_stats = [
    "validation_accuracy_pred", "validation_loss_pred", "validation_loss_reg"
]


class NetBaseNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.extras_to_channel = None

    @profile_helper
    def dict_forward(self, input_x, extra_x = None):
        y_vals = self(input_x, extra_x = extra_x)
        return { k : v for k, v in zip(self.outputs, y_vals)}

    def save(self, fname):
        with open(fname, 'wb') as f:
            torch.save(self, f)

    #Not sure why cuda only effects the lower layers
    def cuda(self, device = None):
        self.output_layers = self.output_layers.cuda(device = device)
        try:
            if self.extras_to_channel is not None:
                self.extras_to_channel = self.extras_to_channel.cuda()
        except AttributeError:
            pass
        return super().cuda(device = device)

    def cpu(self):
        self.output_layers = self.output_layers.cpu()
        try:
            if self.extras_to_channel is not None:
                self.extras_to_channel = self.extras_to_channel.cpu()
        except AttributeError:
            pass
        return super().cpu()

    @profile_helper
    def gen_hist_dicts(self):
        params_dict = {}
        for name, pars in self.named_parameters():
            params_dict[name] = pars.detach().cpu().numpy().flatten()
        return params_dict

    @profile_helper
    def train_batch(self, input_x, real_y, optimizer, loss_reg, loss_class):
        optimizer.zero_grad()
        losses = {}
        self.train(mode = True)
        pred_y = self.dict_forward(input_x, extra_x = real_y if self.has_extras else None)
        for i, n in enumerate(self.outputs):
            if n in move_types:
                Y = real_y[n]
                loss_n = loss_class(pred_y[n], Y)
            else:
                Y = real_y[n].float().reshape(-1, 1)
                loss_n = loss_reg(pred_y[n], Y)
            loss_n.backward(retain_graph = i < self.num_outputs - 1)
            losses[n] = loss_n.detach()
        self.train(mode = False)
        optimizer.step()
        return losses

    @profile_helper
    def test_batch(self, input_x, real_y, loss_reg, loss_class):
        losses = {}
        accuracies = {}
        pred_y = self.dict_forward(input_x, extra_x = real_y if self.has_extras else None)
        for n in self.outputs:
            if n in move_types:
                Y = real_y[n]
                loss_n = loss_class(pred_y[n], Y)
                accuracies[n] = (pred_y[n].argmax(1) == Y)[Y > 0].float().mean()
            else:
                Y = real_y[n].float().reshape(-1, 1)
                loss_n = loss_reg(pred_y[n], Y)
                accuracies[n] = (pred_y[n].round().long() == real_y[n].reshape(-1, 1)).float().mean()
            losses[n] = loss_n.detach()

        return losses, accuracies

    @profile_helper
    def validate_batch(self, input_x, real_y):
        results = {}
        pred_y = self.dict_forward(input_x, extra_x = real_y if self.has_extras else None)
        loss_mse = torch.nn.MSELoss(reduction='mean')

        pred_cls = None
        pred_reg = None
        for n in ['is_blunder_wr', 'is_blunder_mean']:
            try:
                pred_cls = pred_y[n]
                break
            except KeyError:
                pass
        for n in ['winrate_loss', 'is_blunder_wr_mean']:
            try:
                pred_reg = pred_y[n]
                if pred_cls is None:
                    pred_cls = pred_reg * 5
                break
            except KeyError:
                pass
        if pred_reg is None:
            # bit of a hack
            pred_reg = pred_cls / 5

        results['validation_accuracy_pred'] = (pred_cls.round().long() == real_y['is_blunder_wr'].reshape(-1, 1)).float().mean()

        results['validation_loss_pred']  = loss_mse(pred_cls, real_y['is_blunder_wr'].float().reshape(-1, 1)).detach()
        results['validation_loss_reg']  = loss_mse(pred_reg, real_y['winrate_loss'].float().reshape(-1, 1)).detach()

        return results

class LeelaNetNew(NetBaseNew):
    @profile_helper
    def __init__(self, channels, blocks, ouputs, extra_inputs = None, extra_top = False, dropout = 0.0):
        super().__init__()
        self.outputs = ouputs
        self.num_outputs = len(ouputs)
        self.channels = channels
        self.blocks = blocks
        self.dropout = dropout
        self.extra_top = extra_top

        self.has_extras = extra_inputs is not None
        self.extra_inputs = list(extra_inputs) if self.has_extras else None
        self.extras_to_channel = None
        if self.has_extras and not self.extra_top:
            extras_to_channel = {}
            for n in self.extra_inputs:
                extras_to_channel[n] = ScalarToChannel(input_scaling_factors.get(n, 1.0))

            self.extras_to_channel = torch.nn.ModuleDict(extras_to_channel)

        lower_layers = [
            ('conv_input', ConvBlock(3, num_planes + (len(extra_inputs) if (self.has_extras and not self.extra_top) else 0), self.channels))
        ]
        for i in range(self.blocks):
            lower_layers.append((f'residual_block_{i + 1}', ResidualBlock(self.channels)))

        self.lower_layers = torch.nn.Sequential(collections.OrderedDict(lower_layers))

        output_layers = {}
        for output_name in self.outputs:
            if output_name in move_types:
                final_size = 2**12
                final_act = No_op
                #final_act = lambda : torch.nn.Softmax(1)
            elif output_name in unbounded_types:
                final_size = 1
                final_act = No_op
            else:
                final_size = 1
                final_act = torch.nn.Sigmoid
            if self.has_extras and self.extra_top:
                ouput_layers_cnn = [
                        (f'{output_name}_conv', ConvBlock(1, self.channels, 32)),
                        (f'{output_name}_flatten', Flatten()),
                    ]
                ouput_layers_fc = [
                        (f'{output_name}_fc_hidden', torch.nn.Linear(32*8*8 + len(extra_inputs), 128)),
                        (f'activation_{output_name}_hidden', torch.nn.ReLU()),
                        (f'{output_name}_fc_output', torch.nn.Linear(128, final_size)),
                        (f'activation_{output_name}_output', final_act()),
                    ]
                output_layers[output_name + '_cnn'] = torch.nn.Sequential(collections.OrderedDict(ouput_layers_cnn))
                output_layers[output_name + '_fc'] = torch.nn.Sequential(collections.OrderedDict(ouput_layers_fc))
            else:
                ouput_layers = [
                        (f'{output_name}_conv', ConvBlock(1, self.channels, 32)),
                        (f'{output_name}_flatten', Flatten()),
                        (f'{output_name}_fc_hidden', torch.nn.Linear(32*8*8, 128)),
                        (f'activation_{output_name}_hidden', torch.nn.ReLU()),
                        (f'{output_name}_fc_output', torch.nn.Linear(128, final_size)),
                        (f'activation_{output_name}_output', final_act()),
                    ]
                output_layers[output_name] = torch.nn.Sequential(collections.OrderedDict(ouput_layers))
        self.output_layers = torch.nn.ModuleDict(output_layers)

    @profile_helper
    def forward(self, input_x, extra_x = None):
        y_vals = []
        if self.extra_inputs is not None:
            if self.extra_top:
                extra_vals = []
                for n in self.extra_inputs:
                    extra_vals.append(extra_x[n])
                input_extra = torch.stack(extra_vals, 1)
            else:
                extra_vals = []
                for n in self.extra_inputs:
                    extra_vals.append(self.extras_to_channel[n](extra_x[n]))
                input_x = torch.cat([input_x] + extra_vals, 1)
        lower_y = self.lower_layers(input_x)
        for n in self.outputs:
            if self.has_extras and self.extra_top:
                cnn_l = self.output_layers[f"{n}_cnn"](lower_y)
                y_vals.append(self.output_layers[f"{n}_fc"](torch.cat([cnn_l, input_extra], 1)))
            else:
                y_vals.append(self.output_layers[n](lower_y))
        return tuple(y_vals)

class FullyConnectedNetNew(NetBaseNew):
    @profile_helper
    def __init__(self, layer_dims, ouputs, extra_inputs = None, dropout = 0.0):
        super().__init__()

        self.outputs = ouputs
        self.num_outputs = len(ouputs)
        self.layer_dims = layer_dims
        self.num_lower = len(layer_dims)
        self.extra_inputs = extra_inputs
        self.has_extras = extra_inputs is not None
        self.dropout = dropout

        act_func = torch.nn.Tanh

        input_dim = num_planes*8*8

        lower_layers = [
                ('flatten', Flatten()),
                ('dropout_1', torch.nn.Dropout(self.dropout)),
                ('lower_1', torch.nn.Linear(input_dim, layer_dims[0])),
                ('activation_lower_1', act_func())]
        for i in range(len(layer_dims) - 1):
            layer_n = [
                    (f'lower_{i + 2}', torch.nn.Linear(layer_dims[i], layer_dims[i + 1])),
                    (f'dropout_{i + 2}', torch.nn.Dropout(self.dropout)),
                    (f'activation_lower_{i+2}', act_func()),
            ]
            lower_layers = lower_layers + layer_n
        self.lower_layers = torch.nn.Sequential(collections.OrderedDict(lower_layers))

        output_layers = {}

        output_dims = layer_dims[-1]
        if self.extra_inputs is not None:
            output_dims += len(extra_inputs)

        for output_name in self.outputs:
            if output_name in move_types:
                final_size = 2**12
                final_act = lambda : torch.nn.Softmax(1)
            elif output_name in unbounded_types:
                final_size = 1
                final_act = No_op
            else:
                final_size = 1
                final_act = torch.nn.Sigmoid
            ouput_layers = [
                    (f'{output_name}_hidden', torch.nn.Linear(output_dims, output_dims)),
                    (f'{output_name}_dropout', torch.nn.Dropout(self.dropout)),
                    (f'activation_{output_name}_hidden', act_func()),
                    (f'{output_name}_output', torch.nn.Linear(output_dims, final_size)),
                    (f'activation_{output_name}_output', final_act()),
            ]
            output_layers[output_name] = torch.nn.Sequential(collections.OrderedDict(ouput_layers))
        self.output_layers = torch.nn.ModuleDict(output_layers)

    @profile_helper
    def forward(self, input_x, extra_x = None):
        y_vals = []
        lower_y = self.lower_layers(input_x)
        try:
            if self.extra_inputs is not None:
                extra_vals = [extra_x[k].view(-1, 1).float() for k in self.extra_inputs]

                lower_y = torch.cat([lower_y] + extra_vals, dim = 1)
        except AttributeError:
            pass

        for n in self.outputs:
            y_vals.append(self.output_layers[n](lower_y))
        return tuple(y_vals)

def NetFromConfigNew(config):
    if config['type'] == 'fully_connected':
        return FullyConnectedNetNew(
                config['hidden_layers'],
                config['outputs'],
                extra_inputs = config.get('inputs', None),
                dropout = config.get('dropout', 0.0),
                )
    if config['type'] == 'leela':
        return LeelaNetNew(
                config['channels'],
                config['blocks'],
                config['outputs'],
                extra_inputs = config.get('inputs', None),
                extra_top = config.get('extra_top', False),
                #dropout = config.get('dropout', 0.0),
                )
    raise RuntimeError(f"{config['type']} is not a valid model type")
