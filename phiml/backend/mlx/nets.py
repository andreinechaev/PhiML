from typing import Any, Union, Sequence

from mlx.utils import tree_flatten
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from . import MLX
from ... import math


def get_parameters(net: nn.Module) -> dict:
    return {k: v for k, v in tree_flatten(net.parameters())}

def update_weights(net: nn.Module, optimizer: optim.Optimizer, loss_function: Any, x: math.Tensor):
    grads = loss_function(x)
    optimizer.update(model=net, gradients=grads)

def adam(net: nn.Module, learning_rate: float = 1e-3, betas=(0.9, 0.999), epsilon=1e-07):
    return optim.Adam(learning_rate, betas, epsilon)


def sgd(net: nn.Module, learning_rate: float = 1e-3, momentum=0., dampening=0., weight_decay=0., nesterov=False):
    return optim.SGD(learning_rate, momentum, dampening, weight_decay, nesterov)


def adagrad(net: nn.Module, learning_rate: float = 1e-3, lr_decay=0., weight_decay=0., initial_accumulator_value=0., eps=1e-10):
    return optim.Adagrad(learning_rate, lr_decay, weight_decay, initial_accumulator_value, eps)


def rmsprop(net: nn.Module, learning_rate: float = 1e-3, alpha=0.99, eps=1e-08, weight_decay=0., momentum=0., centered=False):
    return optim.RMSprop(learning_rate, alpha, eps, weight_decay, momentum, centered)



def _bias0(conv):
    def initialize(*args, **kwargs):
        module = conv(*args, **kwargs)
        # module.bias.data.fill_(0)
        return module
    return initialize


CONV = [None, _bias0(nn.Conv1d), _bias0(nn.Conv2d), _bias0(nn.Conv3d)]
NORM = [None, nn.GroupNorm, nn.BatchNorm, nn.BatchNorm]
MAX_POOL = [None, nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool2d]
ACTIVATIONS = {'ReLU': nn.ReLU, 'Sigmoid': nn.Sigmoid, 'tanh': nn.Tanh, 'SiLU': nn.SiLU, 'GeLU': nn.GELU}


def u_net(in_channels: int,
          out_channels: int,
          levels: int = 4,
          filters: Union[int, Sequence] = 16,
          batch_norm: bool = True,
          activation: Union[str, type] = 'ReLU',
          in_spatial: Union[tuple, int] = 2,
          periodic=False,
          use_res_blocks: bool = False,
          down_kernel_size=3,
          up_kernel_size=3):
    if isinstance(filters, (tuple, list)):
        assert len(filters) == levels, "Number of filters must match number of levels"
    else:
        filters = (filters,) * levels

    activation = ACTIVATIONS[activation] if isinstance(activation, str) else activation
    if isinstance(in_spatial, int):
        d = in_spatial
    else:
        assert isinstance(in_spatial, tuple)
        d = len(in_spatial)

    net = Unet(d, in_channels, out_channels, filters, batch_norm, activation, periodic, use_res_blocks, down_kernel_size, up_kernel_size)
    return net


class Unet(nn.Module):

    def __init__(self, d: int, in_channels: int, out_channels: int, filters: tuple, batch_norm: bool, activation: type, periodic: bool, use_res_blocks: bool, down_kernel_size: int, up_kernel_size: int):
        super().__init__()
        self._spatial_rank = d
        self._levels = len(filters)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filters = filters
        self.batch_norm = batch_norm
        self.activation = activation
        self.periodic = periodic
        self.use_res_blocks = use_res_blocks
        self.down_kernel_size = down_kernel_size
        self.up_kernel_size = up_kernel_size

        if use_res_blocks:
            self.inc = ResNetBlock(d, in_channels, filters[0], batch_norm, activation, periodic, down_kernel_size)
        else:
            self.inc = DoubleConv(d, in_channels, filters[0], filters[0], batch_norm, activation, periodic, down_kernel_size)
        self.layers_down = []
        self.layers_up = []
        for i in range(1, self._levels):
            self.layers_down.append(Down(d, filters[i - 1], filters[i], batch_norm, activation, use_res_blocks, periodic, down_kernel_size))
            self.layers_up.append(Up(d, filters[i], filters[i - 1], batch_norm, activation, periodic, use_res_blocks, up_kernel_size))

        self.outc = CONV[d](filters[0], out_channels, kernel_size=1, padding=0)

    def __call__(self, x):
        x = self.inc(x)
        xs = [x]
        for i in range(1, self._levels):
            x = self.layers_down[i - 1](x)
            xs.insert(0, x)
        for i in range(1, self._levels):
            x = self.layers_up[i - 1](x, xs[i])
        x = self.outc(x)
        return x


class ResNetBlock(nn.Module):

    def __init__(self, in_spatial, in_channels, out_channels, batch_norm, activation, periodic: bool, kernel_size=3):
        super().__init__()
        if in_channels != out_channels:
            self.sample_input = CONV[in_spatial](in_channels, out_channels, kernel_size=1, padding=0)
            self.bn_sample = NORM[in_spatial](out_channels) if batch_norm else nn.Identity()
        else:
            self.sample_input = nn.Identity()
            self.bn_sample = nn.Identity()
        self.activation = ACTIVATIONS[activation] if isinstance(activation, str) else activation
        self.bn1 = NORM[in_spatial](out_channels) if batch_norm else nn.Identity()
        self.conv1 = CONV[in_spatial](in_channels, out_channels, kernel_size=kernel_size, padding=1)
        self.bn2 = NORM[in_spatial](out_channels) if batch_norm else nn.Identity()
        self.conv2 = CONV[in_spatial](out_channels, out_channels, kernel_size=kernel_size, padding=1)

    def __call__(self, x):
        x = MLX.as_tensor(x)
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = (out + self.bn_sample(self.sample_input(x)))
        return out


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, d: int, in_channels: int, out_channels: int, mid_channels: int, batch_norm: bool, activation: type, periodic: bool, kernel_size=3):
        super().__init__()
        self.double_conv = nn.Sequential(
            CONV[d](in_channels, mid_channels, kernel_size=kernel_size, padding=1),
            NORM[d](mid_channels) if batch_norm else nn.Identity(),
            activation(),
            CONV[d](mid_channels, out_channels, kernel_size=kernel_size, padding=1),
            NORM[d](out_channels) if batch_norm else nn.Identity(),
            nn.ReLU()
        )

    def __call__(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv or resnet_block"""

    def __init__(self, d: int, in_channels: int, out_channels: int, batch_norm: bool, activation: Union[str, type], use_res_blocks: bool, periodic, kernel_size: int):
        super().__init__()
        self.maxpool = MAX_POOL[d](2)
        if use_res_blocks:
            self.conv = ResNetBlock(d, in_channels, out_channels, batch_norm, activation, periodic, kernel_size)
        else:
            self.conv = DoubleConv(d, in_channels, out_channels, out_channels, batch_norm, activation, periodic, kernel_size)

    def __call__(self, x):
        x = self.maxpool(x)
        return self.conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    _MODES = [None, "nearest", "linear"]

    def __init__(self, d: int, in_channels: int, out_channels: int, batch_norm: bool, activation: type, periodic: bool, use_res_blocks: bool, kernel_size: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode=Up._MODES[d])
        if use_res_blocks:
            self.conv = ResNetBlock(d, in_channels, out_channels, batch_norm, activation, periodic, kernel_size)
        else:
            self.conv = DoubleConv(d, in_channels, out_channels, in_channels // 2, batch_norm, activation, periodic, kernel_size)

    def __call__(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        # diff = [x2.size()[i] - x1.size()[i] for i in range(2, len(x1.shape))]
        # x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
        #                 diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = mx.concat([x2, x1], dim=1)
        return self.conv(x)
