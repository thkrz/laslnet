"""4-D convolution and modReLU for tensors shaped (N, C, D1, D2, H, W)."""

import math

import torch
from torch import nn
from torch.nn import functional as F


def _quad(value):
    if isinstance(value, int):
        return (value,) * 4
    value = tuple(value)
    if len(value) != 4:
        raise ValueError("expected an integer or a four-element sequence")
    return value


class Conv4d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        bias=True,
        device=None,
        dtype=torch.complex64,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _quad(kernel_size)
        self.stride = _quad(stride)
        self.padding = _quad(padding)

        if min(self.kernel_size) < 1:
            raise ValueError("kernel sizes must be positive")
        if min(self.stride) < 1:
            raise ValueError("strides must be positive")
        if min(self.padding) < 0:
            raise ValueError("padding must be nonnegative")

        self.weight = nn.Parameter(
            torch.empty(
                out_channels,
                in_channels,
                *self.kernel_size,
                device=device,
                dtype=dtype,
            )
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_channels, device=device, dtype=dtype)
            )
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_channels * math.prod(self.kernel_size)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        p1, p2, ph, pw = self.padding
        if any(self.padding):
            x = F.pad(x, (pw, pw, ph, ph, p2, p2, p1, p1))

        batch, channels, length, depth, height, width = x.shape
        step = self.stride[0]
        count = (length - self.kernel_size[0]) // step + 1
        if count < 1:
            raise ValueError("input is smaller than the kernel along the first axis")

        out = None
        for i in range(self.kernel_size[0]):
            part = x[:, :, i : i + count * step : step]
            part = part.permute(0, 2, 1, 3, 4, 5).reshape(
                batch * count, channels, depth, height, width
            )
            part = F.conv3d(
                part,
                self.weight[:, :, i],
                stride=self.stride[1:],
            )
            out = part if out is None else out + part

        out = out.reshape(batch, count, self.out_channels, *out.shape[2:])
        out = out.permute(0, 2, 1, 3, 4, 5).contiguous()
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1, 1, 1)
        return out


class modReLU(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.bias = nn.Parameter(torch.full((channels,), -0.1))

    def forward(self, z):
        m = z.abs()
        b = self.bias.view(1, -1, *((1,) * (z.ndim - 2)))
        s = F.relu(m + b) / m.clamp_min(1e-8)
        return z * s
