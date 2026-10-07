import torch
from torch import nn
from torch.nn import functional as F


class modReLU(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.bias = nn.Parameter(torch.full((channels,), -0.01))

    def forward(self, z):
        m = z.abs()
        b = self.bias.view(1, -1, 1, 1)
        s = F.relu(m + b) / m.clamp_min(1e-8)
        return z * s


def block(cin, cout, stride=1):
    c = torch.complex64
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride=stride, padding=1, dtype=c),
        modReLU(cout),
        nn.Conv2d(cout, cout, 3, padding=1, dtype=c),
        modReLU(cout),
    )


class Net(nn.Module):
    def __init__(self, voices):
        super().__init__()
        c = torch.complex64
        self.e1 = block(voices, 64)
        self.e2 = block(64, 128, 2)
        self.e3 = block(128, 256, 2)
        self.b = block(256, 512, 2)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2, dtype=c)
        self.d3 = block(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2, dtype=c)
        self.d2 = block(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2, dtype=c)
        self.d1 = block(128, 64)
        self.head = nn.Conv2d(128, 1, 1)

    def forward(self, x):
        x1 = self.e1(x)
        x2 = self.e2(x1)
        x3 = self.e3(x2)
        x = self.b(x3)
        x = self.d3(torch.cat((self.up3(x), x3), 1))
        x = self.d2(torch.cat((self.up2(x), x2), 1))
        x = self.d1(torch.cat((self.up1(x), x1), 1))
        return self.head(torch.cat((x.real, x.imag), 1))
