import fstpack
import torch
from torch import nn
from torch.nn import functional as F


class CReLU(nn.Module):
    def forward(self, z):
        return torch.complex(F.relu(z.real), F.relu(z.imag))


class CLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.1):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, z):
        return torch.complex(
            F.leaky_relu(z.real, negative_slope=self.negative_slope),
            F.leaky_relu(z.imag, negative_slope=self.negative_slope),
        )


class Net(nn.Module):
    def __init__(self, voices, negative_slope=0.1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(voices, 64, 1, dtype=torch.complex64),
            CLeakyReLU(negative_slope),
            nn.Conv2d(64, 64, 3, padding=1, dtype=torch.complex64),
            CLeakyReLU(negative_slope),
            nn.Conv2d(64, 64, 3, padding=1, dtype=torch.complex64),
            CLeakyReLU(negative_slope),
            nn.Conv2d(64, 64, 3, padding=1, dtype=torch.complex64),
            CLeakyReLU(negative_slope),
        )
        self.head = nn.Conv2d(128, 1, 1)

    def forward(self, x):
        z = self.body(x)
        return self.head(torch.cat((z.real, z.imag), dim=1))


def features(a):
    v = fstpack.voices(fstpack.dost(a))
    return v.reshape(1, -1, *v.shape[2:], order="F")


def train(net, data, epochs):
    opt = torch.optim.Adam(net.parameters())
    device = net.head.weight.device

    for _ in range(epochs):
        for a, labels in data():
            x = torch.as_tensor(features(a), device=device)
            y = torch.as_tensor(labels, device=device).float()[None]

            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(net(x)[:, 0], y)
            loss.backward()
            opt.step()

    return net.eval()


@torch.inference_mode()
def predict(net, a):
    x = torch.as_tensor(features(a), device=net.head.weight.device)
    return (net(x)[0, 0] > 0).cpu().numpy()
