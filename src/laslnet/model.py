import torch
from torch import nn
from torch.nn import functional as F


class modReLU(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.bias = nn.Parameter(torch.full((channels,), -0.1))

    def forward(self, z):
        m = z.abs()
        b = self.bias.view(1, -1, *((1,) * (z.ndim - 2)))
        s = F.relu(m + b) / m.clamp_min(1e-8)
        return z * s


class Net(nn.Module):
    def __init__(self, voices):
        super().__init__()
        self.register_buffer("scale", torch.ones(voices))
        self.layers = nn.Sequential(
            nn.Conv2d(2 * voices, 32, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
        )

    def forward(self, x):
        x = x / self.scale.view(1, -1, 1, 1)
        x = torch.cat((x.real, x.imag), dim=1)
        return self.layers(x)


def core(logits, halo):
    if halo == 0:
        return logits
    return logits[..., halo:-halo, halo:-halo]


def save(path, net, voices, patch, window, halo):
    state = {k: v.detach().cpu() for k, v in net.state_dict().items()}
    torch.save(
        {
            "voices": voices,
            "n": patch,
            "w": window,
            "h": halo,
            "preprocess": "center-dost-ortho-mean-rms-v1",
            "state": state,
        },
        path,
    )


def load(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    if ckpt.get("preprocess") != "center-dost-ortho-mean-rms-v1":
        raise ValueError("unsupported checkpoint preprocessing")

    voices = int(ckpt["voices"])
    patch = int(ckpt["n"])
    window = int(ckpt["w"])
    halo = int(ckpt["h"])
    side = 2 * (window.bit_length() - 1)
    if side * side != voices:
        raise ValueError("voices and window mismatch")
    if patch < 16 or patch % 8:
        raise ValueError("invalid patch size")
    if halo < 0 or halo % 8:
        raise ValueError("invalid halo")

    net = Net(voices).to(device)
    net.load_state_dict(ckpt["state"])
    net.eval()
    return net, patch, window, halo
