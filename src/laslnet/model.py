import math

import torch
from torch import nn

from laslnet.nn import Conv4d, modReLU


class Net(nn.Module):
    def __init__(self, voices):
        super().__init__()
        self.side = math.isqrt(voices)
        if self.side * self.side != voices or self.side < 5:
            raise ValueError(
                "expected a square voice grid with at least five voices per axis"
            )

        self.register_buffer("scale", torch.ones(voices))
        self.features = nn.Sequential(
            Conv4d(1, 4, (3, 3, 3, 3), padding=(0, 0, 1, 1)),
            modReLU(4),
            Conv4d(4, 4, (3, 3, 3, 3), padding=(0, 0, 1, 1)),
            modReLU(4),
        )
        self.blocks = nn.ModuleList(
            nn.Sequential(
                Conv4d(4, 4, (3, 3, 1, 1), padding=(1, 1, 0, 0)),
                modReLU(4),
                Conv4d(4, 4, (3, 3, 1, 1), padding=(1, 1, 0, 0)),
            )
            for _ in range(2)
        )

        channels = 2 * 4 * (self.side - 4) ** 2
        self.head = nn.Sequential(
            nn.Conv2d(channels, 32, 1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
        )

    def forward(self, x):
        x = x / self.scale.view(1, self.side, self.side, 1, 1)
        x = self.features(x.unsqueeze(1))
        for block in self.blocks:
            x = x + block(x)
        x = x.flatten(1, 3)
        x = torch.cat((x.real, x.imag), dim=1)
        return self.head(x)


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
