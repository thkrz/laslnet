import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from numpy.lib.stride_tricks import sliding_window_view
from torch import nn
from torch.nn import functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def dost(im, scale=0.5):
    N = im.shape[-1]
    m = N // 2
    n = N.bit_length() - 2
    IM = torch.fft.fft2(im) / (N * N)
    S = torch.zeros_like(IM)

    S[..., 0, 0] = IM[..., 0, 0]
    S[..., 0, m] = IM[..., 0, m]
    S[..., m, 0] = IM[..., m, 0]
    S[..., m, m] = IM[..., m, m]

    for py in range(n):
        ny = 1 << py
        pos = slice(ny, 2 * ny)
        neg = slice(N - 2 * ny + 1, N - ny + 1)
        rows = torch.stack(
            (
                IM[..., pos, 0],
                IM[..., neg, 0],
                IM[..., pos, m],
                IM[..., neg, m],
            )
        )
        rows = torch.fft.ifft(torch.fft.fftshift(rows, dim=-1), dim=-1)
        rows = rows * (ny**scale)
        S[..., pos, 0] = rows[0]
        S[..., neg, 0] = rows[1].flip(-1)
        S[..., pos, m] = rows[2]
        S[..., neg, m] = rows[3].flip(-1)

    for px in range(n):
        nx = 1 << px
        col = slice(nx, 2 * nx)
        x = torch.fft.ifft(torch.fft.fftshift(IM[..., col], dim=-1), dim=-1)
        x = x * (nx**scale)
        S[..., 0, col] = x[..., 0, :]
        S[..., m, col] = x[..., m, :]

        for py in range(n):
            ny = 1 << py
            pos = slice(ny, 2 * ny)
            neg = slice(N - 2 * ny + 1, N - ny + 1)
            y = torch.stack((x[..., pos, :], x[..., neg, :]))
            y = torch.fft.ifft(torch.fft.fftshift(y, dim=-2), dim=-2)
            y = y * (ny**scale)
            S[..., pos, col] = y[0]
            S[..., neg, col] = y[1].flip(-2)

    S[..., 1:m, m + 1 :] = S[..., m + 1 :, 1:m].flip(-1).flip(-2).conj()
    S[..., m + 1 :, m + 1 :] = S[..., 1:m, 1:m].flip(-1).flip(-2).conj()
    S[..., 0, m + 1 :] = S[..., 0, 1:m].flip(-1).conj()
    S[..., m, m + 1 :] = S[..., m, 1:m].flip(-1).conj()
    return S


def voice(s, x, y):
    k = s.shape[-1]
    n = k.bit_length() - 2
    v = 2 * n + 2
    ix = torch.empty(v, dtype=torch.long, device=s.device)
    iy = torch.empty(v, dtype=torch.long, device=s.device)

    for i, p in enumerate(range(-n, n + 2)):
        if p == 0:
            bx = by = 0
        elif p == n + 1:
            bx = by = k // 2
        else:
            b = 1 << (abs(p) - 1)
            tx = x * b // k
            ty = y * b // k
            if p > 0:
                bx, by = b + tx, b + ty
            else:
                bx, by = k - b - tx, k - b - ty
        ix[i] = bx
        iy[i] = by

    return s.index_select(-2, ix).index_select(-1, iy)


def embed(data, patch, window, device):
    half = window // 2
    voices = 2 * (window.bit_length() - 1)
    out = np.empty((patch * patch, voices * voices), np.complex64)
    wins = sliding_window_view(data, (window, window))
    batch = max(1, (16 * 1024 * 1024) // (4 * window * window))

    for i in range(0, len(out), batch):
        j = min(i + batch, len(out))
        indices = np.arange(i, j)
        samples = wins[indices // patch, indices % patch]
        x = torch.as_tensor(samples, device=device)
        x = x - x.mean((-1, -2), keepdim=True)
        z = voice(dost(x), half, half)
        out[i:j] = z.reshape(j - i, -1).cpu().numpy()

    return out.reshape(patch, patch, -1).transpose(2, 0, 1)[None]


def shape(path):
    with h5py.File(path, "r") as f:
        window = int(f.attrs["w"])
        patch = int(f.attrs["p"])
    return (2 * (window.bit_length() - 1)) ** 2, patch


def samples(path):
    with h5py.File(path, "r") as f:
        data = f["DATA"]
        mask = f["MASK"]
        patch = int(f.attrs["p"])
        window = int(f.attrs["w"])
        i = 0
        count = data.shape[0]
        while i < count:
            yield embed(data[i], patch, window, device), mask[i]
            i += 1


def posweight(path):
    pos = 0
    total = 0

    with h5py.File(path, "r") as f:
        mask = f["MASK"]
        for y in mask:
            pos += np.count_nonzero(y)
            total += y.size

    return (total - pos) / pos


def main(argv=None):
    parser = argparse.ArgumentParser(description="train a landslide detector")
    parser.add_argument("-e", "--epochs", type=int, default=20, help="training epochs")
    parser.add_argument("data", type=Path, help="context file")
    parser.add_argument("model", type=Path, help="output model")
    args = parser.parse_args(argv)

    weight = torch.tensor(posweight(args.data), device=device)
    voices, patch = shape(args.data)
    net = Net(voices).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=0.0003)

    for epoch in range(args.epochs):
        total = 0.0
        count = 0
        for x, y in samples(args.data):
            x = torch.as_tensor(x, device=device)
            y = torch.as_tensor(y, dtype=torch.float32, device=device)
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(
                net(x)[0, 0], y, pos_weight=weight
            )
            loss.backward()
            opt.step()
            total += loss.item()
            count += 1
        print(
            f"epoch={epoch + 1} loss={total / count:.6f}",
            file=sys.stderr,
            flush=True,
        )

    state = {k: v.detach().cpu() for k, v in net.state_dict().items()}
    torch.save({"voices": voices, "n": patch, "state": state}, args.model)
    return 0


if __name__ == "__main__":
    sys.exit(main())
