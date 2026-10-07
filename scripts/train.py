import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.nn import functional as F

from nn import Net
from u import positive

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def ivoice(k, x, y, device):
    n = k.bit_length() - 2
    ix = []
    iy = []

    for p in range(-n, n + 2):
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
        ix.append(bx)
        iy.append(by)

    return (
        torch.tensor(ix, dtype=torch.long, device=device),
        torch.tensor(iy, dtype=torch.long, device=device),
    )


@torch.no_grad()
def embed(data, patch, window, device):
    half = window // 2
    voices = 2 * (window.bit_length() - 1)
    channels = voices * voices
    ix, iy = ivoice(window, half, half, device)

    data = torch.as_tensor(data, dtype=torch.float32, device=device)
    wins = data.unfold(0, window, 1).unfold(1, window, 1)
    out = torch.empty(
        (patch * patch, channels),
        dtype=torch.complex64,
        device=device,
    )
    batch = max(1, (256 * 1024 * 1024) // (4 * window * window))

    for i in range(0, len(out), batch):
        j = min(i + batch, len(out))
        indices = torch.arange(i, j, device=device)
        x = wins[indices // patch, indices % patch]
        x = x - x.mean((-1, -2), keepdim=True)
        s = dost(x)
        z = s.index_select(-2, ix).index_select(-1, iy)
        out[i:j] = z.reshape(j - i, channels)
        del x, s, z

    return out.reshape(patch, patch, channels).permute(2, 0, 1).contiguous()


def shape(path):
    with h5py.File(path, "r") as f:
        window = int(f.attrs["w"])
        patch = int(f.attrs["p"])
    return (2 * (window.bit_length() - 1)) ** 2, patch


def samples(path, batch_size):
    with h5py.File(path, "r") as f:
        data = f["DATA"]
        mask = f["MASK"]
        patch = int(f.attrs["p"])
        window = int(f.attrs["w"])
        count = data.shape[0]

        for first in range(0, count, batch_size):
            last = min(first + batch_size, count)
            x = torch.stack(
                [embed(data[i], patch, window, device) for i in range(first, last)]
            )
            y = torch.as_tensor(
                mask[first:last],
                dtype=torch.float32,
                device=device,
            )
            yield x, y


def posweight(path):
    pos = 0
    total = 0

    with h5py.File(path, "r") as f:
        mask = f["MASK"]
        for y in mask:
            pos += np.count_nonzero(y)
            total += y.size

    return (total - pos) / pos


@torch.no_grad()
def validate(net, path, batch_size):
    net.eval()
    intersection = torch.zeros((), dtype=torch.int64, device=device)
    union = torch.zeros((), dtype=torch.int64, device=device)

    for x, y in samples(path, batch_size):
        pred = net(x)[:, 0] >= 0
        truth = y != 0
        intersection += (pred & truth).sum()
        union += (pred | truth).sum()

    if union.item() == 0:
        raise ValueError("validation IoU is undefined: empty union")

    return (intersection.double() / union).item()


def main(argv=None):
    parser = argparse.ArgumentParser(description="train a landslide detector")
    parser.add_argument(
        "-b",
        "--batch-size",
        type=positive,
        default=8,
        help="training and validation batch size",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        type=positive,
        default=20,
        help="training epochs",
    )
    parser.add_argument("data", type=Path, help="training contexts")
    parser.add_argument("validation", type=Path, help="validation contexts")
    parser.add_argument("model", type=Path, help="output model")
    args = parser.parse_args(argv)

    voices, patch = shape(args.data)
    if shape(args.validation) != (voices, patch):
        parser.error("training and validation dimensions must match")

    weight = torch.tensor(posweight(args.data), device=device)
    net = Net(voices).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=0.0003)
    best = -1.0

    for epoch in range(args.epochs):
        net.train()
        total = torch.zeros((), device=device)
        count = 0

        for x, y in samples(args.data, args.batch_size):
            opt.zero_grad(set_to_none=True)
            logits = net(x)[:, 0]
            loss = F.binary_cross_entropy_with_logits(
                logits,
                y,
                pos_weight=weight,
            )
            loss.backward()
            opt.step()

            size = y.shape[0]
            total += loss.detach() * size
            count += size

        train_loss = total.item() / count
        iou = validate(net, args.validation, args.batch_size)
        improved = iou > best

        if improved:
            state = {k: v.detach().cpu() for k, v in net.state_dict().items()}
            torch.save(
                {"voices": voices, "n": patch, "state": state},
                args.model,
            )
            best = iou

        print(
            f"epoch={epoch + 1} loss={train_loss:.6f} "
            f"val_iou={iou:.6f} best_iou={best:.6f} "
            f"saved={int(improved)}",
            file=sys.stderr,
            flush=True,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
