#!/usr/bin/env python3
import argparse
import random
import sys
from pathlib import Path

import fiona
import h5py
import numpy as np
import rasterio
import torch
from rasterio.features import rasterize
from torch import nn
from torch.nn import functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def margin(window):
    half = window // 2
    return half, half - 1


def band(a):
    b = a.shape[-1]
    z = (a[..., ::2].sum(-1) - a[..., 1::2].sum(-1)) * (b**-0.5)
    return -z if b == 2 else z


def dost(im):
    N = im.shape[-1]
    m = N // 2
    n = N.bit_length() - 2
    IM = torch.fft.fft2(im) / (N * N)

    cols = [IM[..., 0]]
    for p in range(n):
        b = 1 << p
        cols.append(band(IM[..., b : 2 * b]))
    cols.append(IM[..., m])
    x = torch.stack(cols, dim=-1)

    rows = []
    for p in range(-n, n + 2):
        if p == 0:
            rows.append(x[..., 0, :])
        elif p == n + 1:
            rows.append(x[..., m, :])
        else:
            b = 1 << (abs(p) - 1)
            sl = slice(b, 2 * b) if p > 0 else slice(N - 2 * b + 1, N - b + 1)
            rows.append(band(x[..., sl, :].transpose(-2, -1)))

    z = torch.stack(rows, dim=-2)
    neg = torch.cat(
        (
            z[..., : 2 * n + 1, 1 : n + 1].flip((-2, -1)),
            z[..., 2 * n + 1 :, 1 : n + 1].flip(-1),
        ),
        dim=-2,
    ).conj()
    return torch.cat((neg, z), dim=-1)


@torch.no_grad()
def embed(data, field, window, device):
    voices = 2 * (window.bit_length() - 1)
    channels = voices * voices

    data = torch.as_tensor(data, dtype=torch.float32, device=device).contiguous()
    wins = data.unfold(0, window, 1).unfold(1, window, 1)
    out = torch.empty((field * field, channels), dtype=torch.complex64, device=device)
    batch = max(1, (256 * 1024 * 1024) // (4 * window * window))

    for i in range(0, len(out), batch):
        j = min(i + batch, len(out))
        indices = torch.arange(i, j, device=device)
        x = wins[indices // field, indices % field]
        x = x - x.mean((-1, -2), keepdim=True)
        z = dost(x)
        out[i:j] = z.reshape(j - i, channels)
        del x, z

    return out.reshape(field, field, channels).permute(2, 0, 1).contiguous()


def shape(path):
    with h5py.File(path, "r") as f:
        window = int(f.attrs["w"])
        patch = int(f.attrs["p"])
        halo = int(f.attrs["h"])
    voices = (2 * (window.bit_length() - 1)) ** 2
    return voices, patch, window, halo


def samples(path, split, batch, shuffle=False):
    with h5py.File(path, "r") as f:
        group = f[split]
        data = group["DATA"]
        mask = group["MASK"]
        patch = int(f.attrs["p"])
        window = int(f.attrs["w"])
        halo = int(f.attrs["h"])
        field = patch + 2 * halo
        order = list(range(data.shape[0]))

        if shuffle:
            random.shuffle(order)

        for first in range(0, len(order), batch):
            indices = order[first : first + batch]
            x = torch.stack([embed(data[i], field, window, device) for i in indices])
            y = torch.as_tensor(
                np.stack([mask[i] for i in indices]),
                dtype=torch.float32,
                device=device,
            )
            yield x, y


def posweight(path):
    pos = 0
    total = 0
    with h5py.File(path, "r") as f:
        mask = f["train"]["MASK"]
        for y in mask:
            pos += np.count_nonzero(y)
            total += y.size
    return (total - pos) / pos


@torch.no_grad()
def scale(path):
    voices, _, window, _ = shape(path)
    energy = torch.zeros(voices, dtype=torch.float64, device=device)
    count = 0

    for x, _ in samples(path, "train", 1):
        energy += x.abs().square().sum((0, 2, 3), dtype=torch.float64)
        count += x.shape[0] * x.shape[2] * x.shape[3]

    sc = (energy / count).sqrt().float()
    sc[sc < 1e-6] = 1
    side = 2 * (window.bit_length() - 1)
    dc = side // 2 - 1
    sc[dc * side + dc] = 1
    return sc


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


def load(path):
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


def annotations(path, shape, transform):
    with fiona.open(path) as src:
        shapes = [
            (feature["geometry"], 1)
            for feature in src
            if feature["geometry"] is not None
        ]
    return rasterize(
        shapes,
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=True,
    )


def tiles(bad, ann, patch, half, far, halo):
    height, width = ann.shape
    pos = []
    neg = []

    for row in range(half + halo, height - patch - far - halo + 1, patch):
        for col in range(half + halo, width - patch - far - halo + 1, patch):
            if bad[
                row - halo - half : row + patch + halo + far,
                col - halo - half : col + patch + halo + far,
            ].any():
                continue
            if ann[row : row + patch, col : col + patch].any():
                pos.append((row, col))
            else:
                neg.append((row, col))

    return pos, neg


def partition(coords, axis, cut, patch, half, far, halo):
    training = []
    validation = []

    for coord in coords:
        start = coord[axis] - halo - half
        stop = coord[axis] + patch + halo + far
        if stop <= cut:
            training.append(coord)
        elif start >= cut:
            validation.append(coord)

    return training, validation


def write(group, dem, ann, coords, patch, half, far, halo):
    size = patch + 2 * halo + half + far
    count = len(coords)
    data = group.create_dataset(
        "DATA",
        shape=(count, size, size),
        dtype=np.float32,
        chunks=(1, size, size),
        compression="gzip",
        shuffle=True,
    )
    mask = group.create_dataset(
        "MASK",
        shape=(count, patch, patch),
        dtype=np.uint8,
        chunks=(1, patch, patch),
        compression="gzip",
    )
    for i, (row, col) in enumerate(coords):
        data[i] = dem[
            row - halo - half : row + patch + halo + far,
            col - halo - half : col + patch + halo + far,
        ]
        mask[i] = ann[row : row + patch, col : col + patch]


def prepare(args):
    parser = args.parser
    if args.patch % 8:
        parser.error("patch size must be a multiple of 8")
    if args.halo < 0 or args.halo % 8:
        parser.error("halo must be a nonnegative multiple of 8")

    half, far = margin(args.window)

    with rasterio.open(args.dem) as src:
        raster = src.read(1, masked=True)
        dem = np.asarray(raster.data, dtype=np.float32)
        bad = np.ma.getmaskarray(raster) | ~np.isfinite(raster.data)
        ann = annotations(args.annotation, (src.height, src.width), src.transform)

    pos, neg = tiles(bad, ann, args.patch, half, far, args.halo)
    target = int(len(pos) * args.ratio)
    if not 0 < target < len(pos):
        parser.error("ratio leaves no annotated patches in one split")

    height, width = ann.shape
    axis = 1 if width >= height else 0
    ordered = sorted(pos, key=lambda coord: coord[axis])
    cut = ordered[-target][axis] - args.halo - half

    train_pos, val_pos = partition(pos, axis, cut, args.patch, half, far, args.halo)
    train_neg, val_neg = partition(neg, axis, cut, args.patch, half, far, args.halo)

    if not train_pos:
        parser.error("no annotated patches in training split")
    if not val_pos:
        parser.error("no annotated patches in validation split")
    if len(train_neg) < len(train_pos):
        parser.error("insufficient empty patches in training split")
    if len(val_neg) < len(val_pos):
        parser.error("insufficient empty patches in validation split")

    rng = random.Random(0)
    training = train_pos + rng.sample(train_neg, len(train_pos))
    validation = val_pos + rng.sample(val_neg, len(val_pos))
    rng.shuffle(training)
    rng.shuffle(validation)

    with h5py.File(args.context, "x") as dst:
        dst.attrs["p"] = args.patch
        dst.attrs["w"] = half + far + 1
        dst.attrs["h"] = args.halo
        write(
            dst.create_group("train"),
            dem,
            ann,
            training,
            args.patch,
            half,
            far,
            args.halo,
        )
        write(
            dst.create_group("val"),
            dem,
            ann,
            validation,
            args.patch,
            half,
            far,
            args.halo,
        )

    dropped_pos = len(pos) - len(train_pos) - len(val_pos)
    dropped_neg = len(neg) - len(train_neg) - len(val_neg)
    actual = len(val_pos) / (len(train_pos) + len(val_pos))
    direction = "columns" if axis == 1 else "rows"

    print(
        f"split_axis={direction} cut={cut} "
        f"target_fraction={args.ratio:.4f} "
        f"actual_fraction={actual:.4f}"
    )
    print(f"crossing annotated={dropped_pos} empty={dropped_neg}")
    print(
        f"training annotated={len(train_pos)} empty={len(train_pos)} total={len(training)}"
    )
    print(
        f"validation annotated={len(val_pos)} empty={len(val_pos)} total={len(validation)}"
    )
    return 0


@torch.no_grad()
def validate(net, path, batch, halo):
    net.eval()
    intersection = 0
    union = 0

    for x, y in samples(path, "val", batch):
        pred = core(net(x)[:, 0], halo) >= 0
        truth = y != 0
        intersection += (pred & truth).sum().item()
        union += (pred | truth).sum().item()

    if union == 0:
        raise ValueError("validation IoU undefined: empty union")

    return intersection / union


def train(args):
    parser = args.parser
    voices, patch, window, halo = shape(args.context)
    if patch < 16 or patch % 8:
        parser.error("patch size must be >= 16 and a multiple of 8")
    if halo < 0 or halo % 8:
        parser.error("halo must be a nonnegative multiple of 8")

    net = Net(voices).to(device)
    net.scale.copy_(scale(args.context))
    weight = torch.tensor(posweight(args.context), device=device)
    opt = torch.optim.Adam(net.parameters(), lr=0.0003)
    best = -1.0

    for epoch in range(args.epochs):
        net.train()
        total = torch.zeros((), device=device)
        count = 0

        for x, y in samples(args.context, "train", args.batch, shuffle=True):
            opt.zero_grad(set_to_none=True)
            logits = core(net(x)[:, 0], halo)
            loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=weight)
            loss.backward()
            opt.step()

            size = y.shape[0]
            total += loss.detach() * size
            count += size

        train_loss = total.item() / count
        iou = validate(net, args.context, args.batch, halo)
        improved = iou > best

        if improved:
            save(args.model, net, voices, patch, window, halo)
            best = iou

        print(
            f"epoch={epoch + 1} loss={train_loss:.6f} "
            f"val_iou={iou:.6f} best_iou={best:.6f} "
            f"saved={int(improved)}",
            file=sys.stderr,
            flush=True,
        )

    return 0


def origins(lo, hi, step):
    if hi < lo:
        return []
    xs = list(range(lo, hi + 1, step))
    if xs[-1] != hi:
        xs.append(hi)
    return xs


@torch.no_grad()
def masks(net, dem, bad, patch, window, halo):
    half, far = margin(window)
    height, width = dem.shape
    out = np.full((height, width), 255, np.uint8)
    rows = origins(half + halo, height - patch - far - halo, patch)
    cols = origins(half + halo, width - patch - far - halo, patch)
    done = 0
    skipped = 0

    for row in rows:
        for col in cols:
            r0 = row - halo - half
            c0 = col - halo - half
            r1 = row + patch + halo + far
            c1 = col + patch + halo + far
            if bad[r0:r1, c0:c1].any():
                skipped += 1
                continue
            x = embed(dem[r0:r1, c0:c1], patch + 2 * halo, window, device)
            pred = (core(net(x.unsqueeze(0))[0, 0], halo) >= 0).cpu().numpy()
            out[row : row + patch, col : col + patch] = pred
            done += 1

    return out, done, skipped, len(rows) * len(cols)


def predict(args):
    parser = args.parser
    net, patch, window, halo = load(args.model)
    if patch % 8:
        parser.error("checkpoint patch size must be a multiple of 8")
    if halo < 0 or halo % 8:
        parser.error("checkpoint halo must be a nonnegative multiple of 8")

    with rasterio.open(args.dem) as src:
        raster = src.read(1, masked=True)
        dem = np.asarray(raster.data, dtype=np.float32)
        bad = np.ma.getmaskarray(raster) | ~np.isfinite(raster.data)
        profile = {
            "driver": "GTiff",
            "height": src.height,
            "width": src.width,
            "count": 1,
            "dtype": "uint8",
            "crs": src.crs,
            "transform": src.transform,
            "nodata": 255,
            "compress": "deflate",
        }

    need = patch + 2 * halo + window - 1
    if dem.shape[0] < need or dem.shape[1] < need:
        parser.error("raster smaller than patch + halo + window")

    mask, done, skipped, total = masks(net, dem, bad, patch, window, halo)
    with rasterio.open(args.mask, "w", **profile) as dst:
        dst.write(mask, 1)

    print(
        f"patch={patch} window={window} halo={halo} "
        f"tiles={done} skipped={skipped} total={total}"
    )
    return 0


def fraction(value):
    n = float(value)
    if not 0 < n < 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return n


def positive(value):
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return n


def nonnegative(value):
    n = int(value)
    if n < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return n


def powersz(value):
    n = positive(value)
    if n < 2 or n & (n - 1):
        raise argparse.ArgumentTypeError("must be a power of two")
    return n


def main(argv=None):
    parser = argparse.ArgumentParser(prog="laslnet")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare", help="prepare balanced HDF5 contexts")
    p.add_argument(
        "-p", dest="patch", type=positive, default=128, help="patch size [%(default)d]"
    )
    p.add_argument(
        "-w", dest="window", type=powersz, default=128, help="window size [%(default)d]"
    )
    p.add_argument(
        "-H",
        dest="halo",
        type=nonnegative,
        default=16,
        help="feature halo [%(default)d]",
    )
    p.add_argument(
        "-v",
        dest="ratio",
        type=fraction,
        default=0.2,
        help="validation ratio [%(default)f]",
    )
    p.add_argument("dem", type=Path, help="input elevation raster")
    p.add_argument("annotation", type=Path, help="input annotation vectors")
    p.add_argument("context", type=Path, help="output data contexts (h5)")
    p.set_defaults(func=prepare, parser=p)

    t = sub.add_parser("train", help="train detector")
    t.add_argument(
        "-b", dest="batch", type=positive, default=8, help="batch size [%(default)d]"
    )
    t.add_argument(
        "-e", dest="epochs", type=positive, default=20, help="epochs [%(default)d]"
    )
    t.add_argument("context", type=Path, help="data contexts (h5)")
    t.add_argument("model", type=Path, help="output model")
    t.set_defaults(func=train, parser=t)

    g = sub.add_parser("predict", help="predict landslide mask")
    g.add_argument("model", type=Path, help="trained model")
    g.add_argument("dem", type=Path, help="elevation raster")
    g.add_argument("mask", type=Path, help="output mask")
    g.set_defaults(func=predict, parser=g)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

# class modReLU(nn.Module):
#     def __init__(self, channels):
#         super().__init__()
#         self.bias = nn.Parameter(torch.full((channels,), -0.1))

#     def forward(self, z):
#         m = z.abs()
#         b = self.bias.view(1, -1, 1, 1)
#         s = F.relu(m + b) / m.clamp_min(1e-8)
#         return z * s


# class ReflectConv(nn.Module):
#     def __init__(self, cin, cout, stride=1):
#         super().__init__()
#         self.conv = nn.Conv2d(cin, cout, 3, stride=stride, dtype=torch.complex64)

#     def forward(self, x):
#         real = F.pad(x.real, (1, 1, 1, 1), mode="reflect")
#         imag = F.pad(x.imag, (1, 1, 1, 1), mode="reflect")
#         return self.conv(torch.complex(real, imag))


# def block(cin, cout, stride=1):
#     return nn.Sequential(
#         ReflectConv(cin, cout, stride),
#         modReLU(cout),
#         ReflectConv(cout, cout),
#         modReLU(cout),
#     )


# class Net(nn.Module):
#     def __init__(self, voices):
#         super().__init__()
#         c = torch.complex64
#         self.register_buffer("scale", torch.ones(voices))
#         self.e1 = block(voices, 64)
#         self.e2 = block(64, 128, 2)
#         self.e3 = block(128, 256, 2)
#         self.b = block(256, 512, 2)
#         self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2, dtype=c)
#         self.d3 = block(512, 256)
#         self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2, dtype=c)
#         self.d2 = block(256, 128)
#         self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2, dtype=c)
#         self.d1 = block(128, 64)
#         self.head = nn.Conv2d(128, 1, 1)
#
#     def forward(self, x):
#         x = x / self.scale.view(1, -1, 1, 1)
#         x1 = self.e1(x)
#         x2 = self.e2(x1)
#         x3 = self.e3(x2)
#         x = self.b(x3)
#         x = self.d3(torch.cat((self.up3(x), x3), 1))
#         x = self.d2(torch.cat((self.up2(x), x2), 1))
#         x = self.d1(torch.cat((self.up1(x), x1), 1))
#         return self.head(torch.cat((x.real, x.imag), 1))
