import argparse
import sys
from pathlib import Path

import fstpack
import numpy as np
import rasterio
from rasterio.windows import Window
import torch
from torch import nn
from torch.nn import functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CLeakyReLU(nn.Module):
    def forward(self, z):
        return torch.complex(
            F.leaky_relu(z.real, negative_slope=0.1),
            F.leaky_relu(z.imag, negative_slope=0.1),
        )


class Net(nn.Module):
    def __init__(self, voices):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(voices, 64, 1, dtype=torch.complex64),
            CLeakyReLU(),
            nn.Conv2d(64, 64, 3, padding=1, dtype=torch.complex64),
            CLeakyReLU(),
            nn.Conv2d(64, 64, 3, padding=1, dtype=torch.complex64),
            CLeakyReLU(),
            nn.Conv2d(64, 64, 3, padding=1, dtype=torch.complex64),
            CLeakyReLU(),
        )
        self.head = nn.Conv2d(128, 1, 1)

    def forward(self, x):
        z = self.body(x)
        return self.head(torch.cat((z.real, z.imag), dim=1))


def features(elev):
    v = fstpack.voices(fstpack.dost(elev))
    return v.reshape(1, -1, *v.shape[2:], order="F")


def patchsz(value):
    n = int(value)
    if n <= 1 or n & (n - 1):
        raise argparse.ArgumentTypeError("must be a power of two greater than 1")
    return n


def windows(height, width, n):
    for row in range(0, height, n):
        for col in range(0, width, n):
            h = min(n, height - row)
            w = min(n, width - col)
            yield Window(col, row, w, h)


def filled(a, n):
    a = np.ma.asarray(a)
    valid = ~np.ma.getmaskarray(a)
    if not valid.any():
        return None
    mean = float(a.mean())
    elev = np.full((n, n), mean, np.float32)
    mask = np.zeros((n, n), bool)
    h, w = a.shape
    elev[:h, :w] = a.filled(mean)
    mask[:h, :w] = valid
    return elev, mask


def targets(y, n):
    y = np.ma.asarray(y)
    valid = ~np.ma.getmaskarray(y)
    raw = np.asarray(y.filled(0))
    if ((raw != 0) & (raw != 1) & valid).any():
        raise ValueError("annotations must be 0 or 1")
    target = np.zeros((n, n), np.float32)
    mask = np.zeros((n, n), bool)
    h, w = y.shape
    target[:h, :w] = raw
    mask[:h, :w] = valid
    return target, mask


def same(a, b):
    return (a.crs, a.transform, a.width, a.height) == (
        b.crs,
        b.transform,
        b.width,
        b.height,
    )


def samples(dem_path, ann_path, n):
    with rasterio.open(dem_path) as dem, rasterio.open(ann_path) as ann:
        if not same(dem, ann):
            raise ValueError("DEM and annotations must share a pixel grid")
        for win in windows(dem.height, dem.width, n):
            elev = filled(dem.read(1, window=win, masked=True), n)
            if elev is None:
                continue
            dense, dem_ok = elev
            y, ann_ok = targets(ann.read(1, window=win, masked=True), n)
            m = dem_ok & ann_ok
            if not m.any():
                continue
            yield dense, y, m


def train(args):
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    n = args.n
    voices = (2 * (n.bit_length() - 1)) ** 2
    net = Net(voices).to(device)
    opt = torch.optim.Adam(net.parameters())
    used = 0
    for _ in range(args.epochs):
        for elev, y, m in samples(args.dem, args.annotations, n):
            used += 1
            x = torch.as_tensor(features(elev), device=device)
            y = torch.as_tensor(y, device=device)
            m = torch.as_tensor(m, device=device)
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(net(x)[0, 0][m], y[m])
            loss.backward()
            opt.step()
    if not used:
        raise ValueError("no usable patches")
    state = {k: v.cpu() for k, v in net.state_dict().items()}
    torch.save({"voices": voices, "n": n, "state": state}, args.model)


@torch.inference_mode()
def predict(args):
    ckpt = torch.load(args.model, map_location=device, weights_only=True)
    net = Net(int(ckpt["voices"])).to(device).eval()
    net.load_state_dict(ckpt["state"])
    n = int(ckpt["n"])
    nodata = 255

    with rasterio.open(args.dem) as dem:
        profile = dem.profile.copy()
        profile.update(dtype="uint8", count=1, nodata=nodata)
        with rasterio.open(args.out, "w", **profile) as dst:
            for win in windows(dem.height, dem.width, n):
                tile = np.full((win.height, win.width), nodata, np.uint8)
                elev = filled(dem.read(1, window=win, masked=True), n)
                if elev is not None:
                    dense, valid = elev
                    x = torch.as_tensor(features(dense), device=device)
                    pred = (net(x)[0, 0] > 0).cpu().numpy()
                    h, w = win.height, win.width
                    ok = valid[:h, :w]
                    tile[ok] = pred[:h, :w][ok]
                dst.write(tile, 1, window=win)


def main(argv=None):
    p = argparse.ArgumentParser(description="train and apply a landslide detector")
    p.add_argument(
        "-m",
        "--model",
        metavar="PATH",
        type=Path,
        default=Path("laslnet.pt"),
        help="model checkpoint (default: %(default)s)",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="train and save a model")
    t.add_argument(
        "-e",
        "--epochs",
        metavar="N",
        type=int,
        default=20,
        help="passes over the training raster (default: %(default)s)",
    )
    t.add_argument(
        "-n",
        metavar="N",
        type=patchsz,
        default=128,
        help="patch width and height; power of two (default: %(default)s)",
    )
    t.add_argument("dem", type=Path, help="input elevation raster")
    t.add_argument(
        "annotations",
        type=Path,
        help="0/1 annotation raster on the same pixel grid",
    )
    t.set_defaults(func=train)

    r = sub.add_parser("predict", help="predict and write a raster mask")
    r.add_argument("dem", type=Path, help="input elevation raster")
    r.add_argument(
        "out",
        type=Path,
        help="output in DEM format; uint8 values 0, 1, nodata 255",
    )
    r.set_defaults(func=predict)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
