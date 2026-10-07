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


def powersz(value):
    n = positive(value)
    if n < 2 or n & (n - 1):
        raise argparse.ArgumentTypeError("must be a power of two")
    return n


def margin(window):
    half = window // 2
    return half, half - 1


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
    data = data.contiguous()
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


def tiles(bad, ann, patch, half, far):
    height, width = ann.shape
    pos = []
    neg = []

    for row in range(half, height - patch - far + 1, patch):
        for col in range(half, width - patch - far + 1, patch):
            if bad[
                row - half : row + patch + far,
                col - half : col + patch + far,
            ].any():
                continue
            if ann[row : row + patch, col : col + patch].any():
                pos.append((row, col))
            else:
                neg.append((row, col))

    return pos, neg


def partition(coords, axis, cut, patch, half, far):
    training = []
    validation = []

    for coord in coords:
        start = coord[axis] - half
        stop = coord[axis] + patch + far
        if stop <= cut:
            training.append(coord)
        elif start >= cut:
            validation.append(coord)

    return training, validation


def write(path, dem, ann, coords, patch, half, far):
    size = patch + half + far
    count = len(coords)

    with h5py.File(path, "x") as dst:
        dst.attrs["p"] = patch
        dst.attrs["w"] = half + far + 1
        data = dst.create_dataset(
            "DATA",
            shape=(count, size, size),
            dtype=np.float32,
            chunks=(1, size, size),
            compression="gzip",
            shuffle=True,
        )
        mask = dst.create_dataset(
            "MASK",
            shape=(count, patch, patch),
            dtype=np.uint8,
            chunks=(1, patch, patch),
            compression="gzip",
        )
        for i, (row, col) in enumerate(coords):
            data[i] = dem[
                row - half : row + patch + far,
                col - half : col + patch + far,
            ]
            mask[i] = ann[row : row + patch, col : col + patch]


def prepare(args):
    parser = args.parser
    if args.patch % 8:
        parser.error("patch size must be a multiple of 8")
    if args.window < 2:
        parser.error("window size must be at least 2")
    if args.data.resolve() == args.validation.resolve():
        parser.error("training and validation outputs must differ")

    half, far = margin(args.window)

    with rasterio.open(args.dem) as src:
        raster = src.read(1, masked=True)
        dem = np.asarray(raster.data, dtype=np.float32)
        bad = np.ma.getmaskarray(raster) | ~np.isfinite(raster.data)
        ann = annotations(
            args.annotation,
            (src.height, src.width),
            src.transform,
        )

    pos, neg = tiles(bad, ann, args.patch, half, far)
    target = int(len(pos) * args.ratio)
    if not 0 < target < len(pos):
        parser.error(
            "ratio and annotated patch count must leave "
            "at least one annotated patch in each set"
        )

    height, width = ann.shape
    axis = 1 if width >= height else 0
    ordered = sorted(pos, key=lambda coord: coord[axis])
    cut = ordered[len(pos) - target][axis] - half

    train_pos, val_pos = partition(pos, axis, cut, args.patch, half, far)
    train_neg, val_neg = partition(neg, axis, cut, args.patch, half, far)

    if not train_pos:
        parser.error("training region has no eligible annotated patches")
    if not val_pos:
        parser.error("validation region has no eligible annotated patches")
    if len(train_neg) < len(train_pos):
        parser.error("training region has insufficient empty patches for 1:1 balancing")
    if len(val_neg) < len(val_pos):
        parser.error(
            "validation region has insufficient empty patches for 1:1 balancing"
        )

    rng = random.Random(0)
    training = train_pos + rng.sample(train_neg, len(train_pos))
    validation = val_pos + rng.sample(val_neg, len(val_pos))
    rng.shuffle(training)
    rng.shuffle(validation)

    write(args.data, dem, ann, training, args.patch, half, far)
    write(args.validation, dem, ann, validation, args.patch, half, far)

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
        f"training annotated={len(train_pos)} "
        f"empty={len(train_pos)} total={len(training)}"
    )
    print(
        f"validation annotated={len(val_pos)} "
        f"empty={len(val_pos)} total={len(validation)}"
    )
    return 0


def shape(path):
    with h5py.File(path, "r") as f:
        window = int(f.attrs["w"])
        patch = int(f.attrs["p"])
    voices = (2 * (window.bit_length() - 1)) ** 2
    return voices, patch, window


def samples(path, batch):
    with h5py.File(path, "r") as f:
        data = f["DATA"]
        mask = f["MASK"]
        patch = int(f.attrs["p"])
        window = int(f.attrs["w"])
        count = data.shape[0]

        for first in range(0, count, batch):
            last = min(first + batch, count)
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
def validate(net, path, batch):
    net.eval()
    intersection = torch.zeros((), dtype=torch.int64, device=device)
    union = torch.zeros((), dtype=torch.int64, device=device)

    for x, y in samples(path, batch):
        pred = net(x)[:, 0] >= 0
        truth = y != 0
        intersection += (pred & truth).sum()
        union += (pred | truth).sum()

    if union.item() == 0:
        raise ValueError("validation IoU is undefined: empty union")

    return (intersection.double() / union).item()


def save(path, net, voices, patch, window):
    state = {k: v.detach().cpu() for k, v in net.state_dict().items()}
    torch.save(
        {"voices": voices, "n": patch, "w": window, "state": state},
        path,
    )


def window_from_voices(voices):
    side = 1
    while side * side < voices:
        side += 1
    if side * side != voices or side % 2:
        raise ValueError("cannot recover window from voices")
    return 1 << (side // 2)


def load(path):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    voices = int(ckpt["voices"])
    patch = int(ckpt["n"])
    if "w" in ckpt:
        window = int(ckpt["w"])
    else:
        window = window_from_voices(voices)
    side = 2 * (window.bit_length() - 1)
    if side * side != voices:
        raise ValueError("checkpoint voices and window disagree")
    net = Net(voices).to(device)
    net.load_state_dict(ckpt["state"])
    net.eval()
    return net, patch, window


def train(args):
    parser = args.parser
    voices, patch, window = shape(args.data)
    if shape(args.validation) != (voices, patch, window):
        parser.error("training and validation dimensions must match")

    weight = torch.tensor(posweight(args.data), device=device)
    net = Net(voices).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=0.0003)
    best = -1.0

    for epoch in range(args.epochs):
        net.train()
        total = torch.zeros((), device=device)
        count = 0

        for x, y in samples(args.data, args.batch):
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
        iou = validate(net, args.validation, args.batch)
        improved = iou > best

        if improved:
            save(args.model, net, voices, patch, window)
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
def masks(net, dem, bad, patch, window):
    half, far = margin(window)
    height, width = dem.shape
    out = np.full((height, width), 255, np.uint8)
    rows = origins(half, height - patch - far, patch)
    cols = origins(half, width - patch - far, patch)
    done = 0
    skipped = 0

    for row in rows:
        for col in cols:
            if bad[
                row - half : row + patch + far,
                col - half : col + patch + far,
            ].any():
                skipped += 1
                continue
            x = embed(
                dem[
                    row - half : row + patch + far,
                    col - half : col + patch + far,
                ],
                patch,
                window,
                device,
            )
            pred = (net(x.unsqueeze(0))[0, 0] >= 0).cpu().numpy()
            tile = out[row : row + patch, col : col + patch]
            fill = tile == 255
            tile[fill] = pred[fill]
            done += 1

    return out, done, skipped, len(rows) * len(cols)


def predict(args):
    parser = args.parser
    net, patch, window = load(args.model)
    if patch % 8:
        parser.error("checkpoint patch size must be a multiple of 8")

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

    need = patch + window - 1
    if dem.shape[0] < need or dem.shape[1] < need:
        parser.error("raster is smaller than one context")

    mask, done, skipped, total = masks(net, dem, bad, patch, window)
    with rasterio.open(args.mask, "w", **profile) as dst:
        dst.write(mask, 1)

    print(f"patch={patch} window={window} tiles={done} skipped={skipped} total={total}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="laslnet")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser(
        "prepare",
        help="build spatially separated, balanced contexts",
        description="prepare spatially separated, balanced DEM contexts",
    )
    p.add_argument(
        "-p",
        dest="patch",
        type=positive,
        default=128,
        help="output patch size",
    )
    p.add_argument(
        "-w",
        dest="window",
        type=powersz,
        default=128,
        help="DOST window size",
    )
    p.add_argument(
        "-v",
        dest="ratio",
        type=fraction,
        default=0.2,
        help="target validation fraction of annotated patches",
    )
    p.add_argument("dem", type=Path, help="elevation raster")
    p.add_argument("annotation", type=Path, help="annotation vectors")
    p.add_argument("data", type=Path, help="training contexts")
    p.add_argument("validation", type=Path, help="validation contexts")
    p.set_defaults(func=prepare, parser=p)

    t = sub.add_parser(
        "train",
        help="train a landslide detector",
        description="train a landslide detector",
    )
    t.add_argument(
        "-b",
        dest="batch",
        type=positive,
        default=8,
        help="training and validation batch size",
    )
    t.add_argument(
        "-e",
        dest="epochs",
        type=positive,
        default=20,
        help="training epochs",
    )
    t.add_argument("data", type=Path, help="training contexts")
    t.add_argument("validation", type=Path, help="validation contexts")
    t.add_argument("model", type=Path, help="output model")
    t.set_defaults(func=train, parser=t)

    g = sub.add_parser(
        "predict",
        help="write a landslide mask",
        description="predict landslides; 1 where logit >= 0, 255 is nodata",
    )
    g.add_argument("model", type=Path, help="trained model")
    g.add_argument("dem", type=Path, help="elevation raster")
    g.add_argument("mask", type=Path, help="output mask")
    g.set_defaults(func=predict, parser=g)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
