import random

import fiona
import h5py
import numpy as np
import rasterio
import torch
from rasterio.features import rasterize

from laslnet.dost import embed, margin


def shape(path):
    with h5py.File(path, "r") as f:
        window = int(f.attrs["w"])
        patch = int(f.attrs["p"])
        halo = int(f.attrs["h"])
    voices = (2 * (window.bit_length() - 1)) ** 2
    return voices, patch, window, halo


def samples(path, split, batch, device, shuffle=False):
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
def scale(path, device):
    voices, _, window, _ = shape(path)
    energy = torch.zeros(voices, dtype=torch.float64, device=device)
    count = 0

    for x, _ in samples(path, "train", 1, device):
        energy += x.abs().square().sum((0, 2, 3), dtype=torch.float64)
        count += x.shape[0] * x.shape[2] * x.shape[3]

    sc = (energy / count).sqrt().float()
    sc[sc < 1e-6] = 1
    side = 2 * (window.bit_length() - 1)
    dc = side // 2 - 1
    sc[dc * side + dc] = 1
    return sc


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
