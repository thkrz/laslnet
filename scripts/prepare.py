import argparse
import random
import sys
from pathlib import Path

import fiona
import h5py
import numpy as np
import rasterio
from rasterio.features import rasterize

from u import fraction, positive, powersz


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


def balanced(pos, neg, rng):
    n = len(pos)
    if len(neg) < n:
        raise ValueError("not enough empty patches to match annotations")

    coords = pos + rng.sample(neg, n)
    rng.shuffle(coords)
    return coords


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


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="prepare spatially separated, balanced DEM contexts"
    )
    parser.add_argument(
        "-p",
        dest="patch",
        type=positive,
        default=128,
        help="output patch size",
    )
    parser.add_argument(
        "-w",
        dest="window",
        type=powersz,
        default=128,
        help="DOST window size",
    )
    parser.add_argument(
        "-v",
        dest="ratio",
        type=fraction,
        default=0.2,
        help="target validation fraction of annotated patches",
    )
    parser.add_argument("dem", type=Path, help="elevation raster")
    parser.add_argument("annotation", type=Path, help="annotation vectors")
    parser.add_argument("data", type=Path, help="training contexts")
    parser.add_argument("validation", type=Path, help="validation contexts")
    args = parser.parse_args(argv)

    if args.patch % 8:
        parser.error("patch size must be a multiple of 8")
    if args.window < 2:
        parser.error("window size must be at least 2")
    if args.data.resolve() == args.validation.resolve():
        parser.error("training and validation outputs must differ")

    half = args.window // 2
    far = half - 1

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

    write(
        args.data,
        dem,
        ann,
        training,
        args.patch,
        half,
        far,
    )
    write(
        args.validation,
        dem,
        ann,
        validation,
        args.patch,
        half,
        far,
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
        f"training annotated={len(train_pos)} "
        f"empty={len(train_pos)} total={len(training)}"
    )
    print(
        f"validation annotated={len(val_pos)} "
        f"empty={len(val_pos)} total={len(validation)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
