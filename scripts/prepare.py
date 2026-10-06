import argparse
import random
import sys
from pathlib import Path

import fiona
import h5py
import numpy as np
import rasterio
from rasterio.features import rasterize


def positive(value):
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return n


def powersz(value):
    n = positive(value)
    if n & (n - 1):
        raise argparse.ArgumentTypeError("must be a power of two")
    return n


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
    parser = argparse.ArgumentParser(description="prepare balanced DEM contexts")
    parser.add_argument(
        "-p", dest="patch", type=positive, default=128, help="output patch size"
    )
    parser.add_argument(
        "-w",
        dest="window",
        type=powersz,
        default=128,
        help="DOST window size",
    )
    parser.add_argument("dem", type=Path, help="elevation raster")
    parser.add_argument("annotation", type=Path, help="annotation vectors")
    parser.add_argument("data", type=Path, help="output contexts")
    args = parser.parse_args(argv)

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
    if not pos:
        parser.error("no eligible annotated patches")

    rng = random.Random(0)
    empty = min(len(pos), len(neg))
    coords = pos + rng.sample(neg, empty)
    rng.shuffle(coords)
    write(args.data, dem, ann, coords, args.patch, half, far)
    print(f"annotated={len(pos)} empty={empty} total={len(coords)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
