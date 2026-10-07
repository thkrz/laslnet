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
    stride = patch + half + far
    pos = []
    neg = []

    for row in range(half, height - patch - far + 1, stride):
        for col in range(half, width - patch - far + 1, stride):
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


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="prepare balanced training and validation contexts"
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
        help="validation fraction of annotated patches (default: 0.2)",
    )
    parser.add_argument("dem", type=Path, help="elevation raster")
    parser.add_argument("annotation", type=Path, help="annotation vectors")
    parser.add_argument("data", type=Path, help="training contexts")
    parser.add_argument("validation", type=Path, help="validation contexts")
    args = parser.parse_args(argv)

    if args.patch % 8:
        parser.error("patch size must be a multiple of 8")
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
    n = len(pos)
    valid = int(n * args.ratio)

    if not 0 < valid < n:
        parser.error(
            "ratio and annotated patch count must leave "
            "at least one annotated patch in each set"
        )
    if len(neg) < n:
        parser.error("not enough empty patches to match all annotated patches")

    rng = random.Random(0)
    rng.shuffle(pos)
    neg = rng.sample(neg, n)

    validation = pos[:valid] + neg[:valid]
    training = pos[valid:] + neg[valid:]
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

    print(f"training annotated={n - valid} empty={n - valid} total={len(training)}")
    print(f"validation annotated={valid} empty={valid} total={len(validation)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
