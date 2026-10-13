import argparse
import sys
from pathlib import Path

import torch

from laslnet.data import prepare
from laslnet.predict import predict
from laslnet.train import train

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
    t.set_defaults(func=lambda args: train(args, device), parser=t)

    g = sub.add_parser("predict", help="predict landslide mask")
    g.add_argument("model", type=Path, help="trained model")
    g.add_argument("dem", type=Path, help="elevation raster")
    g.add_argument("mask", type=Path, help="output mask")
    g.set_defaults(func=lambda args: predict(args, device), parser=g)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
