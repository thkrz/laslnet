import argparse
import sys
from pathlib import Path


def train(args):
    from . import model

    model.train(None)


def main(argv=None):
    p = argparse.ArgumentParser(description="landslide detection")
    sub = p.add_subparsers(dest="COMMAND", required=True)

    t = sub.add_parser("train", help="train the model")
    t.add_argument("FILES", nargs="+", type=Path)
    t.set_defaults(func=train)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
