import argparse


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
