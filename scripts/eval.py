import argparse
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

from train import Net, device, samples


def fraction(num, den):
    return num / den if den else float("nan")


def main(argv=None):
    parser = argparse.ArgumentParser(description="evaluate a landslide detector")
    parser.add_argument(
        "-t", dest="threshold", type=float, default=0, help="logit threshold"
    )
    parser.add_argument("data", type=Path, help="context file")
    parser.add_argument("model", type=Path, help="trained model")
    args = parser.parse_args(argv)

    ckpt = torch.load(args.model, map_location="cpu", weights_only=True)
    net = Net(ckpt["voices"]).to(device)
    net.load_state_dict(ckpt["state"])
    net.eval()

    total = 0.0
    pixels = 0
    tp = fp = fn = 0

    with torch.inference_mode():
        for x, y in samples(args.data):
            x = torch.as_tensor(x, device=device)
            y = torch.as_tensor(y, dtype=torch.float32, device=device)
            logits = net(x)[0, 0]

            loss = F.binary_cross_entropy_with_logits(logits, y, reduction="sum")
            total += loss.item()
            pixels += y.numel()

            pred = logits >= args.threshold
            truth = y != 0
            tp += (pred & truth).sum().item()
            fp += (pred & ~truth).sum().item()
            fn += (~pred & truth).sum().item()

    if not pixels:
        parser.error("no samples")

    print(
        f"loss={total / pixels:.6f} "
        f"precision={fraction(tp, tp + fp):.6f} "
        f"recall={fraction(tp, tp + fn):.6f} "
        f"iou={fraction(tp, tp + fp + fn):.6f}"
    )
    print(f"pixels={pixels} tp={tp} fp={fp} fn={fn}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
