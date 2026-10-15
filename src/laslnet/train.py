import sys

import torch
from torch.nn import functional as F

from laslnet.data import posweight, samples, scale, shape
from laslnet.model import Net, core, save


@torch.no_grad()
def validate(net, path, batch, halo, device, weight):
    net.eval()
    total = torch.zeros((), device=device)
    count = 0
    intersection = 0
    union = 0

    for x, y in samples(path, "val", batch, device):
        logits = core(net(x)[:, 0], halo)
        loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=weight)

        size = y.shape[0]
        total += loss * size
        count += size

        pred = logits >= 0
        truth = y != 0
        intersection += (pred & truth).sum().item()
        union += (pred | truth).sum().item()

    if union == 0:
        raise ValueError("validation IoU undefined: empty union")

    return total.item() / count, intersection / union


def train(args, device):
    parser = args.parser
    voices, patch, window, halo = shape(args.context)
    if patch < 16 or patch % 8:
        parser.error("patch size must be >= 16 and a multiple of 8")
    if halo < 0 or halo % 8:
        parser.error("halo must be a nonnegative multiple of 8")

    net = Net(voices).to(device)
    net.scale.copy_(scale(args.context, device))
    weight = torch.tensor(posweight(args.context), device=device)
    opt = torch.optim.Adam(net.parameters(), lr=0.0003)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="min",
        factor=0.5,
        patience=10,
    )
    best = -1.0

    for epoch in range(args.epochs):
        net.train()
        total = torch.zeros((), device=device)
        count = 0

        for x, y in samples(args.context, "train", args.batch, device, shuffle=True):
            opt.zero_grad(set_to_none=True)
            logits = core(net(x)[:, 0], halo)
            loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=weight)
            loss.backward()
            opt.step()

            size = y.shape[0]
            total += loss.detach() * size
            count += size

        train_loss = total.item() / count
        val_loss, iou = validate(net, args.context, args.batch, halo, device, weight)
        improved = iou > best

        if improved:
            save(args.model, net, voices, patch, window, halo)
            best = iou

        print(
            f"epoch={epoch + 1} loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} val_iou={iou:.6f} "
            f"best_iou={best:.6f} lr={opt.param_groups[0]['lr']:.6g} "
            f"saved={int(improved)}",
            file=sys.stderr,
            flush=True,
        )

        scheduler.step(val_loss)

    return 0
