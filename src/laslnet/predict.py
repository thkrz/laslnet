import numpy as np
import rasterio
import torch

from laslnet.dost import embed, margin
from laslnet.model import core, load


def origins(lo, hi, step):
    if hi < lo:
        return []
    xs = list(range(lo, hi + 1, step))
    if xs[-1] != hi:
        xs.append(hi)
    return xs


@torch.no_grad()
def masks(net, dem, bad, patch, window, halo, device):
    half, far = margin(window)
    height, width = dem.shape
    out = np.full((height, width), 255, np.uint8)
    rows = origins(half + halo, height - patch - far - halo, patch)
    cols = origins(half + halo, width - patch - far - halo, patch)
    done = 0
    skipped = 0

    for row in rows:
        for col in cols:
            r0 = row - halo - half
            c0 = col - halo - half
            r1 = row + patch + halo + far
            c1 = col + patch + halo + far
            if bad[r0:r1, c0:c1].any():
                skipped += 1
                continue
            x = embed(dem[r0:r1, c0:c1], patch + 2 * halo, window, device)
            pred = (core(net(x.unsqueeze(0))[0, 0], halo) >= 0).cpu().numpy()
            out[row : row + patch, col : col + patch] = pred
            done += 1

    return out, done, skipped, len(rows) * len(cols)


def predict(args, device):
    parser = args.parser
    net, patch, window, halo = load(args.model, device)
    if patch % 8:
        parser.error("checkpoint patch size must be a multiple of 8")
    if halo < 0 or halo % 8:
        parser.error("checkpoint halo must be a nonnegative multiple of 8")

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

    need = patch + 2 * halo + window - 1
    if dem.shape[0] < need or dem.shape[1] < need:
        parser.error("raster smaller than patch + halo + window")

    mask, done, skipped, total = masks(net, dem, bad, patch, window, halo, device)
    with rasterio.open(args.mask, "w", **profile) as dst:
        dst.write(mask, 1)

    print(
        f"patch={patch} window={window} halo={halo} "
        f"tiles={done} skipped={skipped} total={total}"
    )
    return 0
