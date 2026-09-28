import numpy as np
import rasterio
from rasterio.windows import Window


def patches(dem_path, label_path, n):
    with rasterio.open(dem_path) as dem, rasterio.open(label_path) as lab:
        if (dem.crs, dem.transform, dem.width, dem.height) != (
            lab.crs,
            lab.transform,
            lab.width,
            lab.height,
        ):
            raise ValueError("DEM and labels must have the same pixel grid")
        for row in range(0, dem.height - n + 1, n):
            for col in range(0, dem.width - n + 1, n):
                win = Window(col, row, n, n)
                a = dem.read(1, window=win, masked=True)
                y = lab.read(1, window=win, masked=True)
                if np.ma.is_masked(a) or np.ma.is_masked(y):
                    continue
                yield a.data.astype("float32"), y.data
