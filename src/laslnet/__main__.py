import rasterio
from rasterio.windows import Window


def patches(dem_path, label_path, start=0, stop=None):
    with rasterio.open(dem_path) as dem, rasterio.open(label_path) as lab:
        grid = (dem.crs, dem.transform, dem.width, dem.height)
        label_grid = (lab.crs, lab.transform, lab.width, lab.height)
        if grid != label_grid:
            raise ValueError("DEM and labels must have the same pixel grid")

        end = dem.height if stop is None else stop
        for row in range(start, end - 255, 256):
            for col in range(0, dem.width - 255, 256):
                win = Window(col, row, 256, 256)
                a = dem.read(1, window=win, masked=True)
                y = lab.read(1, window=win, masked=True)

                if a.mask.any() or y.mask.any():
                    continue

                yield a.data.astype("float32"), y.data.astype("int64")
