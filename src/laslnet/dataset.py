import numpy as np
from pathlib import Path
from torch.utils.data import Dataset

from xyston import signal


def _dost_transform(im):
    return signal.real(signal.DostImage(im))


def _monogenic_transform(im):
    return np.array(signal.MonogenicImage(im))


class _LASLDataset(Dataset):
    def __init__(
        self, annotations_file, img_dir, transform=None, target_transform=None
    ):
        with open(annotations_file) as f:
            self.img_labels = [s.split() for s in f]
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, i):
        img_path = self.img_dir / (self.img_labels[i][1] + ".gz")
        image = np.loadtxt(img_path)
        label = self.img_labels[i][0]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


class LASLDataset:
    tr = {"dost": _dost_transform, "monogenic": _monogenic_transform}

    def __init__(self, base_dir, transform="dost"):
        self.base_dir = Path(base_dir)
        self.transform = self.tr[transform]
        self._load("train")
        self._load("val")
        self._load("test")

    def _load(self, name):
        cwd = self.base_dir / name
        self.__dict__[name + "_data"] = _LASLDataset(
            cwd / "map.txt",
            cwd,
            transform=self.transform,
            target_transform=np.float32,
        )


class WrappedDataLoader:
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            yield (self.func(*b))
