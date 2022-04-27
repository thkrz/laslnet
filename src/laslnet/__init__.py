import configparser
import numpy as np
from sklearn.metrics import classification_report
import torch
import sys

dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
import torch.nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from . import dataset
from .model import LaslNet

config = configparser.ConfigParser()


def preprocess(x, y):
    return x.to(dev), y.to(dev)


def train(model, data):
    if config["model"]["optim"] == "ADAM":
        learning_rate = float(config["model.adam.train"]["learning_rate"])
        epoch = int(config["model.adam.train"]["epoch"])
        optim = Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.BCELoss()
    gamma = float(config["model.scheduler"]["gamma"])
    scheduler = ExponentialLR(optim, gamma=gamma)
    batch_size = int(config["model"]["batch_size"])
    train_dl = DataLoader(data, batch_size=batch_size, shuffle=True)
    train_dl = dataset.WrappedDataLoader(train_dl, preprocess)
    for _ in range(epoch):
        for input, target in train_dl:
            optim.zero_grad()
            loss = loss_fn(model(input), target)
            loss.backward()
            optim.step()
        scheduler.step()


def val(model, data):
    batch_size = int(config["model"]["batch_size"])
    val_dl = DataLoader(data, batch_size=batch_size, shuffle=True)
    val_dl = dataset.WrappedDataLoader(val_dl, preprocess)
    y_true = []
    y_pred = []
    with torch.no_grad():
        for input, target in val_dl:
            pred = model(input)
            y_true += target.tolist()
            y_pred += pred.tolist()
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)
    print(classification_report(y_true, y_pred))


def laslnet_main():
    assert len(sys.argv) > 1, "missing argument"
    config.read(sys.argv[-1])

    dataset_path = config["DEFAULT"]["dataset_path"]
    model_path = config["DEFAULT"].get("model_path", "./laslnet45.pth")

    model = LaslNet()
    model.to(dev)
    print(f"Model on {dev}")
    ds = dataset.LASLDataset(dataset_path)
    print(f"Start training with {len(ds.train_data)} records")
    train(model, ds.train_data)
    torch.save(model.state_dict(), model_path)
    val(model, ds.val_data)
    return 0
