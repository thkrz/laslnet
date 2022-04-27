import torch.nn as nn
import xyston.nn as xy


class LaslNet(nn.Module):
    def __init__(self):
        self._forward = nn.Sequential(
            xy.CConv4d(1, 2, 3),
            xy.CReLU(),
            xy.Modulus(),
        )

    def forward(self, input):
        x = self._forward(input)
        x = nn.Linear(in_features=input.shape[2], out_features=1)
        x = nn.Sigmoid(x)
        return x[:, 0]
