import torch.nn as nn
from torch import Tensor
import xyston.nn as xy


class LaslNet(nn.Module):
    def __init__(self):
        super(LaslNet, self).__init__()
        self._forward = nn.Sequential(
            xy.CConv4d(1, 8, 1, stride=2),
            xy.CReLU(),
            xy.CAvgPool4d(3, stride=2),
            xy.CConv4d(8, 8, 1, stride=2),
            xy.CReLU(),
            xy.CConv4d(8, 16, 1, stride=2),
            xy.CReLU(),
            xy.CConv4d(16, 32, 1, stride=2),
            xy.CReLU(),
            xy.Modulus(),
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=32, out_features=1),
            nn.Sigmoid(),
        )

    def forward(self, input: Tensor) -> Tensor:
        x = self._forward(input)
        return x[:, 0]
