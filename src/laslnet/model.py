import torch.nn as nn
from torch import Tensor
import xyston.nn as xy


class LaslNet(nn.Module):
    def __init__(self):
        super(LaslNet, self).__init__()
        self._forward = nn.Sequential(
            xy.CConv4d(1, 4, 3),
            xy.CReLU(),
            xy.Modulus(),
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=121212, out_features=1),
            nn.Sigmoid(),
        )

    def forward(self, input: Tensor) -> Tensor:
        x = self._forward(input)
        return x[:, 0]
