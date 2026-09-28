import fstpack
import torch
from torch import nn
from torch.nn import functional as F


class CReLU(nn.Module):
    def forward(self, z):
        return torch.complex(F.relu(z.real), F.relu(z.imag))


class Net(nn.Module):
    def __init__(self, voices, classes):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(voices, 64, 1, dtype=torch.complex64),
            CReLU(),
            nn.Conv2d(64, 64, 3, padding=1, dtype=torch.complex64),
            CReLU(),
            nn.Conv2d(64, 64, 3, padding=1, dtype=torch.complex64),
            CReLU(),
            nn.Conv2d(64, 64, 3, padding=1, dtype=torch.complex64),
            CReLU(),
        )
        self.head = nn.Conv2d(128, classes, 1)

    def forward(self, x):
        z = self.body(x)
        return self.head(torch.cat((z.real, z.imag), dim=1))


def features(a):
    v = fstpack.voices(fstpack.dost(a))
    return v.reshape(1, -1, *v.shape[2:], order="F")


def train(data, val, classes, epochs=50, path="model.pt", device="cuda"):
    net = None
    best = float("inf")
    stale = 0

    for epoch in range(epochs):
        for a, labels in data():
            x = torch.as_tensor(features(a), device=device)
            y = torch.as_tensor(labels, device=device).long()[None]

            if net is None:
                voices = x.shape[1]
                net = Net(voices, classes).to(device)
                opt = torch.optim.Adam(net.parameters())

            net.train()
            opt.zero_grad()
            loss = F.cross_entropy(net(x), y)
            loss.backward()
            opt.step()

        net.eval()
        total = 0.0
        pixels = 0

        with torch.inference_mode():
            for a, labels in val():
                x = torch.as_tensor(features(a), device=device)
                y = torch.as_tensor(labels, device=device).long()[None]
                total += F.cross_entropy(net(x), y, reduction="sum").item()
                pixels += y.numel()

        score = total / pixels
        if score < best:
            best = score
            best_epoch = epoch + 1
            stale = 0
            torch.save(
                {
                    "voices": voices,
                    "classes": classes,
                    "weights": net.state_dict(),
                },
                path,
            )
        else:
            stale += 1
            if stale == 5:
                break

    return best_epoch


def predict(a, path="model.pt", device="cuda"):
    saved = torch.load(path, map_location="cpu", weights_only=True)

    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("patch must be square")

    k = a.shape[0]
    if k < 2 or k & (k - 1) or (2 * (k.bit_length() - 1)) ** 2 != saved["voices"]:
        raise ValueError("patch size does not match model voices")

    net = Net(saved["voices"], saved["classes"])
    net.load_state_dict(saved["weights"])
    net.to(device).eval()

    with torch.inference_mode():
        x = torch.as_tensor(features(a), device=device)
        return net(x).argmax(dim=1)[0].cpu().numpy()
