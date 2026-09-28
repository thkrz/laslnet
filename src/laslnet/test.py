import numpy as np
import fstpack
import torch
from torch import nn

from dost_patch_model import ComplexDOSTClassifier, selected_voice_maps


def test_selected_maps_against_binding():
    size = 16
    levels = size.bit_length() - 1
    rows, cols = np.indices((size, size))

    impulse = np.zeros((size, size), dtype=np.float32)
    impulse[2, 11] = 1
    signals = (
        np.ones((size, size), dtype=np.float32),
        np.asarray(
            np.cos(2 * np.pi * cols / size) + 0.3 * np.cos(4 * np.pi * rows / size),
            dtype=np.float32,
        ),
        impulse,
        np.random.default_rng(7).standard_normal((size, size)).astype(np.float32),
    )
    orders = [
        (0, 0),
        (4, 0),
        (0, 4),
        (4, 4),  # DC/Nyquist
        (1, 2),
        (-1, 2),
        (2, -1),
        (-3, -2),
    ]
    locations = [(0, 0), (0, 15), (15, 0), (15, 15), (3, 7)]

    for dem in signals:
        maps = selected_voice_maps(dem, orders)
        assert maps.shape == (len(orders), size, size)
        assert maps.dtype == np.complex64

        packed = fstpack.dost(np.asfortranarray(dem.T, dtype=np.complex64))
        for row, col in locations:
            reference = fstpack.local_spectrum(packed, col, row)
            for channel, (px, py) in enumerate(orders):
                # Fortran h index is px+n+1; NumPy index is px+n.
                expected = reference[px + levels - 1, py + levels - 1]
                np.testing.assert_allclose(
                    maps[channel, row, col],
                    expected,
                    rtol=1e-5,
                    atol=1e-5,
                )

    bad = np.ones((size, size), dtype=np.float32)
    bad[0, 0] = -9999
    try:
        selected_voice_maps(bad, [(1, 2)], nodata=-9999)
    except ValueError:
        pass
    else:
        raise AssertionError("No-data patch was accepted")


def test_model_forward_backward():
    model = ComplexDOSTClassifier(n_voices=3)
    x = torch.complex(
        torch.randn(2, 3, 32, 32),
        torch.randn(2, 3, 32, 32),
    )
    logits = model(x)
    assert logits.shape == (2,)
    assert not logits.is_complex()

    loss = nn.BCEWithLogitsLoss()(logits, torch.tensor([0.0, 1.0]))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )


if __name__ == "__main__":
    test_selected_maps_against_binding()
    test_model_forward_backward()
    print("Checks passed")
