import spectral
import torch


def test_it_works():
    conv = spectral.nets.conv2d(
        10, 12, 3, spectral_norm=True, spectral_norm_kwargs={"mode": "fix/bug"}
    )
    img = torch.randn(1, 10, 32, 32)
    conv(img)
