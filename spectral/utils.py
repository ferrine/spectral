import collections
from torch import nn as nn
import itertools
import torch


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.Iterable):
            return x
        return tuple(itertools.repeat(x, n))

    return parse


single = _ntuple(1)
pair = _ntuple(2)
triple = _ntuple(3)
quadruple = _ntuple(4)


def to_image_range(tensor):
    return tensor * 0.5 + 0.5


def isfloat(s):
    try:
        float(s)
    except ValueError:
        return False
    return True


class Spectrum(nn.Module):
    """

    Parameters
    ----------
    tau : curvature
        for tau = 0, spectrum is constant and equal to sigma,
        for 0 < tau < 1 it is concave,
        for tau = 1 it is linear
        for tau > 1, convex
    eps : minimum singular value
    sigma : maximum singular value
    """

    def __init__(self, tau, eps, sigma, tau_cast=None, eps_cast=None, sigma_cast=None):
        super().__init__()
        self._tau = tau
        self._eps = eps
        self._sigma = sigma
        self.tau_cast = tau_cast
        self.eps_cast = eps_cast
        self.sigma_cast = sigma_cast

    def forward(self, n):
        grid = torch.linspace(0, 1, n) ** self.tau
        return (1 - grid) * self.eps + grid * self.sigma

    @property
    def tau(self):
        tau = self._tau
        if self.tau_cast is not None:
            tau = self.tau_cast(tau)
        return tau

    @property
    def eps(self):
        eps = self._eps
        if self.tau_cast is not None:
            eps = self.tau_cast(eps)
        return eps

    @property
    def sigma(self):
        sigma = self._sigma
        if self.tau_cast is not None:
            sigma = self.tau_cast(sigma)
        return sigma


def is_conv(mod):
    return isinstance(
        mod,
        (
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.ConvTranspose1d,
            nn.ConvTranspose2d,
            nn.ConvTranspose3d,
        ),
    )
