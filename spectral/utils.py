import collections
from torch import nn as nn
import itertools
import torch
import torch.nn.functional


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
        None is learnable curvature
    eps : minimum singular value, None is for learnable
    sigma : maximum singular value, None is for learnable
    """

    def __init__(
        self,
        tau=None,
        eps=None,
        sigma=None,
        tau_cast=None,
        eps_cast=None,
        sigma_cast=None,
    ):
        super().__init__()
        if tau is not None:
            self._tau = tau
            self.tau_cast = tau_cast
        else:
            self._tau = nn.Parameter(torch.tensor(0))
            assert tau_cast is None
            self.tau_cast = torch.exp
        if eps is not None:
            self._eps = eps
            self.eps_cast = eps_cast
        else:
            self._eps = nn.Parameter(torch.tensor(0))
            assert eps_cast is None
            self.eps_cast = torch.exp
        if sigma is not None:
            self._sigma = sigma
            self.sigma_cast = sigma_cast
        else:
            self._sigma = nn.Parameter(torch.tensor(0))
            self.sigma_cast = torch.exp

    def forward(self, n):
        grid = torch.linspace(0, 1, n) ** self.tau
        # there are parametrization issues if some parameters are learnable
        # we have to manually fix them
        if isinstance(self._eps, nn.Parameter) or isinstance(self._sigma, nn.Parameter):
            if isinstance(self._eps, nn.Parameter) and not isinstance(
                self._sigma, nn.Parameter
            ):
                # learnable eps, fixed sigma
                # eps should be lower than sigma but positive
                # the upper bound should ofc be dependent on sigma
                eps = torch.nn.functional.sigmoid(self._eps) * self.sigma
                sigma = self.sigma
            elif not isinstance(self._eps, nn.Parameter) and isinstance(
                self._sigma, nn.Parameter
            ):
                # fixed eps, learnable sigma
                eps, sigma = self.eps, self.sigma
                sigma = eps + sigma
            else:
                # all learnable
                eps = torch.nn.functional.sigmoid(self._eps) * self.sigma
                sigma = self.sigma
        else:
            eps, sigma = self.eps, self.sigma
        return (1 - grid) * eps + grid * sigma

    @property
    def tau(self):
        tau = self._tau
        if self.tau_cast is not None:
            tau = self.tau_cast(tau)
        return tau

    @property
    def eps(self):
        eps = self._eps
        if self.eps_cast is not None:
            eps = self.eps_cast(eps)
        return eps

    @property
    def sigma(self):
        sigma = self._sigma
        if self.sigma_cast is not None:
            sigma = self.sigma_cast(sigma)
        return sigma

    @classmethod
    def from_formula(cls, formula):
        tau, eps, sigma = formula.split(":")
        if tau == "*":
            tau = None
        else:
            tau = float(tau)
        if eps == "*":
            eps = None
        else:
            eps = float(eps)
        if sigma == "*":
            sigma = None
        else:
            sigma = float(sigma)
        return cls(tau, eps, sigma)

    def extra_repr(self):
        tau = "*" if isinstance(self._tau, nn.Parameter) else str(self.tau)
        eps = "*" if isinstance(self._eps, nn.Parameter) else str(self.eps)
        sigma = "*" if isinstance(self._sigma, nn.Parameter) else str(self.sigma)
        return "{}, {}, {}".format(tau, eps, sigma)


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
