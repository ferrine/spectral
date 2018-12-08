import collections
from torch import nn as nn
import itertools
import torch
import torch.nn.functional
import abc
import re


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


class Spectrum(nn.Module, metaclass=abc.ABCMeta):
    """

    Parameters
    ----------
    eps : minimum singular value, None is for learnable
    sigma : maximum singular value, None is for learnable
    """

    formula_shortcuts = {}

    def __init__(self, eps=None, sigma=None, eps_cast=None, sigma_cast=None):
        super().__init__()
        if eps is not None:
            self._eps = eps
            self.eps_cast = eps_cast
        else:
            self._eps = nn.Parameter(torch.tensor(0.0))
            assert eps_cast is None
            self.eps_cast = torch.exp
        if sigma is not None:
            self._sigma = sigma
            self.sigma_cast = sigma_cast
        else:
            self._sigma = nn.Parameter(torch.tensor(0.0))
            self.sigma_cast = torch.exp

    def forward(self, n):
        grid = self.grid(n)
        # there are parametrization issues if some parameters are learnable
        # we have to manually fix them
        if isinstance(self._eps, nn.Parameter) or isinstance(self._sigma, nn.Parameter):
            if isinstance(self._eps, nn.Parameter) and not isinstance(
                self._sigma, nn.Parameter
            ):
                # learnable eps, fixed sigma
                # eps should be lower than sigma but positive
                # the upper bound should ofc be dependent on sigma
                eps = torch.sigmoid(self._eps) * self.sigma
                sigma = self.sigma
            elif not isinstance(self._eps, nn.Parameter) and isinstance(
                self._sigma, nn.Parameter
            ):
                # fixed eps, learnable sigma
                eps, sigma = self.eps, self.sigma
                sigma = eps + sigma
            else:
                # all learnable
                eps = torch.sigmoid(self._eps) * self.sigma
                sigma = self.sigma
        else:
            eps, sigma = self.eps, self.sigma
        return (1 - grid) * eps + grid * sigma

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

    @abc.abstractmethod
    def grid(self, n):
        raise NotImplementedError

    @classmethod
    def from_formula(cls, formula):
        name, formula = formula.split(":", 1)
        subcls = cls.formula_shortcuts[name]
        return subcls.from_formula(formula)

    @classmethod
    def register_for_formula(cls, *names):
        def wrap(subls):
            if not issubclass(subls, cls):
                raise TypeError(
                    subls, "the wrapped object is not a subclass of {}".format(cls)
                )
            for name in names:
                if name in cls.formula_shortcuts:
                    raise ValueError(name, "the name is already taken")
                cls.formula_shortcuts[name] = subls
            return subls

        return wrap


@Spectrum.register_for_formula("c", "curve")
class CurveSpectrum(Spectrum):
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
        super().__init__(eps=eps, eps_cast=eps_cast, sigma=sigma, sigma_cast=sigma_cast)
        if tau is not None:
            self._tau = tau
            self.tau_cast = tau_cast
        else:
            self._tau = nn.Parameter(torch.tensor(0.0))
            assert tau_cast is None
            self.tau_cast = torch.exp

    @property
    def tau(self):
        tau = self._tau
        if self.tau_cast is not None:
            tau = self.tau_cast(tau)
        return tau

    def grid(self, n):
        return torch.linspace(0, 1, n) ** self.tau

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


@Spectrum.register_for_formula("g", "grid")
class GridSpectrum(Spectrum):
    """

    Parameters
    ----------
    n: size of grid
    eps : minimum singular value, None is for learnable
    sigma : maximum singular value, None is for learnable
    sort: whether sort is applied
    """

    def __init__(self, n, *args, sort=True, **kwargs):
        super().__init__(*args, **kwargs)
        # initialize with linear spectrum
        linear = torch.linspace(1e-2, 1 - 1e-2, n)
        odds = linear.exp_().sub_(1).log_()
        self.sort = sort
        self._tau = torch.nn.Parameter(odds)

    def grid(self, n):
        if n != len(self._tau):
            raise ValueError(
                n, "n is not equal to length of grid: {}".format(len(self._tau))
            )
        tau = torch.nn.functional.softplus(self._tau)
        grid = tau / tau.max()
        if self.sort:
            grid = grid.sort()[0]
        return grid

    def extra_repr(self):
        return "{}, sort={}".format(str(len(self._tau)), self.sort)

    @classmethod
    def from_formula(cls, formula):
        sp_formula = formula.split(":")
        if len(sp_formula) > 3:
            n, eps, sigma, sort = sp_formula
            sort = sort.lower() in {"t", "1", "true", "sort"}
        else:
            n, eps, sigma = sp_formula
            sort = False
        n = int(n)
        if eps == "*":
            eps = None
        else:
            eps = float(eps)
        if sigma == "*":
            sigma = None
        else:
            sigma = float(sigma)
        return cls(n, eps, sigma, sort=sort)


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


SPECTRUM_REGEXP = re.compile(
    r"/(?P<begin>(?:-)?\d+)(?:-(?P<end>\d+))?/(?P<spectrum>[\w]+:[\w\d:.\-*+]+)"
)


def parse_spectrums(formulas):
    result = dict()
    result[-1] = dict()
    for begin, end, formula in SPECTRUM_REGEXP.findall(formulas):
        begin = int(begin)
        if begin == -1:
            assert not end, "wrong format"
            assert not result[-1], "duplicate default"
            result[-1] = dict(spectrum=formula.replace("+", "/"))
            continue
        if end:
            end = int(end) + 1
        else:
            end = begin + 1
        for i in range(begin, end):
            assert i not in result, "duplicate {i}".format(i=i)
            result[i] = dict(spectrum=formula.replace("+", "/"))
    return result
