import collections
from torch import nn as nn
import itertools


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
