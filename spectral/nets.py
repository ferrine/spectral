import torch.nn as nn
import torch
import numpy as np
import spectral
import spectral.norm
import geoopt


def conv_arithmetic(sin, pad, out_pad, stride, kernel):
    res = []
    for s, p, po, st, k in zip(*np.broadcast_arrays(sin, pad, out_pad, stride, kernel)):
        res.append((s + 2 * p - k - po) // st + 1)
    return tuple(map(int, res))


def deconv_arithmetic(sin, pad, out_pad, stride, kernel):
    res = []
    for s, p, po, st, k in zip(*np.broadcast_arrays(sin, pad, out_pad, stride, kernel)):
        res.append((s - 1) * st - 2 * p + k + po)
    return tuple(map(int, res))


def find_deconv_out_padding(sout, pad, stride, kernel):
    sin = conv_arithmetic(sout, pad, 0, stride, kernel)
    smin = deconv_arithmetic(sin, pad, 0, stride, kernel)
    return tuple(so - sm for so, sm, in zip(sout, smin))


def reverse_deconv_arithmetic(sout, pad, stride, kernel):
    sin = conv_arithmetic(sout, pad, 0, stride, kernel)
    out_pad = find_deconv_out_padding(sout, pad, stride, kernel)
    return sin, out_pad


class _BasicResNetBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
        output_padding=0,
        batch_norm=True,
        spectral_norm=False,
        transposed=False,
        init="xavier",
        spectral_norm_fix=False,
    ):
        assert stride in (1, 2)
        assert not (
            batch_norm and spectral_norm
        ), "BatchNorm violates Lipschitz constraint"
        super().__init__()
        self.transposed = transposed
        self.bn1 = nn.BatchNorm2d(in_channels) if batch_norm else nn.Sequential()
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = conv2d(
            in_channels,
            out_channels,
            spectral_norm=spectral_norm,
            spectral_norm_fix=spectral_norm_fix,
            transposed=transposed,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=(not batch_norm),
            init=init,
            nonlin="relu",
        )

        self.bn2 = nn.BatchNorm2d(out_channels) if batch_norm else nn.Sequential()
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = conv2d(
            out_channels,
            out_channels,
            spectral_norm=spectral_norm,
            spectral_norm_fix=spectral_norm_fix,
            transposed=False,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=(not batch_norm),
            init=init,
            nonlin="relu",
        )
        if batch_norm:
            self.bn1.weight.data.fill_(1)
            self.bn1.bias.data.zero_()
            self.bn2.weight.data.fill_(1)
            self.bn2.bias.data.zero_()
        self.prepare = nn.Sequential(self.bn1, self.relu1)
        self.residual_first = self.conv1
        self.residual = nn.Sequential(self.bn2, self.relu2, self.conv2)
        self.equal_in_out = in_channels == out_channels and stride == 1
        if self.equal_in_out:
            self.bypass = nn.Sequential()
        else:
            self.bypass = conv2d(
                in_channels,
                out_channels,
                spectral_norm=spectral_norm,
                spectral_norm_fix=spectral_norm_fix,
                transposed=transposed,
                kernel_size=1,
                stride=stride,
                padding=0,
                bias=(not batch_norm),
                init=init,
                nonlin="relu",
            )
        if self.transposed:
            self.residual_first.output_padding = spectral.utils.pair(output_padding)
            self.bypass.output_padding = spectral.utils.pair(output_padding)

    def forward(self, x):
        if self.equal_in_out:
            out = self.prepare(x)
            out = self.residual_first(out)
            return self.residual(out) + self.bypass(x)
        else:
            x = self.prepare(x)
            out = self.residual_first(x)
            return self.residual(out) + self.bypass(x)


class _ResNetBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        nb_layers,
        stride,
        output_padding=0,
        batch_norm=True,
        spectral_norm=False,
        transposed=False,
        init="xavier",
        spectral_norm_fix=False,
    ):
        super().__init__()
        layers = []
        for i in range(int(nb_layers)):
            layers.append(
                _BasicResNetBlock(
                    i == 0 and in_channels or out_channels,
                    out_channels,
                    i == 0 and stride or 1,
                    spectral_norm=spectral_norm,
                    spectral_norm_fix=spectral_norm_fix,
                    batch_norm=batch_norm,
                    transposed=i == 0 and transposed or False,
                    output_padding=output_padding,
                    init=init,
                )
            )
        self.module = nn.Sequential(*layers)

    def forward(self, x):
        return self.module(x)


class ResNetBlock(_ResNetBlock):
    def __init__(
        self,
        in_channels,
        out_channels,
        nb_layers,
        stride,
        batch_norm=True,
        spectral_norm=False,
        init="xavier",
        spectral_norm_fix=False,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            nb_layers=nb_layers,
            stride=stride,
            batch_norm=batch_norm,
            spectral_norm=spectral_norm,
            spectral_norm_fix=spectral_norm_fix,
            transposed=False,
            init=init,
        )


class TransposedResNetBlock(_ResNetBlock):
    def __init__(
        self,
        in_channels,
        out_channels,
        nb_layers,
        stride,
        output_padding=0,
        batch_norm=True,
        spectral_norm=False,
        init="xavier",
        spectral_norm_fix=False,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            nb_layers=nb_layers,
            stride=stride,
            batch_norm=batch_norm,
            spectral_norm=spectral_norm,
            spectral_norm_fix=spectral_norm_fix,
            transposed=True,
            output_padding=output_padding,
            init=init,
        )


def resnet_block(*args, transposed=False, **kwargs):
    if transposed:
        return TransposedResNetBlock(*args, **kwargs)
    else:
        return ResNetBlock(*args, **kwargs)


SPECTRAL_NORM_DEFAULTS = dict(
    mode="", strict=True, name="weight", n_power_iterations=1, eps=1e-12, dim=0
)


def regular_conv2d(
    *args,
    spectral_norm=False,
    transposed=False,
    init="normal",
    nonlin="relu",
    spectral_norm_kwargs=None,
    **kwargs
):
    if not transposed:
        module = nn.Conv2d(*args, **kwargs)
    else:
        module = nn.ConvTranspose2d(*args, **kwargs)
    if init == "normal":
        nn.init.normal_(module.weight.data, 0, 0.02)
    elif init == "xavier":
        nonlin, *param = nonlin.split("=")
        if param:
            param = eval(param[0])
        else:
            param = None
        gain = nn.init.calculate_gain(nonlin, param)
        nn.init.xavier_uniform_(module.weight.data, gain=gain)
    else:
        raise NotImplementedError(init)
    if getattr(module, "bias", None) is not None:
        nn.init.zeros_(module.bias.data)
    if spectral_norm:
        sn_kw = SPECTRAL_NORM_DEFAULTS.copy()
        if isinstance(module, nn.ConvTranspose2d):
            sn_kw["dim"] = 1
        if spectral_norm_kwargs is not None:
            sn_kw.update(spectral_norm_kwargs)
        spectral.norm.SmartSpectralNorm.apply(module, **sn_kw)
    return module


def stiefel_conv2d(*args, spectrum=None, **kwargs):
    module = OrthConv2d(*args, **kwargs, spectrum=spectrum)
    return module


def conv2d(
    *args,
    spectral_norm=False,
    transposed=False,
    init="normal",
    nonlin="relu",
    spectral_norm_kwargs=None,
    **kwargs
):
    if spectral_norm and spectral_norm_kwargs.get("spectrum") is not None:
        if not transposed:
            return stiefel_conv2d(
                *args, spectrum=spectral_norm_kwargs.get("spectrum"), **kwargs
            )
        else:
            raise NotImplementedError(
                "Sorry, transposed stiefel conv is not yet implemented"
            )
    else:
        return regular_conv2d(
            *args,
            init=init,
            nonlin=nonlin,
            spectral_norm_kwargs=spectral_norm_kwargs,
            **kwargs
        )


def linear(*args, spectral_norm=False, init="normal", nonlin="relu", **kwargs):
    module = nn.Linear(*args, **kwargs)
    if init == "normal":
        nn.init.normal_(module.weight.data, 0, 0.02)
    elif init == "xavier":
        nonlin, *param = nonlin.split("=")
        if param:
            param = eval(param[0])
        else:
            param = None
        gain = nn.init.calculate_gain(nonlin, param)
        nn.init.xavier_uniform_(module.weight.data, gain=gain)
    else:
        raise NotImplementedError(init)
    if getattr(module, "bias", None) is not None:
        nn.init.zeros_(module.bias.data)
    if spectral_norm:
        module = nn.utils.spectral_norm(module)
    return module


class DataParallelWrap(nn.DataParallel):
    def __getattr__(self, item):
        try:
            return super().__getattr__(item)
        except AttributeError as e:
            try:
                return getattr(super().__getattr__("module"), item)
            except AttributeError:
                raise AttributeError(
                    str(e).replace(
                        self.__class__.__name__,
                        "DataParallel"
                        + super().__getattr__("module").__class__.__name__,
                    )
                ) from e


class OrthConv2d(nn.Conv2d):
    def __init__(self, *args, spectrum=None, **kwargs):
        super().__init__(*args, **kwargs)
        weight = self._parameters.pop("weight")
        self._weight_shape = weight.shape
        self.weight_orig = geoopt.ManifoldParameter(
            weight.data.reshape(weight.shape[0], -1).t(), manifold=geoopt.Stiefel()
        )
        self.weight_orig.proj_()
        if spectrum is None:
            self.spectrum = spectral.utils.GridSpectrum(self._weight_shape[0])
        elif isinstance(spectrum, str):
            self.spectrum = spectral.utils.Spectrum.from_formula(spectrum)
        else:
            raise TypeError(spectrum, "data type not understood")

    @property
    def weight(self):
        return (self.spectrum(self._weight_shape[0])[:, None] * self.weight_orig).t().view(
            *self._weight_shape
        )
