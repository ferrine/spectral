import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional
from torch.nn.utils.spectral_norm import SpectralNorm
import spectral


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


def conv2d(
    *args,
    spectral_norm=False,
    transposed=False,
    init="normal",
    nonlin="relu",
    spectral_norm_fix=False,
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
        if spectral_norm_fix:
            module = spectral.nets.spectral_norm(module)
        else:
            module = nn.utils.spectral_norm(module)
    return module


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


def is_transposed(conv):
    return isinstance(
        conv, (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)
    )


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


class ConvSpectralNorm(SpectralNorm):
    def compute_weight(self, module):
        weight = getattr(module, self.name + "_orig")
        u = getattr(module, self.name + "_u")
        conv_params = dict(
            padding=module.padding,
            stride=module.stride,
            dilation=module.dilation,
            groups=module.groups,
        )
        with torch.no_grad():
            for _ in range(self.n_power_iterations):
                # Spectral norm of weight equals to `u^T W v`, where `u` and `v`
                # are the first left and right singular vectors.
                # This power iteration produces approximations of `u` and `v`.
                u, v = conv_power_iteration(weight, u, **conv_params)

        sigma = conv_sigma(weight, u, v, **conv_params)
        weight = weight / sigma
        return weight, u

    def __call__(self, module, inputs):
        if getattr(module, self.name + "_u") is None:
            delattr(module, self.name + "_u")
            u = inputs[0][0][None]
            if is_transposed(module):
                u = module.forward(u)
            module.register_buffer(self.name + "_u", u)  # first item in batch
        super().__call__(module, inputs)

    @staticmethod
    def apply(module, name, n_power_iterations, dim, eps):
        fn = ConvSpectralNorm(name, n_power_iterations, None, eps)
        weight = module._parameters[name]

        delattr(module, fn.name)
        setattr(module, fn.name + "_u", None)
        module.register_parameter(fn.name + "_orig", weight)
        # We still need to assign weight back as fn.name because all sorts of
        # things may assume that it exists, e.g., when initializing weights.
        # However, we can't directly assign as it could be an nn.Parameter and
        # gets added as a parameter. Instead, we register weight.data as a
        # buffer, which will cause weight to be included in the state dict
        # and also supports nn.init due to shared storage.
        module.register_buffer(fn.name, weight.data)
        module.register_forward_pre_hook(fn)
        return fn


def normalize(input, p=2, dim=1, eps=1e-12):
    if dim is None:
        return input / input.view(-1).norm(p, 0, True).clamp(min=eps).expand_as(input)
    else:
        return torch.nn.functional.normalize(input, p, dim, eps)


def conv_power_iteration(
    kernel, u, stride=1, padding=0, dilation=1, groups=1, normalize_uh=True
):
    convnd, deconvnd = {
        # len of shape
        # 1d
        3: (torch.nn.functional.conv1d, torch.nn.functional.grad.conv1d_input),
        # 2d
        4: (torch.nn.functional.conv2d, torch.nn.functional.grad.conv2d_input),
        # 3d
        5: (torch.nn.functional.conv3d, torch.nn.functional.grad.conv3d_input),
    }[len(kernel.shape)]
    v = convnd(
        u, kernel, stride=stride, padding=padding, dilation=dilation, groups=groups
    )
    v = normalize(v, dim=None)
    u = deconvnd(
        u.size(),
        kernel,
        v,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )
    if normalize_uh:
        u = normalize(u, dim=None)
    return u, v


def conv_sigma(kernel, u, v, stride=1, padding=0, dilation=1, groups=1):
    convnd, deconvnd = {
        # len of shape
        # 1d
        3: (torch.nn.functional.conv1d, torch.nn.functional.grad.conv1d_input),
        # 2d
        4: (torch.nn.functional.conv2d, torch.nn.functional.grad.conv2d_input),
        # 3d
        5: (torch.nn.functional.conv3d, torch.nn.functional.grad.conv3d_input),
    }[len(kernel.shape)]
    uh = deconvnd(
        u.size(),
        kernel,
        v,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )
    return torch.dot(u.view(-1), uh.view(-1))


def conv_power_iteration_sigma(
    kernel,
    u=None,
    u_shape=None,
    iterations=1,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
):
    if (u is None) ^ (u_shape is None):
        if u is not None:
            pass
        else:
            u = kernel.new_empty(u_shape).normal_(0, 1)[None]
            u = normalize(u, dim=None)
    else:
        if (u is not None) and (u_shape is not None):
            assert len(u.shape) == (1,) + u_shape
        else:
            raise ValueError("need one of u or u_shape")
    for _ in range(iterations):
        u, v = conv_power_iteration(
            kernel, u, stride=stride, padding=padding, dilation=dilation, groups=groups
        )
    return conv_sigma(
        kernel, u, v, stride=stride, padding=padding, dilation=dilation, groups=groups
    )


def spectral_norm(module, name="weight", n_power_iterations=1, eps=1e-12, dim=None):
    if is_conv(module):
        ConvSpectralNorm.apply(module, name, n_power_iterations, dim, eps)
    else:
        if dim is None:
            dim = 0
        SpectralNorm.apply(module, name, n_power_iterations, dim, eps)
    return module
