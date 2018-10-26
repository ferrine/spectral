import torch
import torch.nn.functional
from torch import nn as nn
from torch.nn.utils.spectral_norm import SpectralNorm
import spectral


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
        u = getattr(module, self.name + "_ux")
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

    def remove(self, module):
        weight = getattr(module, self.name)
        delattr(module, self.name)
        delattr(module, self.name + "_u")
        delattr(module, self.name + "_orig")
        module.register_parameter(self.name, torch.nn.Parameter(weight))

    def __call__(self, module, inputs):
        if getattr(module, self.name + "_ux") is None:
            delattr(module, self.name + "_ux")
            u = inputs[0][0][None]
            if is_transposed(module):
                u = module.forward(u)
            module.register_buffer(self.name + "_ux", u)  # first item in batch
        if module.training:
            weight, u = self.compute_weight(module)
            setattr(module, self.name, weight)
            setattr(module, self.name + "_ux", u)
        else:
            r_g = getattr(module, self.name + "_orig").requires_grad
            getattr(module, self.name).detach_().requires_grad_(r_g)

    @staticmethod
    def apply(module, name, n_power_iterations, dim, eps):
        fn = ConvSpectralNorm(name, n_power_iterations, None, eps)
        weight = module._parameters[name]

        delattr(module, fn.name)
        setattr(module, fn.name + "_ux", None)
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


def validate_mode(mode):
    return mode in {
        "bug",  # dense power iteration
        "fix",  # conv power iteration
        "bug/fix",  # scaled as fix with alpha as bug
        "fix/bug",  # scaled as bug with alpha as fix
        "learn/bug",  # scaled as bug with learnable alpha
        "learn/fix",  # scaled as fix with learnable alpha
        "",  # do nothing
    } or (  # 0.1/bug, etc
        len(mode.split("/")) == 2
        and spectral.utils.isfloat(mode.split("/")[0])
        and mode.split("/")[1] in {"", "bug", "fix"}
    )


def parse_mode(mode):
    if mode in {
        "bug",  # dense power iteration
        "fix",  # conv power iteration
        "bug/fix",  # scaled as fix with alpha as bug
        "fix/bug",  # scaled as bug with alpha as fix
        "learn/bug",  # scaled as bug with learnable alpha
        "learn/fix",  # scaled as fix with learnable alpha
        "",  # do nothing
    }:
        return None, mode
    else:
        return float(mode.split("/")[0]), mode.split("/")[1]


class SmartSpectralNorm(SpectralNorm):
    def __init__(
        self,
        name="weight",
        n_power_iterations=1,
        dim=1,
        eps=1e-12,
        mode="",
        strict=True,
    ):
        super().__init__(name, n_power_iterations, dim, eps)
        assert validate_mode(mode)
        self.alpha, self.mode = parse_mode(mode)
        self.strict = strict
        self.dense = SpectralNorm(name, n_power_iterations, dim, eps)
        self.conv = ConvSpectralNorm(name, n_power_iterations, None, eps)

    @staticmethod
    def apply(module, name, n_power_iterations, dim, eps, mode="bug", strict=True):
        # conv spectral norm
        fn = SmartSpectralNorm(name, n_power_iterations, dim, eps, mode, strict)
        weight = module._parameters[name]

        delattr(module, fn.name)
        setattr(module, fn.name + "_ux", None)
        module.register_parameter(fn.name + "_orig", weight)

        # regular spectral norm
        weight = module._parameters[name]
        height = weight.size(dim)

        u = normalize(weight.new_empty(height).normal_(0, 1), dim=0, eps=fn.eps)
        delattr(module, fn.name)
        module.register_parameter(fn.name + "_orig", weight)
        # We still need to assign weight back as fn.name because all sorts of
        # things may assume that it exists, e.g., when initializing weights.
        # However, we can't directly assign as it could be an nn.Parameter and
        # gets added as a parameter. Instead, we register weight.data as a
        # buffer, which will cause weight to be included in the state dict
        # and also supports nn.init due to shared storage.
        module.register_buffer(fn.name, weight.data)
        module.register_buffer(fn.name + "_u", u)

        if fn.mode.startswith("learn"):
            alpha = weight.new_empty(()).fill_(1.0)
            module.register_parameter(fn.name + "_alpha", torch.nn.Parameter(alpha))
        module.register_forward_pre_hook(fn)

    def init_ux(self, module, inputs):
        if getattr(module, self.name + "_ux") is None:
            delattr(module, self.name + "_ux")
            u = inputs[0][0][None]
            if is_transposed(module):
                u = module.forward(u)
            module.register_buffer(self.name + "_ux", u)  # first item in batch

    def compute_weight(self, module):
        weight = getattr(module, self.name + "_orig")
        weight_mat = weight
        if self.dim != 0:
            # permute dim to front
            weight_mat = weight_mat.permute(
                self.dim, *[d for d in range(weight_mat.dim()) if d != self.dim]
            )

        weight_dense, u = self.dense.compute_weight(module)
        v = normalize(torch.matmul(weight_mat.t(), u), dim=0, eps=self.eps)
        sigma_dense = torch.dot(u, torch.matmul(weight_mat, v))

        weight_conv, ux = self.conv.compute_weight(module)
        conv_params = dict(
            padding=module.padding,
            stride=module.stride,
            dilation=module.dilation,
            groups=module.groups,
        )
        ux, vx = conv_power_iteration(weight, ux, **conv_params)
        sigma_conv = conv_sigma(weight, ux, vx, **conv_params)
        if self.mode == "fix":
            sigma = sigma_conv
            alpha = self.alpha if self.alpha is not None else 1.0
        elif self.mode == "bug":
            sigma = sigma_dense
            alpha = self.alpha if self.alpha is not None else 1.0
        elif self.mode == "bug/fix":
            alpha = (sigma_conv / sigma_dense).detach()  # alpha as bug
            sigma = sigma_conv  # scaled as fix
        elif self.mode == "fix/bug":
            alpha = (sigma_dense / sigma_conv).detach()  # alpha as fix
            sigma = sigma_dense  # scaled as bug
        elif self.mode == "learn/fix":
            alpha = getattr(module, self.name + "_alpha")
            sigma = sigma_conv
        elif self.mode == "learn/bug":
            alpha = getattr(module, self.name + "_alpha")
            sigma = sigma_dense
        else:  # self.mode == '':
            sigma = 1.0
            alpha = 1.0
        if self.strict:
            sigma = sigma
        else:
            sigma = max(1.0, sigma)
        weight = weight / sigma * alpha
        return weight, u, ux

    def remove(self, module):
        delattr(module, self.name + "_ux")
        super().remove(module)

    def __call__(self, module, inputs):
        if getattr(module, self.name + "_ux") is None:
            self.init_ux(module, inputs)
        if module.training:
            weight, u, ux = self.compute_weight(module)
            setattr(module, self.name, weight)
            setattr(module, self.name + "_ux", ux)
            setattr(module, self.name + "_u", u)
        else:
            r_g = getattr(module, self.name + "_orig").requires_grad
            getattr(module, self.name).detach_().requires_grad_(r_g)
