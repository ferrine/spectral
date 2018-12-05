import spectral
import torch.nn as nn
import torch


class BaseGenerator(nn.Module):
    def __init__(self, input_shape, output_shape, **kwargs):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape

    def generate(self, n, reparametrize=False):
        device = next(self.parameters()).device
        z = torch.randn(n, *self.input_shape, device=device)
        if not reparametrize:
            with torch.no_grad():
                return self(z)
        else:
            return self(z)


class BaseImageGenerator(BaseGenerator):
    def __init__(
        self, input_shape, output_shape, fc_out, conv_in_h, conv_in_w, **kwargs
    ):
        super().__init__(input_shape, output_shape, **kwargs)
        self.fc_out = fc_out
        self.conv_in_h = conv_in_h
        self.conv_in_w = conv_in_w

    def forward(self, code):
        x = self.dense(code)
        x = x.view(-1, self.fc_out, self.conv_in_h, self.conv_in_w)
        return self.deconv(x)


class DCV2ImageGenerator(BaseImageGenerator):
    def __init__(self, input_shape, output_shape, init=None, **kwargs):
        d, = input_shape
        c, h, w = output_shape
        assert h == w, "Need square image"
        assert (h % 32) == 0, "Need image size as 32*k x 32*k"
        k = h // 32
        kwargs["fc_out"] = 4 * d
        super().__init__(
            input_shape=input_shape,
            output_shape=output_shape,
            conv_in_h=4 * k,
            conv_in_w=4 * k,
            **kwargs
        )
        (h, w), op3 = spectral.nets.reverse_deconv_arithmetic(
            (h, w), kernel=3, stride=1, pad=1
        )
        (h, w), op2 = spectral.nets.reverse_deconv_arithmetic(
            (h, w), kernel=4, stride=2, pad=1
        )
        (h, w), op1 = spectral.nets.reverse_deconv_arithmetic(
            (h, w), kernel=4, stride=2, pad=1
        )
        _, op0 = spectral.nets.reverse_deconv_arithmetic(
            (h, w), kernel=4, stride=2, pad=1
        )
        self.dense = nn.Sequential(
            spectral.nets.linear(
                in_features=d, out_features=4 * 4 * 4 * d * k * k, bias=False
            ),
            nn.BatchNorm1d(4 * 4 * 4 * d * k * k),
            nn.ReLU(True),
        )
        self.deconv = nn.Sequential(
            spectral.nets.conv2d(
                in_channels=4 * d,
                out_channels=2 * d,
                kernel_size=4,
                stride=2,
                padding=1,
                transposed=True,
                bias=False,
                output_padding=op0,
                init=init or "normal",
                nonlin="relu",
            ),
            nn.BatchNorm2d(2 * d),
            nn.ReLU(True),
            spectral.nets.conv2d(
                in_channels=2 * d,
                out_channels=d,
                kernel_size=4,
                stride=2,
                padding=1,
                transposed=True,
                bias=False,
                output_padding=op1,
                init=init or "normal",
                nonlin="relu",
            ),
            nn.BatchNorm2d(d),
            nn.ReLU(True),
            spectral.nets.conv2d(
                in_channels=d,
                out_channels=d // 2,
                kernel_size=4,
                stride=2,
                padding=1,
                transposed=True,
                bias=True,
                output_padding=op2,
                init=init or "normal",
                nonlin="relu",
            ),
            nn.BatchNorm2d(d // 2),
            nn.ReLU(True),
            spectral.nets.conv2d(
                in_channels=d // 2,
                out_channels=c,
                kernel_size=3,
                stride=1,
                padding=1,
                transposed=True,
                bias=True,
                output_padding=op3,
                init=init or "normal",
                nonlin="tanh",
            ),
            nn.Tanh(),
        )
