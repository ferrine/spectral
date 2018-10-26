import torch.nn as nn
import spectral


class BaseDiscriminator(nn.Module):
    def __init__(
        self,
        input_shape,
        wasserstein=False,
        logits=True,
        spectral_norm_kwargs=None,
        **kwargs
    ):
        super().__init__()
        self.input_shape = self.image_shape = input_shape
        self.wasserstein = wasserstein
        self.spectral_norm_kwargs = spectral_norm_kwargs or dict()
        self.logits = logits

    @property
    def batch_norm(self):
        return not (self.wasserstein or self.spectral_norm)

    @property
    def spectral_norm(self):
        return bool(self.spectral_norm_kwargs.get("mode", ""))


class BaseImageDiscriminator(BaseDiscriminator):
    def __init__(self, input_shape, fc_in=128, wasserstein=False, **kwargs):
        super().__init__(input_shape, wasserstein=wasserstein, **kwargs)
        self.fc_in = fc_in

    def forward(self, *image):
        image, = image
        x = self.conv(image)
        x = x.view(-1, self.fc_in)
        assert x.shape[0] == image.shape[0]
        return self.dense(x)


class DCV2ImageDiscriminator(BaseImageDiscriminator):
    def __init__(self, input_shape, fc_in=128, wasserstein=False, init=None, **kwargs):
        fc_in = fc_in * 4
        c, h, w = input_shape
        assert (h % 32) == 0, "Need image size as 32*k x 32*k"
        assert (w % 32) == 0, "Need image size as 32*k x 32*k"
        h, w = spectral.nets.conv_arithmetic(
            (h, w), kernel=3, stride=1, pad=1, out_pad=0
        )
        h, w = spectral.nets.conv_arithmetic(
            (h, w), kernel=4, stride=2, pad=1, out_pad=0
        )

        h, w = spectral.nets.conv_arithmetic(
            (h, w), kernel=3, stride=1, pad=1, out_pad=0
        )
        h, w = spectral.nets.conv_arithmetic(
            (h, w), kernel=4, stride=2, pad=1, out_pad=0
        )

        h, w = spectral.nets.conv_arithmetic(
            (h, w), kernel=3, stride=1, pad=1, out_pad=0
        )
        h, w = spectral.nets.conv_arithmetic(
            (h, w), kernel=4, stride=2, pad=1, out_pad=0
        )

        h, w = spectral.nets.conv_arithmetic(
            (h, w), kernel=3, stride=1, pad=1, out_pad=0
        )

        super().__init__(
            input_shape, fc_in=fc_in * h * w, wasserstein=wasserstein, **kwargs
        )

        self.conv = nn.Sequential(
            self.conv_3x3_4x4s2(
                c, fc_in // 8, fc_in // 4, bn=[False, "auto"], init=init
            ),
            self.conv_3x3_4x4s2(fc_in // 4, fc_in // 4, fc_in // 2, init=init),
            self.conv_3x3_4x4s2(fc_in // 2, fc_in // 2, fc_in // 1, init=init),
            spectral.nets.conv2d(
                in_channels=fc_in // 1,
                out_channels=fc_in // 1,
                spectral_norm=self.spectral_norm,
                spectral_norm_fix=self.spectral_norm_fix,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=not self.batch_norm,
                init=init or "normal",
                nonlin="leaky_relu=.2",
            ),
            nn.BatchNorm2d(fc_in) if self.batch_norm else nn.Sequential(),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.dense = nn.Sequential(
            spectral.nets.linear(
                spectral_norm=self.spectral_norm,
                in_features=self.fc_in,
                out_features=1,
                init=init or "normal",
                bias=not wasserstein,
            ),
            nn.Sigmoid() if not (wasserstein or self.logits) else nn.Sequential(),
        )

    def conv_3x3_4x4s2(self, *channels, init=None, bn=("auto", "auto")):
        """
        (3x3 pad 1 stride 1) -> (4x4 pad 1 stride 2)
        """
        assert len(channels) == 3
        bn_1 = self.batch_norm if bn[0] == "auto" else bn[0]
        bn_2 = self.batch_norm if bn[1] == "auto" else bn[1]
        return nn.Sequential(
            spectral.nets.conv2d(
                in_channels=channels[0],
                out_channels=channels[1],
                spectral_norm=self.spectral_norm,
                spectral_norm_fix=self.spectral_norm_fix,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=not bn_1,
                init=init or "normal",
                nonlin="leaky_relu=.2",
            ),
            nn.BatchNorm2d(channels[1]) if bn_1 else nn.Sequential(),
            nn.LeakyReLU(0.2, inplace=True),
            spectral.nets.conv2d(
                in_channels=channels[1],
                out_channels=channels[2],
                spectral_norm=self.spectral_norm,
                spectral_norm_fix=self.spectral_norm_fix,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=not bn_2,
                init=init or "normal",
                nonlin="leaky_relu=.2",
            ),
            nn.BatchNorm2d(channels[2]) if bn_2 else nn.Sequential(),
            nn.LeakyReLU(0.2, inplace=True),
        )
