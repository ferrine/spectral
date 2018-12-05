import torch

class GAN(torch.nn.Module):
    def __init__(self, generator, discriminator):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        if not discriminator.wasserstein:
            if discriminator.logits:
                self.bce = torch.nn.BCEWithLogitsLoss()
            else:
                self.bce = torch.nn.BCELoss()

    @property
    def wasserstein(self):
        return self.discriminator.wasserstein

    def discriminator_loss(self, x_real, x_fake):
        real_discrimination = self.discriminator(x_real)
        fake_discrimination = self.discriminator(x_fake)
        if not self.discriminator.wasserstein:
            real_loss = self.bce(
                real_discrimination, torch.ones_like(real_discrimination)
            )
            fake_loss = self.bce(
                fake_discrimination, torch.zeros_like(fake_discrimination)
            )
            loss = real_loss + fake_loss
        else:
            # change sign here
            fake_loss = fake_discrimination.mean()
            real_loss = -real_discrimination.mean()
            loss = fake_loss + real_loss

        return loss

    def generator_loss(self, x_fake):
        fake_discrimination = self.discriminator(x_fake)
        if not self.discriminator.wasserstein:
            loss = self.bce(fake_discrimination, torch.ones_like(fake_discrimination))
        else:
            loss = -fake_discrimination.mean()
        return loss

    def device(self):
        return next(self.parameters()).device
