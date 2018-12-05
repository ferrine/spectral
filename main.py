#!/usr/bin/env python
import os
import sys
import itertools
import csv
import time
import tabulate
# import spectral first
import spectral.inception.score
import spectral
import torch
import torchvision.utils
import scipy.misc
import exman

parser = exman.ExParser(
    root=exman.simpleroot(__file__), default_config_files=["local.cfg"]
)

# set defaults
parser.add_argument("--name", type=str, default="unnamed")
parser.add_argument("--wasserstein", type=bool, default=False)
parser.add_argument("--mode", type=str, default="")
parser.add_argument("--sn_strict", type=bool, default=True)
parser.register_validator(
    lambda p: spectral.norm.validate_mode(p.mode), "Wrong spectral norm parameter"
)
parser.add_argument("--latent", type=int, default=128)
parser.add_argument("--dataset", default="cifar10")
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--data_root", default=os.environ.get("DATA_ROOT", ""))
parser.add_argument("--iters", type=int, default=50000)
parser.add_argument("--save_every", type=int, default=1000)
parser.add_argument("--eval_every", type=int, default=1000)
parser.add_argument("--eval_sample", type=int, default=50000)
parser.add_argument("--log_every", type=int, default=10)
parser.add_argument("--cuda", type=bool, default=True)
parser.add_argument("--lrd", type=float, default=1e-4)
parser.add_argument("--lrg", type=float, default=1e-4)
parser.add_argument("--num_d", type=int, default=3)
parser.add_argument("--corr_reg", type=float, default=0)
parser.add_argument("--orth_reg", type=float, default=0)
parser.add_argument("--d_fc_in_k", type=float, default=1)

class Main(object):
    def __init__(self, args):
        self.loader = spectral.datasets.dataloader(
            args.data_root, args.dataset, batch_size=args.batch_size
        )
        image = self.loader.dataset[0]

        self.args = args
        self.discriminator = spectral.discriminator.DCV2ImageDiscriminator(
            wasserstein=args.wasserstein,
            d_fc_in_k=self.args.d_fc_in_k,
            spectral_norm_kwargs=dict(mode=args.mode, strict=args.sn_strict),
            input_shape=image.shape,
        )
        self.generator = spectral.generator.DCV2ImageGenerator(
            input_shape=(args.latent,), output_shape=image.shape
        )


        def _fakes(reparametrize):
            while True:
                yield self.generator.generate(args.batch_size, reparametrize)

        self.fakes = map(self.to, _fakes(False))
        self.rfakes = map(self.to, _fakes(True))
        self.reals = map(self.to, spectral.datasets.endless(self.loader))

        self.gan = spectral.gan.GAN(
            self.generator, self.discriminator
        )
        self.gan.to(self.device)
        self.opt_d = torch.optim.Adam(
            self.discriminator.parameters(), betas=(0.5, 0.99), lr=self.args.lrd
        )
        self.opt_g = torch.optim.Adam(
            self.generator.parameters(), betas=(0.5, 0.99), lr=self.args.lrg
        )
        self._csv_created = False
        self._csv_fields = None

    def to(self, tensor):
        return tensor.to(self.device)

    @staticmethod
    def numpy(tensor):
        return tensor.detach().cpu().numpy()

    @staticmethod
    def cpu(tensor):
        return tensor.detach().cpu()

    @property
    def root(self):
        return self.args.root

    @property
    def logs(self):
        path = self.root / "logs"
        os.makedirs(path, exist_ok=True)
        return path

    @property
    def models(self):
        path = self.root / "models"
        os.makedirs(path, exist_ok=True)
        return path

    @property
    def images(self):
        path = self.root / "images"
        os.makedirs(path, exist_ok=True)
        return path

    @property
    def device(self):
        return "cuda" if self.args.cuda else "cpu"

    def regularization(self, corr_reg=False, orth_reg=False):
        reg_params = [
            mod.weight
            for mod in self.discriminator.modules()
            if spectral.utils.is_conv(mod)
        ]
        loss = 0
        if corr_reg:
            corr_penalty = (spectral.norm.correlation_regularization(reg_params) * self.args.corr_reg)
            loss += corr_penalty
            corr_penalty = corr_penalty.item()
        else:
            corr_penalty = 0

        if orth_reg:
            orth_penalty = spectral.norm.orthogonality_regularization(reg_params) * self.args.orth_reg
            loss += orth_penalty
            orth_penalsty = orth_penalty.item()
        else:
            orth_penalty = 0
        return loss, corr_penalty, orth_penalty

    def run_iteration(self, info=None):
        orth_reg = self.args.orth_reg > 0
        corr_reg = self.args.corr_reg > 0

        if info is None:
            info = spectral.logging.StreamingMeans()

        t0 = time.time()
        self.gan.train()
        for _ in range(self.args.num_d):
            self.opt_d.zero_grad()
            dloss = self.gan.discriminator_loss(next(self.reals), next(self.fakes))
            dlossitem = dloss.item()

            add_loss, corr_penalty, orth_penalty = self.regularization(corr_reg=corr_reg, orth_reg=orth_reg)
            dloss += add_loss
            dloss.backward()
            self.opt_d.step()
            info.update(
                dloss_total=dlossitem + corr_penalty + orth_penalty,
                dloss=dlossitem,
                corr_penalty=corr_penalty,
                orth_penalty=orth_penalty,
            )

        self.opt_g.zero_grad()
        gloss = self.gan.generator_loss(next(self.rfakes))
        gloss.backward()
        self.opt_g.step()
        info.update(gloss=gloss.item(), duration=time.time() - t0)
        return info

    def save_models(self, it):
        path = os.path.join(self.models, str(it).zfill(8))
        gpath = os.path.join(path, "generator.t7")
        dpath = os.path.join(path, "discriminator.t7")
        os.makedirs(path, exist_ok=True)
        torch.save(self.generator.state_dict(), gpath)
        torch.save(self.discriminator.state_dict(), dpath)

    def evaluate(self, n):
        import spectral.inception.score

        self.gan.eval()
        examples = itertools.chain.from_iterable(self.fakes)
        examples = itertools.islice(examples, n)
        examples = map(spectral.utils.to_image_range, examples)
        examples = map(self.numpy, examples)
        inception_score = spectral.inception.score.calculate_inception_score(examples)
        return inception_score

    def train(self):
        info = spectral.logging.StreamingMeans()
        for i in range(1, self.args.iters + 1):
            info = self.run_iteration(info)

            if i % self.args.log_every == 0:
                iter_info = dict(iter=i)
                iter_info.update(info.to_dict())
                self.log_iteration(iter_info)

            if (
                self.args.dataset.lower() == "cifar10"
                and self.args.eval_every != -1
                and i % self.args.eval_every == 0
            ):
                self.log_evaluate(i)

            if i % self.args.save_every == 0:
                self.save_models(i)
                self.log_pictures(i)


    def log_evaluate(self, it):
        inception_score = self.evaluate(self.args.eval_sample)
        print("inception score : {:.3f} +/-({:.3f})".format(*inception_score))
        with (self.logs / "inception_score.txt").open("a") as f:
            f.write("{:d},{},{}\n".format(it, *inception_score))

    def log_iteration(self, info):
        if not self._csv_created:
            self._csv_fields = list(info.keys())
            with (self.logs / "hist.csv").open("w") as f:
                writer = csv.DictWriter(f, self._csv_fields)
                writer.writeheader()
                writer.writerow(info)
            self._csv_created = True
        else:
            with (self.logs / "hist.csv").open("a") as f:
                writer = csv.DictWriter(f, self._csv_fields)
                writer.writerow(info)
        header = [f"iter ({self.args.iters})"] + list(info.keys())[1:]
        if ((info["iter"] - 1) // self.args.log_every) % 10 != 0:
            table = tabulate.tabulate(
                [list(info.values())], header, floatfmt=".4f", tablefmt="plain"
            )
            table = table.split("\n")[1]
        else:
            table = tabulate.tabulate(
                [list(info.values())], header, floatfmt=".4f", tablefmt="simple"
            )
        print(table)

    def log_pictures(self, it):
        self.gan.eval()
        examples = itertools.chain.from_iterable(self.fakes)
        examples = map(spectral.utils.to_image_range, examples)
        examples = map(self.cpu, examples)
        examples = list(itertools.islice(examples, 100))
        grid = torchvision.utils.make_grid(examples, 10)
        path = self.images / (str(it).zfill(8) + ".png")
        scipy.misc.imsave(path, grid.permute(1, 2, 0))


if __name__ == "__main__":
    args = parser.parse_args()
    with args.safe_experiment:
        main = Main(args)
        sys.stdout = spectral.logging.TeeOutput(sys.stdout, main.logs / "stdout")
        sys.stderr = spectral.logging.TeeOutput(sys.stderr, main.logs / "stderr")
        print(main.gan)
        main.train()
