import os
import glob
import torch.utils.data
from torchvision import datasets, transforms
import numpy as np
import spectral
import itertools


DATA_DEFAULT_SIZE = dict(
    mnist=28,
    omniglot=28,
    fachion_mnist=28,
    svhn=32,
    cifar10=32,
    stl10=64,
    lsun=64,
    scaly=288,
    celeba=64,
)

GREY_DATASETS = {"mnist", "fachion_mnist", "omniglot"}
COLORED_DATASETS = set(DATA_DEFAULT_SIZE) - GREY_DATASETS


def input_shape(dataset, input_size=None):
    dataset = dataset.lower().replace("-", "_")
    if dataset.startswith("test"):
        _, dim, size = dataset.split("_")
        size = input_size or int(size)
        dim = int(dim)
    else:
        size = input_size or DATA_DEFAULT_SIZE[dataset]
        dim = 1 if dataset in GREY_DATASETS else 3
    return dim, size, size


class SingleFolderDataset(torch.utils.data.Dataset):
    """Scaly (texture) dataset."""

    def __init__(self, root, transform=None, loader=datasets.folder.default_loader):
        """
        Args:
            root (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root = root
        self.transform = transform
        self.loader = loader
        self.images = []
        for fname in sorted(glob.glob(os.path.join(self.root, "*"))):
            if datasets.folder.has_file_allowed_extension(
                fname, datasets.folder.IMG_EXTENSIONS
            ):
                self.images.append(fname)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.loader(self.images[idx])

        if self.transform:
            image = self.transform(image)

        return image, 0

    def __repr__(self):
        fmt_str = "Dataset " + self.__class__.__name__ + "\n"
        fmt_str += "    Number of datapoints: {}\n".format(self.__len__())
        fmt_str += "    Root Location: {}\n".format(self.root)
        tmp = "    Transforms (if any): "
        fmt_str += "{0}{1}\n".format(
            tmp, self.transform.__repr__().replace("\n", "\n" + " " * len(tmp))
        )
        return fmt_str


def create_image_transforms(input_size):
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    resize = transforms.Resize((input_size, input_size))

    return transforms.Compose([resize, to_tensor, normalize])


def load_dataset(
    root,
    name,
    dataset_size=None,
    transform=None,
    split="train",
    fraction=1.0,
    batch_size=None,
):
    name = name.lower().replace("-", "_")
    assert split in ("train", "test")

    if name == "mnist":
        data = datasets.MNIST(
            os.path.join(root, "MNIST"),
            train=split == "train",
            download=True,
            transform=transform,
        )
    elif name == "fashion-mnist":
        data = datasets.FashionMNIST(
            os.path.join(root, "FASHION-MNIST"),
            train=split == "train",
            download=True,
            transform=transform,
        )
    elif name == "cifar10":
        data = datasets.CIFAR10(
            os.path.join(root, "CIFAR10"),
            train=split == "train",
            download=True,
            transform=transform,
        )
    elif name == "svhn":
        data = datasets.SVHN(
            os.path.join(root, "SVHN"), split=split, download=True, transform=transform
        )
    elif name == "stl10":
        if split == "train":
            split += "+unlabeled"
        data = datasets.STL10(
            os.path.join(root, "STL10"), split=split, download=True, transform=transform
        )
    elif name == "lsun-bed":
        data = datasets.LSUN(
            os.path.join(root, "LSUN"), classes=["bedroom_train"], transform=transform
        )
    elif name == "omniglot":
        data = datasets.Omniglot(
            os.path.join(root, "OMNIGLOT"),
            background=split == "train",
            download=True,
            transform=transform,
        )
    elif name.startswith("test"):
        _, c, d = name.split("_")
        data = datasets.FakeData(
            size=dataset_size,
            image_size=(int(c), int(d), int(d)),
            num_classes=2,
            transform=transform,
        )
        data.labels = np.random.randint(0, 2, dataset_size)
    elif name == "scaly":
        data = SingleFolderDataset(os.path.join(root, "SCALY"), transform=transform)
    elif name == "celeba":
        center_crop = transforms.CenterCrop(178)
        # deal with non squared celebA
        transform = transforms.Compose([center_crop, transform])
        data = SingleFolderDataset(
            os.path.join(root, "celebA", "img_align_celeba"), transform=transform
        )
    else:
        raise NotImplementedError(name)
    assert 0 < fraction <= 1
    assert dataset_size is None or dataset_size <= len(data)
    if dataset_size is None:
        dataset_size = len(data)
    size = min(int(fraction * len(data)), dataset_size)
    size = max(
        size, batch_size if batch_size is not None else 0, torch.cuda.device_count(), 1
    )
    if size < len(data):
        points = np.random.choice(range(len(data)), replace=False, size=size)
        data = torch.utils.data.Subset(data, points)
    return IgnoreLabelDataset(data)


def dataloader(
    root,
    dataset,
    batch_size,
    input_size=None,
    split="train",
    num_workers=1,
    pin_memory=True,
    shuffle=True,
    dataset_size=None,
    fraction=1.0,
):
    dataset = dataset.lower().replace("-", "_")
    if input_size is None and not dataset.startswith("test"):
        input_size = DATA_DEFAULT_SIZE[dataset]
    elif dataset.startswith("test"):
        _, _, d = dataset.split("_")
        input_size = int(d)
    transform = create_image_transforms(input_size)
    data = load_dataset(
        root=root,
        name=dataset,
        dataset_size=dataset_size,
        transform=transform,
        split=split,
        fraction=fraction,
        batch_size=batch_size,
    )
    return torch.utils.data.DataLoader(
        data,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def dataset_is_texture(name):
    # need some texture datasets from aibek
    return name in ("scaly",)


class IgnoreLabelDataset(torch.utils.data.Dataset):
    def __init__(self, orig):
        self.orig = orig

    def __getitem__(self, index):
        return self.orig[index][0]

    def __len__(self):
        return len(self.orig)


class GeneratorDataset(torch.utils.data.Dataset):
    def __init__(self, generator, n, device="cuda", buffer_size=64, transform=None):
        self.generator = generator.to(device).eval()
        self.buffer = None
        self.counter = itertools.count()
        self.n = n
        if buffer_size < 1:
            raise ValueError("buffer size should be larger than 1")
        self.buffer_size = buffer_size
        self.transform = transform

    def renew_buffer(self):
        with torch.no_grad():
            self.buffer = spectral.utils.to_image_range(
                self.generator.generate(self.buffer_size)
            )
            self.counter = itertools.count()

    def __len__(self):
        return self.n

    def __getitem__(self, item):
        if 0 <= item < self.n:
            if self.buffer is None:
                self.renew_buffer()
            i = next(self.counter)
            if i < self.buffer_size:
                if self.transform is not None:
                    return self.transform(self.buffer[i])
                else:
                    return self.buffer[i]
            else:
                self.renew_buffer()
                return self.__getitem__(item)
        else:
            raise IndexError(item)


class ReconstructionDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        source,
        encoder,
        generator,
        return_pair=False,
        device="cuda",
        transform=None,
    ):
        self.source = source
        self.encoder = encoder.to(device).eval()
        self.generator = generator.to(device).eval()
        self.return_pair = return_pair
        self.transform = transform
        self.normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    def __getitem__(self, item):
        with torch.no_grad():
            device = next(self.encoder.parameters()).device
            image = self.source[item]
            z, _ = self.encoder.encode(
                self.normalize(image.clone())[None, :].to(device)
            )
            recon = self.generator(z)[0]
            recon = spectral.utils.to_image_range(recon)
            if self.transform:
                recon = self.transform(recon)
            if self.return_pair:
                return image.to(device), recon
            else:
                return recon

    def __len__(self):
        return len(self.source)
