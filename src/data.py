import os

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms.v2 as T

CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)


def normalize(images_uint8):
    """(N, H, W, 3) uint8 array -> (N, 3, H, W) normalized float tensor."""
    x = torch.tensor(images_uint8).permute(0, 3, 1, 2).float() / 255
    return (x - MEAN) / STD


def stratified_split(labels, val_per_class, seed=0):
    g = torch.Generator().manual_seed(seed)
    train_idx, val_idx = [], []
    for c in range(len(CLASSES)):
        idx = torch.where(labels == c)[0]
        idx = idx[torch.randperm(len(idx), generator=g)]
        val_idx.append(idx[:val_per_class])
        train_idx.append(idx[val_per_class:])
    return torch.cat(train_idx), torch.cat(val_idx)


def to_uint8_tensor(images_uint8):
    return torch.tensor(images_uint8).permute(0, 3, 1, 2).contiguous()


def cifar10(root="data", train=True):
    """The torchvision CIFAR-10 dataset, downloaded to root on first use."""
    missing = not os.path.isdir(os.path.join(root, "cifar-10-batches-py"))
    return torchvision.datasets.CIFAR10(root, train=train, download=missing)


def load_cifar10(root="data", val_size=5000, seed=0, normalized=True):
    """Official CIFAR-10 train set split into train/val, plus the official CIFAR-10 test set.
    normalized=False returns raw uint8 (N, 3, H, W) images, for CPU augmentation pipelines."""
    train, test = cifar10(root, train=True), cifar10(root, train=False)
    prepare = normalize if normalized else to_uint8_tensor

    x, y = prepare(train.data), torch.tensor(train.targets)
    train_idx, val_idx = stratified_split(y, val_size // len(CLASSES), seed)
    return {
        "train": (x[train_idx], y[train_idx]),
        "val": (x[val_idx], y[val_idx]),
        "test": (prepare(test.data), torch.tensor(test.targets)),
    }


def train_indices(root="data", val_size=5000, seed=0):
    """Original CIFAR-10 indices of the images in load_cifar10(...)["train"], in the same order."""
    labels = torch.tensor(cifar10(root, train=True).targets)
    return stratified_split(labels, val_size // len(CLASSES), seed)[0]


class TrivialAugmentCifar(torch.utils.data.Dataset):
    """Teacher training data: reflect-pad crop, flip and TrivialAugment (Müller & Hutter, 2021) on CPU.
    Cutout is applied afterwards on the GPU batch."""

    def __init__(self, images_uint8, labels):
        self.images, self.labels = images_uint8, labels
        self.transform = T.Compose([
            T.RandomCrop(32, padding=4, padding_mode="reflect"),
            T.RandomHorizontalFlip(),
            T.TrivialAugmentWide(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(MEAN.flatten().tolist(), STD.flatten().tolist()),
        ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.transform(self.images[i]), self.labels[i]


def random_translate(x, pad):
    n, _, h, w = x.shape
    padded = F.pad(x, (pad,) * 4, mode="reflect")
    dy, dx = torch.randint(0, 2 * pad + 1, (2, n, 1), device=x.device)
    rows = (dy + torch.arange(h, device=x.device))[:, :, None]
    cols = (dx + torch.arange(w, device=x.device))[:, None, :]
    batch = torch.arange(n, device=x.device)[:, None, None]
    return padded[batch, :, rows, cols].permute(0, 3, 1, 2)


def cutout(x, size):
    n, _, h, w = x.shape
    y0 = torch.randint(0, h - size + 1, (n, 1, 1), device=x.device)
    x0 = torch.randint(0, w - size + 1, (n, 1, 1), device=x.device)
    ys = torch.arange(h, device=x.device).view(1, h, 1)
    xs = torch.arange(w, device=x.device).view(1, 1, w)
    mask = (ys >= y0) & (ys < y0 + size) & (xs >= x0) & (xs < x0 + size)
    return x.masked_fill(mask[:, None], 0)


class RandAugmentCifar(TrivialAugmentCifar):
    """CCT teacher data, following the CCT CIFAR recipe (Hassani et al., 2021): random-resized crop (scale 0.8-1),
    flip, RandAugment (2 ops, magnitude 9; Cubuk et al., 2020) and random erasing (p = 0.25)."""

    def __init__(self, images_uint8, labels):
        super().__init__(images_uint8, labels)
        self.transform = T.Compose([
            T.RandomResizedCrop(32, scale=(0.8, 1.0), antialias=True),
            T.RandomHorizontalFlip(),
            T.RandAugment(num_ops=2, magnitude=9),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(MEAN.flatten().tolist(), STD.flatten().tolist()),
            T.RandomErasing(p=0.25),
        ])


class AutoAugmentCifar(TrivialAugmentCifar):
    """The AutoAugment CIFAR pipeline used by the SAM paper (Cubuk et al., 2019; Foret et al., 2021): the learned
    CIFAR-10 policy, then 4-pixel crop padded with the mean color and flip. Cutout 16 is applied on the GPU batch."""

    def __init__(self, images_uint8, labels):
        super().__init__(images_uint8, labels)
        mean_color = [round(255 * m) for m in MEAN.flatten().tolist()]
        self.transform = T.Compose([
            T.AutoAugment(T.AutoAugmentPolicy.CIFAR10),
            T.RandomCrop(32, padding=4, fill=mean_color),
            T.RandomHorizontalFlip(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(MEAN.flatten().tolist(), STD.flatten().tolist()),
        ])


class CpuAugmentLoader:
    """Yields GPU batches like GpuLoader, but from a CPU-augmented dataset (e.g. TrivialAugmentCifar), plus Cutout.
    `images` (optional) are clean normalized training images, used by AirBench to initialise its whitening layer."""

    def __init__(self, dataset, images, batch_size, cutout=0, workers=8, device="cuda", sample_weights=None):
        sampler = None
        if sample_weights is not None:
            sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(dataset), replacement=True)
        self.loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=sampler is None,
                                                  sampler=sampler, drop_last=True, num_workers=workers,
                                                  pin_memory=True, persistent_workers=True)
        if images is not None:
            self.images = images.to(device, torch.float16).contiguous(memory_format=torch.channels_last)
        self.cutout = cutout
        self.device = device

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        for x, y in self.loader:
            x = x.to(self.device, torch.float16, non_blocking=True).contiguous(memory_format=torch.channels_last)
            if self.cutout:
                x = cutout(x, self.cutout)
            yield x, y.to(self.device, non_blocking=True)


class GpuLoader:
    """Keeps the whole dataset on the GPU and augments batch by batch (much faster than a DataLoader on CIFAR).

    Flipping follows airbench's "alternating flip": each image gets a fixed random flip,
    which is inverted every other epoch, so every image is seen in both orientations.
    With sample_weights, each epoch draws its images with replacement in proportion to the weights.
    """

    def __init__(self, images, labels, batch_size, train=False, translate=0, cutout=0, device="cuda",
                 sample_weights=None):
        self.images = images.to(device, torch.float16).contiguous(memory_format=torch.channels_last)
        self.labels = labels.to(device)
        self.batch_size = batch_size
        self.train = train
        self.translate = translate
        self.cutout = cutout
        self.flip = torch.rand(len(labels), device=device) < 0.5
        self.sample_weights = None if sample_weights is None else sample_weights.to(device)
        self.epoch = 0

    def __len__(self):
        n = len(self.labels)
        return n // self.batch_size if self.train else -(-n // self.batch_size)

    def __iter__(self):
        if self.train and self.sample_weights is not None:
            indices = torch.multinomial(self.sample_weights, len(self.labels), replacement=True)
        else:
            order = torch.randperm if self.train else torch.arange
            indices = order(len(self.labels), device=self.labels.device)
        for i in range(len(self)):
            idx = indices[i * self.batch_size:(i + 1) * self.batch_size]
            x = self.images[idx]
            if self.train:
                flip = self.flip[idx] ^ (self.epoch % 2 == 1)
                x = torch.where(flip.view(-1, 1, 1, 1), x.flip(-1), x)
                if self.translate:
                    x = random_translate(x, self.translate)
                if self.cutout:
                    x = cutout(x, self.cutout)
            yield x, self.labels[idx]
        self.epoch += 1
