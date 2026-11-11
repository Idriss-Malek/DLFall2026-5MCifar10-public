"""Self-supervised pretraining of a teacher backbone on the 45k training images, without labels: SimCLR
(Chen et al., 2020) and LeJEPA (Balestriero & LeCun, 2025). The pretrained backbone is then fine-tuned with labels
(train_standard) into a distillation teacher whose errors should differ from those of supervised teachers
(Gontijo-Lopes et al., 2022)."""
import math
import time

import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as T
from torch import nn
from torchvision.ops import MLP
from tqdm import tqdm

from .data import MEAN, STD
from .results import save_logs
from .train import TEACHER_HYP, warmup_then_cosine

SIMCLR_HYP = {"method": "simclr", "epochs": 500, "batch_size": 512, "views": 2, "lr": 1e-3, "weight_decay": 1e-6,
              "warmup_epochs": 10, "temperature": 0.5, "projector": [2048, 128]}
# LeJEPA minimal recipe (official MINIMAL.md): 4 views, lambda 0.02, 16-d projection; weight decay 5e-4 for ResNets
LEJEPA_HYP = {"method": "lejepa", "epochs": 300, "batch_size": 256, "views": 4, "lr": 2e-3, "weight_decay": 5e-4,
              "warmup_epochs": 10, "lambda": 0.02, "projector": [2048, 2048, 16]}
# fine-tuning the pretrained backbone with labels: the WRN teacher recipe, shorter, lower lr, without SAM
FINETUNE_HYP = dict(TEACHER_HYP, epochs=100, lr=0.05, warmup_epochs=2, sam_rho=0.0)


class MultiViewCifar(torch.utils.data.Dataset):
    """`views` independently augmented copies of each image, with the SimCLR CIFAR augmentation: random-resized crop
    (scale 0.2-1), flip, color jitter (p = 0.8) and grayscale (p = 0.2). No labels."""

    def __init__(self, images_uint8, views):
        self.images, self.views = images_uint8, views
        self.transform = T.Compose([
            T.RandomResizedCrop(32, scale=(0.2, 1.0), antialias=True),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(MEAN.flatten().tolist(), STD.flatten().tolist()),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return torch.stack([self.transform(self.images[i]) for _ in range(self.views)])


class SSLModel(nn.Module):
    """Backbone (its pooled features, without the classification head) followed by an MLP projector."""

    def __init__(self, backbone, feature_dim, projector_dims):
        super().__init__()
        self.backbone = backbone
        self.projector = MLP(feature_dim, projector_dims, norm_layer=nn.BatchNorm1d)

    def forward(self, x):
        return self.projector(self.backbone.features(x))


def nt_xent(z, temperature):
    """SimCLR loss. z: (2N, D), the first N rows are view 1 and the last N view 2 of the same images."""
    z = F.normalize(z, dim=1)
    n = len(z) // 2
    similarity = z @ z.T / temperature
    similarity.fill_diagonal_(float("-inf"))
    positives = torch.cat([torch.arange(n, 2 * n), torch.arange(n)]).to(z.device)
    return F.cross_entropy(similarity, positives)


class SIGReg(nn.Module):
    """Sketched Isotropic Gaussian Regularization (LeJEPA): projects the embeddings on random unit directions and
    compares each 1-D projection with a standard normal through the Epps-Pulley statistic (squared distance between
    characteristic functions, integrated with a Gaussian window over t in [0, 3]). Follows the official MINIMAL.md."""

    def __init__(self, knots=17, slices=256):
        super().__init__()
        self.slices = slices
        t = torch.linspace(0, 3, knots)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        directions = torch.randn(proj.size(-1), self.slices, device=proj.device)
        directions = directions / directions.norm(dim=0)
        x_t = (proj @ directions).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        return ((err @ self.weights) * proj.size(-2)).mean()


def lejepa_loss(proj, sigreg, lam):
    """proj: (views, N, D). Each view's embedding predicts the mean embedding of all views of its image (invariance),
    plus SIGReg on every view. Returns (loss, invariance term, SIGReg term)."""
    invariance = (proj.mean(0) - proj).square().mean()
    regularization = sigreg(proj)
    return lam * regularization + (1 - lam) * invariance, invariance, regularization


@torch.no_grad()
def knn_accuracy(backbone, train_images, train_labels, val_images, val_labels, k=200, temperature=0.1):
    """Weighted k-NN classifier on normalized backbone features (Wu et al., 2018). Tracks the quality of the
    representation during pretraining; labels are used only for this measurement, never for training."""
    def features(images):
        out = []
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for x in images.split(1000):
                x = x.cuda().contiguous(memory_format=torch.channels_last)
                out.append(F.normalize(backbone.features(x).float(), dim=1))
        return torch.cat(out)

    backbone.eval()
    train_features, val_features = features(train_images), features(val_images)
    train_labels = train_labels.cuda()
    correct = 0
    for f, y in zip(val_features.split(500), val_labels.cuda().split(500)):
        similarity, idx = (f @ train_features.T).topk(k, dim=1)
        votes = torch.zeros(len(f), 10, device=f.device)
        votes.scatter_add_(1, train_labels[idx], (similarity / temperature).exp())
        correct += (votes.argmax(1) == y).sum().item()
    return correct / len(val_labels)


def pretrain(model, loader, hyp, knn_data, log_dir, knn_every=50):
    """Self-supervised training of an SSLModel.
    Logs per epoch (pretrain_epochs.csv) and per step (pretrain_steps.csv)."""
    lejepa = hyp["method"] == "lejepa"
    sigreg = SIGReg().cuda() if lejepa else None
    no_decay = [p for p in model.parameters() if p.ndim <= 1]
    decay = [p for p in model.parameters() if p.ndim > 1]
    optimizer = torch.optim.AdamW([dict(params=decay, weight_decay=hyp["weight_decay"]),
                                   dict(params=no_decay, weight_decay=0.0)], lr=hyp["lr"])
    total_steps = hyp["epochs"] * len(loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, warmup_then_cosine(total_steps, hyp["warmup_epochs"] * len(loader)))

    history = {k: [] for k in ["epoch", "loss", "invariance_loss", "sigreg_loss", "lr", "time", "knn_val_acc"]}
    steps = {"step": [], "epoch": [], "loss": [], "lr": []}
    step, start = 0, time.time()
    bar = tqdm(total=total_steps, desc=f"pretraining ({hyp['method']})", unit="step", mininterval=10)
    for epoch in range(hyp["epochs"]):
        model.train()
        sums, epoch_losses = torch.zeros(3, device="cuda"), []
        for views in loader:
            n_views = views.size(1)
            x = views.cuda(non_blocking=True).transpose(0, 1).flatten(0, 1)
            x = x.contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z = model(x).float()
            if lejepa:
                loss, invariance, regularization = lejepa_loss(z.view(n_views, -1, z.size(-1)), sigreg, hyp["lambda"])
                sums += torch.stack([loss.detach(), invariance.detach(), regularization.detach()])
            else:
                loss = nt_xent(z, hyp["temperature"])
                sums[0] += loss.detach()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            steps["lr"].append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()
            step += 1
            steps["step"].append(step)
            steps["epoch"].append(epoch + 1)
            epoch_losses.append(loss.detach())
            bar.update()

        means = (sums / len(loader)).tolist()
        last_epoch = epoch + 1 == hyp["epochs"]
        knn = knn_accuracy(model.backbone, *knn_data) if (epoch + 1) % knn_every == 0 or last_epoch else math.nan
        history["epoch"].append(epoch + 1)
        history["loss"].append(means[0])
        history["invariance_loss"].append(means[1] if lejepa else math.nan)
        history["sigreg_loss"].append(means[2] if lejepa else math.nan)
        history["lr"].append(optimizer.param_groups[0]["lr"])
        history["time"].append(time.time() - start)
        history["knn_val_acc"].append(knn)
        steps["loss"].extend(torch.stack(epoch_losses).tolist())
        save_logs(log_dir, history, steps, prefix="pretrain_")
        bar.set_postfix(loss=f"{means[0]:.4f}", knn_val_acc="-" if math.isnan(knn) else f"{knn:.4f}")
    bar.close()
    return history
