import math
import time

import torch
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from tqdm import tqdm

from .models import BatchNorm
from .results import save_logs

AIRBENCH_HYP = {
    "epochs": 37,
    "batch_size": 1024,
    "lr": 9.0,              # per 1024 examples, decoupled from momentum (see make_optimizer)
    "momentum": 0.85,
    "weight_decay": 0.012,  # per 1024 examples, decoupled from lr
    "bias_scaler": 64.0,    # BatchNorm biases get a 64x larger lr
    "label_smoothing": 0.2,
    "whiten_bias_epochs": 3,
    "translate": 4,
    "cutout": 12,
    "mixup_alpha": 0.0,     # 0 disables Mixup (typical: 0.2-1.0)
    "cutmix_alpha": 0.0,    # 0 disables CutMix (typical: 1.0); if both are on, one is picked per batch
}

KD_HYP = dict(AIRBENCH_HYP, epochs=150, kd_weight=0.9, kd_temperature=4.0)  # KD defaults from CRD (Tian et al., 2020)


def orthogonalize(g, steps=3, eps=1e-7):
    """Pushes the singular values of the matrix g towards 1 with a quintic Newton-Schulz iteration (airbench94_muon)."""
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.bfloat16()
    x = x / (x.norm() + eps)
    transposed = x.size(0) > x.size(1)
    if transposed:
        x = x.T
    for _ in range(steps):
        A = x @ x.T
        x = a * x + (b * A + c * A @ A) @ x
    return (x.T if transposed else x).to(g.dtype)


class Muon(torch.optim.Optimizer):
    """Muon (Jordan, 2024) as in airbench94_muon: for groups with muon=True (conv filters), the Nesterov momentum
    update of each filter bank, seen as an out_channels x rest matrix, is orthogonalised, and the filters are
    renormalised to a fixed norm before each step. Other groups get Nesterov SGD, identical to torch.optim.SGD."""

    def __init__(self, params, momentum, muon_momentum=0.6):
        super().__init__(params, dict(lr=0.0, momentum=momentum, weight_decay=0.0, muon=False))
        self.muon_momentum = muon_momentum

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            momentum = self.muon_momentum if group["muon"] else group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.add(p, alpha=group["weight_decay"])
                buf = self.state[p].setdefault("momentum_buffer", torch.zeros_like(g))
                buf.mul_(momentum).add_(g)
                g = g.add(buf, alpha=momentum)
                if group["muon"]:
                    p.mul_(len(p) ** 0.5 / p.norm())
                    g = orthogonalize(g.reshape(len(g), -1)).reshape(g.shape)
                p.add_(g, alpha=-group["lr"])


def make_optimizer(model, hyp):
    # airbench parametrisation: lr and wd are expressed per 1024 examples and corrected for
    # Nesterov momentum, so each knob can be tuned independently. The loss is summed, not averaged.
    kilostep_scale = 1024 * (1 + 1 / (1 - hyp["momentum"]))
    lr = hyp["lr"] / kilostep_scale
    wd = hyp["weight_decay"] * hyp["batch_size"] / kilostep_scale
    lr_bias = lr * hyp["bias_scaler"]

    norm_biases = [m.bias for m in model.modules() if isinstance(m, BatchNorm)]
    norm_ids = {id(p) for p in norm_biases}
    others = [p for p in model.parameters() if p.requires_grad and id(p) not in norm_ids]
    groups = [dict(params=norm_biases, lr=lr_bias, weight_decay=wd / lr_bias),
              dict(params=others, lr=lr, weight_decay=wd / lr)]

    optimizer = hyp.get("optimizer", "sgd")
    if optimizer == "adamw":
        trainable = [p for p in model.parameters() if p.requires_grad]
        return torch.optim.AdamW([dict(params=[p for p in trainable if p.ndim > 1]),
                                  dict(params=[p for p in trainable if p.ndim <= 1], weight_decay=0.0)],
                                 lr=hyp["adam_lr"], weight_decay=hyp["adam_weight_decay"])
    if optimizer == "muon":
        # Muon on the conv filters only; BatchNorm biases, whitening bias and head keep the SGD settings above
        groups = [groups[0],
                  dict(params=[p for p in others if p.ndim == 4], lr=hyp["muon_lr"], muon=True),
                  dict(params=[p for p in others if p.ndim != 4], lr=lr, weight_decay=wd / lr)]
        return Muon(groups, momentum=hyp["momentum"])
    return torch.optim.SGD(groups, momentum=hyp["momentum"], nesterov=True)


def warmup_then_linear_decay(total_steps, warmup_frac=0.1):
    warmup = int(total_steps * warmup_frac)

    def factor(step):
        if step < warmup:
            return 0.2 + 0.8 * step / warmup
        return (total_steps - step) / (total_steps - warmup)
    return factor


class Lookahead:
    """Every few steps, pull the weights towards a slow EMA copy and reset them to it."""

    def __init__(self, model):
        self.slow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model, decay):
        for slow, fast in zip(self.slow.values(), model.state_dict().values()):
            if fast.dtype.is_floating_point:
                slow.lerp_(fast, 1 - decay)
                fast.copy_(slow)


@torch.no_grad()
def predict(model, images, tta=False, batch_size=2000):
    """Logits for a tensor of images. tta=True averages over flips and +-1 pixel translations (airbench)."""
    def flip_avg(x):
        return 0.5 * model(x) + 0.5 * model(x.flip(-1))

    def flip_translate_avg(x):
        padded = F.pad(x, (1,) * 4, mode="reflect")
        shifted = 0.5 * flip_avg(padded[:, :, :-2, :-2]) + 0.5 * flip_avg(padded[:, :, 2:, 2:])
        return 0.5 * flip_avg(x) + 0.5 * shifted

    model.eval()
    device = next(model.parameters()).device
    logits = []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for x in images.split(batch_size):
            x = x.to(device, torch.float16).contiguous(memory_format=torch.channels_last)
            logits.append((flip_translate_avg(x) if tta else model(x)).float())
    return torch.cat(logits)


def evaluate(model, images, labels, tta=False):
    """Returns (cross-entropy loss, accuracy)."""
    logits = predict(model, images, tta).cpu()
    labels = labels.cpu()
    return F.cross_entropy(logits, labels).item(), (logits.argmax(1) == labels).float().mean().item()


def accuracy(model, images, labels, tta=False):
    return evaluate(model, images, labels, tta)[1]


def standardize(logits):
    """Z-score over the classes (logit standardization, Sun et al., 2024)."""
    return (logits - logits.mean(1, keepdim=True)) / (logits.std(1, keepdim=True) + 1e-7)


def distillation_loss(student_logits, teacher_logits, temperature, standardize_logits=False):
    """Hinton et al. (2015): KL(teacher || student) on temperature-softened outputs, scaled by T^2.
    Summed over the batch. With standardize_logits, both are z-scored first, so the student matches the teacher's
    logit pattern rather than its logit magnitude."""
    if standardize_logits:
        student_logits, teacher_logits = standardize(student_logits), standardize(teacher_logits)
    t = temperature
    return F.kl_div(F.log_softmax(student_logits / t, dim=1), F.log_softmax(teacher_logits / t, dim=1),
                    log_target=True, reduction="sum") * t ** 2


def mix_batch(x, y, mixup_alpha, cutmix_alpha):
    """Mixup (Zhang et al., 2018) or CutMix (Yun et al., 2019) with a random partner from the same batch.
    If both are enabled, one of the two is picked at random for each batch.
    Returns (mixed images, labels, partner labels, lam) where lam is the weight of the original image."""
    use_cutmix = cutmix_alpha > 0 and (mixup_alpha == 0 or torch.rand(1).item() < 0.5)
    alpha = cutmix_alpha if use_cutmix else mixup_alpha
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    partner = torch.randperm(len(x), device=x.device)

    if not use_cutmix:
        return lam * x + (1 - lam) * x[partner], y, y[partner], lam

    h, w = x.shape[-2:]
    cut_h, cut_w = int(h * math.sqrt(1 - lam)), int(w * math.sqrt(1 - lam))
    cy, cx = torch.randint(h, (1,)).item(), torch.randint(w, (1,)).item()
    y0, y1 = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, h)
    x0, x1 = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, w)
    x = x.clone()
    x[:, :, y0:y1, x0:x1] = x[partner, :, y0:y1, x0:x1]
    lam = 1 - (y1 - y0) * (x1 - x0) / (h * w)
    return x, y, y[partner], lam


def student_loss(logits, y, y_mix, lam, hyp, teacher_logits=None):
    """Summed over the batch: CE on the (possibly mixed) labels, blended with the KD loss when a teacher is given."""
    smoothing = hyp["label_smoothing"]
    loss = lam * F.cross_entropy(logits, y, label_smoothing=smoothing, reduction="sum")
    if lam < 1:
        loss = loss + (1 - lam) * F.cross_entropy(logits, y_mix, label_smoothing=smoothing, reduction="sum")
    if teacher_logits is not None:
        kd = distillation_loss(logits, teacher_logits, hyp["kd_temperature"], hyp.get("kd_logit_std", False))
        loss = (1 - hyp["kd_weight"]) * loss + hyp["kd_weight"] * kd
    return loss


def train(model, train_loader, val_images, val_labels, hyp, teacher=None, log_dir=None):
    """AirBench training, in place. Returns two logs (dicts of lists): one row per epoch and one row per step.
    With log_dir, both logs are written to disk after every epoch.

    With a teacher, the loss is (1 - kd_weight) * CE + kd_weight * KD, where the teacher sees exactly the same
    augmented batch as the student ("consistent teaching", Beyer et al., 2022). With Mixup/CutMix, the teacher
    labels the mixed image itself ("function matching"), while the CE term uses the two mixed labels."""
    mixing = hyp["mixup_alpha"] > 0 or hyp["cutmix_alpha"] > 0
    model.init_whitening(train_loader.images[:5000])
    if teacher is not None:
        teacher.eval()

    optimizer = make_optimizer(model, hyp)
    total_steps = hyp["epochs"] * len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_then_linear_decay(total_steps))
    lookahead = Lookahead(model)
    lookahead_decay = 0.95 ** 5 * (torch.arange(total_steps + 1) / total_steps) ** 3

    history = {k: [] for k in ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "time"]}
    steps = {"step": [], "epoch": [], "train_loss": [], "lr": []}
    step, start = 0, time.time()
    bar = tqdm(total=total_steps, desc="training", unit="step", mininterval=5)
    for epoch in range(hyp["epochs"]):
        model.train()
        model.whiten.bias.requires_grad = epoch < hyp["whiten_bias_epochs"]
        loss_sum, correct, seen, epoch_losses = 0.0, 0, 0, []
        for x, y in train_loader:
            y_mix, lam = y, 1.0
            if mixing:
                x, y, y_mix, lam = mix_batch(x, y, hyp["mixup_alpha"], hyp["cutmix_alpha"])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x).float()
                with torch.no_grad():
                    teacher_logits = teacher(x).float() if teacher is not None else None
            loss = student_loss(logits, y, y_mix, lam, hyp, teacher_logits)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            steps["lr"].append(optimizer.param_groups[1]["lr"])
            optimizer.step()
            scheduler.step()
            step += 1
            steps["step"].append(step)
            steps["epoch"].append(epoch + 1)
            epoch_losses.append(loss.detach() / len(y))
            if step % 5 == 0:
                lookahead.update(model, lookahead_decay[step].item())

            loss_sum += loss.detach()
            preds = logits.argmax(1)
            correct += lam * (preds == y).sum() + (1 - lam) * (preds == y_mix).sum()
            seen += len(y)
            bar.update()
        if step == total_steps:
            lookahead.update(model, decay=1.0)

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(loss_sum.item() / seen)
        history["train_acc"].append(correct.item() / seen)
        val_loss, val_acc = evaluate(model, val_images, val_labels)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(optimizer.param_groups[1]["lr"])
        history["time"].append(time.time() - start)
        steps["train_loss"].extend(torch.stack(epoch_losses).tolist())
        if log_dir is not None:
            save_logs(log_dir, history, steps)
        bar.set_postfix(loss=f"{history['train_loss'][-1]:.3f}", val_acc=f"{val_acc:.4f}")
    bar.close()
    return history, steps


TEACHER_HYP = {
    "optimizer": "sgd",
    "epochs": 200,
    "batch_size": 128,
    "lr": 0.1,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "warmup_epochs": 5,
    "translate": 4,
    "cutout": 16,
    "label_smoothing": 0.0,
    "mixup_alpha": 0.0,
    "cutmix_alpha": 0.0,
    "sam_rho": 0.05,    # 0 disables SAM
    "ema_decay": 0.0,   # 0 disables the exponential moving average of the weights
    "clip_norm": 0.0,   # 0 disables gradient clipping
    "sam_chunks": 1,    # >1: m-sharpness, SAM computed separately on that many sub-batches and averaged
}

# The SAM paper's WRN-28-10 CIFAR-10 recipe (Foret et al., 2021, appendix C.1 / Table 6: 97.9% with AutoAugment):
# batch 256 split into 8 sub-batches of 32 for SAM (their 8 GPUs), lr 0.1, wd 5e-4, rho 0.05, cosine without
# warmup, 200 epochs; AutoAugment + Cutout 16 (AutoAugmentCifar)
SAM_PAPER_HYP = dict(TEACHER_HYP, batch_size=256, warmup_epochs=0, translate=0, sam_chunks=8)

# CCT-7/3x1 teacher: the CCT CIFAR recipe (AdamW, warmup + cosine, RandAugment/random erasing in RandAugmentCifar,
# Mixup/CutMix), without label smoothing since smoothed teachers distill worse (Muller et al., 2019)
CCT_HYP = dict(TEACHER_HYP, optimizer="adamw", epochs=1000, lr=5.5e-4, weight_decay=0.06, warmup_epochs=10,
               translate=0, cutout=0, sam_rho=0.0, mixup_alpha=0.8, cutmix_alpha=1.0)

# ConvMixer-384/16 teacher (patch 1, kernel 9): the official CIFAR recipe (AdamW lr 0.01, wd 0.01, batch 512,
# gradient clipping 1.0, RandAugment + random erasing in RandAugmentCifar)
CONVMIXER_HYP = dict(TEACHER_HYP, optimizer="adamw", epochs=300, batch_size=512, lr=0.01, weight_decay=0.01,
                     warmup_epochs=10, translate=0, cutout=0, sam_rho=0.0, clip_norm=1.0,
                     dim=384, depth=16, kernel_size=9, patch_size=1)

# CCT-7/3x1 with the CCT long-schedule setup (their 5000-epoch config: sinusoidal position embedding, lr 5e-4,
# wd 0.05), 1500 epochs, still without label smoothing
CCT_SINE_HYP = dict(CCT_HYP, epochs=1500, lr=5e-4, weight_decay=0.05, positional="sine")

# ResNet students: standard SGD recipe + the best AirBench setting (KD + Mixup/CutMix, run 07)
# + Cutout 16 (DeVries & Taylor, 2017) and an EMA of the weights (cf. SWA, Izmailov et al., 2018)
RESNET_HYP = dict(TEACHER_HYP, epochs=300, cutout=16, sam_rho=0.0, ema_decay=0.999, mixup_alpha=0.8,
                  cutmix_alpha=1.0, kd_weight=0.9, kd_temperature=4.0)


class SAM:
    """Sharpness-Aware Minimization (Foret et al., 2021): the gradient is taken at w + rho * g / ||g||."""

    def __init__(self, params, rho):
        self.params = [p for p in params if p.requires_grad]
        self.rho = rho
        self.perturbations = []

    @torch.no_grad()
    def perturb(self):
        grad_norm = torch.norm(torch.stack([p.grad.norm() for p in self.params if p.grad is not None]))
        self.perturbations = []
        for p in self.params:
            if p.grad is None:
                continue
            e = p.grad * (self.rho / (grad_norm + 1e-12))
            p.add_(e)
            self.perturbations.append((p, e))

    @torch.no_grad()
    def restore(self):
        for p, e in self.perturbations:
            p.sub_(e)


def warmup_then_cosine(total_steps, warmup_steps):
    def factor(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return factor


def sam_m_sharpness(model, sam, compute_loss, x, y, y_mix, lam, teacher_logits, chunks):
    """m-sharpness (Foret et al., 2021, Sec. 4.1): the batch is split into `chunks` sub-batches, each gets its own SAM
    perturbation and gradient (with BatchNorm statistics of that sub-batch, as on separate GPUs), and the gradients
    are averaged into p.grad. Returns the first-pass loss and logits of the whole batch."""
    params = [p for p in model.parameters() if p.requires_grad]
    total = [torch.zeros_like(p) for p in params]
    losses, all_logits = [], []
    teacher_chunks = teacher_logits.chunk(chunks) if teacher_logits is not None else [None] * chunks
    for xc, yc, ymc, tc in zip(x.chunk(chunks), y.chunk(chunks), y_mix.chunk(chunks), teacher_chunks):
        for p in params:
            p.grad = None
        loss, logits = compute_loss(xc, yc, ymc, lam, tc)
        loss.backward()
        sam.perturb()
        for p in params:
            p.grad = None
        compute_loss(xc, yc, ymc, lam, tc)[0].backward()
        sam.restore()
        for t, p in zip(total, params):
            if p.grad is not None:
                t.add_(p.grad, alpha=1 / chunks)
        losses.append(loss.detach() * len(yc))
        all_logits.append(logits.detach())
    for t, p in zip(total, params):
        p.grad = t
    return torch.stack(losses).sum() / len(y), torch.cat(all_logits)


def train_standard(model, train_loader, val_images, val_labels, hyp, teacher=None, log_dir=None):
    """Standard SGD + warmup/cosine training, used for the teacher and the ResNet students. Optional SAM,
    Mixup/CutMix, distillation (same loss as train()) and EMA of the weights. The loader yields augmented GPU
    batches. Same logs as train(); with EMA, val_* is the EMA model (the one kept at the end) and raw_val_* the
    model being trained."""
    mixing = hyp["mixup_alpha"] > 0 or hyp["cutmix_alpha"] > 0
    if teacher is not None:
        teacher.eval()
    if hyp["optimizer"] == "adamw":
        no_decay = [p for n, p in model.named_parameters() if p.ndim <= 1 or "position" in n]
        decay = [p for n, p in model.named_parameters() if not (p.ndim <= 1 or "position" in n)]
        optimizer = torch.optim.AdamW([dict(params=decay, weight_decay=hyp["weight_decay"]),
                                       dict(params=no_decay, weight_decay=0.0)], lr=hyp["lr"])
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=hyp["lr"], momentum=hyp["momentum"],
                                    weight_decay=hyp["weight_decay"], nesterov=True)
    total_steps = hyp["epochs"] * len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, warmup_then_cosine(total_steps, hyp["warmup_epochs"] * len(train_loader)))
    sam = SAM(model.parameters(), hyp["sam_rho"]) if hyp["sam_rho"] > 0 else None
    ema = None
    if hyp["ema_decay"] > 0:
        ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(hyp["ema_decay"]), use_buffers=True)

    def compute_loss(x, y, y_mix, lam, teacher_logits):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x).float()
        return student_loss(logits, y, y_mix, lam, hyp, teacher_logits) / len(y), logits

    columns = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "time"]
    history = {k: [] for k in columns + (["raw_val_loss", "raw_val_acc"] if ema is not None else [])}
    steps = {"step": [], "epoch": [], "train_loss": [], "lr": []}
    step, start = 0, time.time()
    bar = tqdm(total=total_steps, desc="training", unit="step", mininterval=10)
    for epoch in range(hyp["epochs"]):
        model.train()
        loss_sum, correct, seen, epoch_losses = 0.0, 0, 0, []
        for x, y in train_loader:
            y_mix, lam = y, 1.0
            if mixing:
                x, y, y_mix, lam = mix_batch(x, y, hyp["mixup_alpha"], hyp["cutmix_alpha"])
            teacher_logits = None
            if teacher is not None:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    teacher_logits = teacher(x).float()

            if sam is not None and hyp["sam_chunks"] > 1:
                loss, logits = sam_m_sharpness(model, sam, compute_loss, x, y, y_mix, lam, teacher_logits,
                                               hyp["sam_chunks"])
            else:
                loss, logits = compute_loss(x, y, y_mix, lam, teacher_logits)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if sam is not None:
                    sam.perturb()
                    optimizer.zero_grad(set_to_none=True)
                    compute_loss(x, y, y_mix, lam, teacher_logits)[0].backward()
                    sam.restore()
            if hyp["clip_norm"] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), hyp["clip_norm"])
            steps["lr"].append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()
            if ema is not None:
                ema.update_parameters(model)
            step += 1
            steps["step"].append(step)
            steps["epoch"].append(epoch + 1)
            epoch_losses.append(loss.detach())

            loss_sum += loss.detach() * len(y)
            preds = logits.argmax(1)
            correct += lam * (preds == y).sum() + (1 - lam) * (preds == y_mix).sum()
            seen += len(y)
            bar.update()

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(loss_sum.item() / seen)
        history["train_acc"].append(correct.item() / seen)
        if ema is not None:
            raw_val_loss, raw_val_acc = evaluate(model, val_images, val_labels)
            history["raw_val_loss"].append(raw_val_loss)
            history["raw_val_acc"].append(raw_val_acc)
        val_loss, val_acc = evaluate(model if ema is None else ema.module, val_images, val_labels)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(optimizer.param_groups[0]["lr"])
        history["time"].append(time.time() - start)
        steps["train_loss"].extend(torch.stack(epoch_losses).tolist())
        if log_dir is not None:
            save_logs(log_dir, history, steps)
        bar.set_postfix(loss=f"{history['train_loss'][-1]:.3f}", val_acc=f"{val_acc:.4f}")
    bar.close()
    if ema is not None:
        model.load_state_dict(ema.module.state_dict())
    return history, steps
