import torch
from torch import nn


class BatchNorm(nn.BatchNorm2d):
    """BatchNorm with a frozen scale (airbench): only the shift is learned."""

    def __init__(self, num_features):
        super().__init__(num_features, eps=1e-12)
        self.weight.requires_grad = False


class Conv(nn.Conv2d):
    """3x3 conv initialised close to identity (Dirac) on the first in_channels filters."""

    def __init__(self, in_channels, out_channels, kernel_size=3, padding="same", bias=False):
        super().__init__(in_channels, out_channels, kernel_size, padding=padding, bias=bias)

    def reset_parameters(self):
        super().reset_parameters()
        if self.bias is not None:
            self.bias.data.zero_()
        nn.init.dirac_(self.weight.data[:self.in_channels])


class ConvGroup(nn.Module):
    """conv -> pool -> BN -> GELU, followed by a residual block of two convs."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = Conv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = BatchNorm(out_channels)
        self.conv2 = Conv(out_channels, out_channels)
        self.norm2 = BatchNorm(out_channels)
        self.conv3 = Conv(out_channels, out_channels)
        self.norm3 = BatchNorm(out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.act(self.norm1(self.pool(self.conv1(x))))
        residual = x
        x = self.act(self.norm2(self.conv2(x)))
        x = self.norm3(self.conv3(x))
        return self.act(x + residual)


class AirBench(nn.Module):
    """airbench96 (Jordan, 2024) with narrower widths to fit under 5M parameters.

    The first layer is a fixed 2x2 "whitening" conv whose filters are the PCA basis of
    training-image patches (see init_whitening); only its bias is trained, for a few epochs.
    """

    def __init__(self, widths=(128, 256, 352), num_classes=10, logit_scale=1 / 9):
        super().__init__()
        whiten_width = 2 * 3 * 2 ** 2
        self.whiten = Conv(3, whiten_width, kernel_size=2, padding=0, bias=True)
        self.whiten.weight.requires_grad = False
        self.layers = nn.Sequential(
            ConvGroup(whiten_width, widths[0]),
            ConvGroup(widths[0], widths[1]),
            ConvGroup(widths[1], widths[2]),
        )
        self.pool = nn.AdaptiveMaxPool2d(1)
        self.head = nn.Linear(widths[2], num_classes, bias=False)
        self.act = nn.GELU()
        self.logit_scale = logit_scale

    @torch.no_grad()
    def init_whitening(self, images, eps=5e-4):
        k = self.whiten.kernel_size[0]
        patches = images.unfold(2, k, 1).unfold(3, k, 1).transpose(1, 3).reshape(-1, 3 * k * k).float()
        eigenvalues, eigenvectors = torch.linalg.eigh(patches.T @ patches / len(patches))
        filters = (eigenvectors.T / torch.sqrt(eigenvalues + eps)[:, None]).reshape(-1, 3, k, k).flip(0)
        self.whiten.weight.copy_(torch.cat([filters, -filters]))

    def forward(self, x):
        x = self.act(self.whiten(x))
        x = self.layers(x)
        x = self.pool(x).flatten(1)
        return self.head(x) * self.logit_scale


class SqueezeExcite(nn.Module):
    """Squeeze-and-Excitation (Hu et al., 2018): per-channel weights in (0, 1) computed from the globally averaged
    feature map by a small bottleneck MLP, used to rescale the channels."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.fc = nn.Sequential(nn.Linear(channels, hidden), nn.ReLU(), nn.Linear(hidden, channels), nn.Sigmoid())

    def forward(self, x):
        return x * self.fc(x.mean((2, 3)))[:, :, None, None]


class PreActBlock(nn.Module):
    """Pre-activation residual block (He et al., 2016): BN -> ReLU -> conv -> BN -> ReLU -> conv, plus shortcut.
    With se=True, the residual branch is rescaled by a Squeeze-and-Excitation module before the addition."""

    def __init__(self, in_channels, out_channels, stride, dropout=0.0, se=False):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.shortcut = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, stride, bias=False)
        self.se = SqueezeExcite(out_channels) if se else None

    def forward(self, x):
        out = torch.relu(self.bn1(x))
        shortcut = x if self.shortcut is None else self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(self.dropout(torch.relu(self.bn2(out))))
        if self.se is not None:
            out = self.se(out)
        return out + shortcut


class PreActResNet(nn.Module):
    """Pre-activation ResNet for 32x32 images: a 3x3 stem, then stages of residual blocks. The first stage keeps the
    full 32x32 resolution, every following stage halves it. Covers both ResNet-18-style (4 stages) and
    Wide-ResNet-style (3 stages) layouts."""

    def __init__(self, widths, blocks_per_stage, stem_width=16, num_classes=10, dropout=0.0, se=False):
        super().__init__()
        self.stem = nn.Conv2d(3, stem_width, 3, padding=1, bias=False)
        layers, in_channels = [], stem_width
        for stage, (channels, n_blocks) in enumerate(zip(widths, blocks_per_stage)):
            for i in range(n_blocks):
                stride = 2 if stage > 0 and i == 0 else 1
                layers.append(PreActBlock(in_channels, channels, stride, dropout, se))
                in_channels = channels
        self.blocks = nn.Sequential(*layers)
        self.bn = nn.BatchNorm2d(in_channels)
        self.head = nn.Linear(in_channels, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def features(self, x):
        x = self.blocks(self.stem(x))
        return torch.relu(self.bn(x)).mean((2, 3))

    def forward(self, x):
        return self.head(self.features(x))


class RIC(nn.Module):
    """Reinforced Iterative Classification (Kallel et al., 2026): a ResNet encoder followed by a GRU that refines a
    belief over the classes, with a Dirichlet policy and a value head. The paper's thought state is as wide as the
    encoding; here it is smaller (with a linear layer in between) to stay under 5M parameters."""

    def __init__(self, encoder, feature_dim, thought_dim=128, num_classes=10, eps=0.01):
        super().__init__()
        encoder.head = nn.Identity()
        self.encoder = encoder
        self.num_classes, self.thought_dim, self.eps = num_classes, thought_dim, eps
        self.encode = nn.Linear(feature_dim, thought_dim)
        self.gru = nn.GRUCell(thought_dim + num_classes, thought_dim)
        self.classifier = nn.Linear(thought_dim, num_classes)
        self.value = nn.Linear(thought_dim, 1)
        self.concentration = nn.Linear(thought_dim, 1)
        nn.init.zeros_(self.concentration.weight)
        nn.init.constant_(self.concentration.bias, 5.5)  # middle of the clipping range

    def features(self, x):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self.encoder.features(x).float()

    def initial_state(self, n, device):
        belief = torch.full((n, self.num_classes), 1 / self.num_classes, device=device)
        return belief, torch.zeros(n, self.thought_dim, device=device)

    def step(self, encoding, action, thought):
        """One refinement step: new thought state, Dirichlet parameters of the next action, value."""
        thought = self.gru(torch.cat([encoding, action], 1), thought)
        c = self.concentration(thought).clamp(1, 10)
        alpha = torch.softmax(self.classifier(thought), 1) * c + self.eps
        return thought, alpha, self.value(thought).squeeze(1)

    def trajectory(self, features, steps=20):
        """Deterministic refinement (each action is the Dirichlet mean). Means (steps, N, K), values (steps, N)."""
        encoding = self.encode(features)
        action, thought = self.initial_state(len(features), features.device)
        means, values = [], []
        for _ in range(steps):
            thought, alpha, value = self.step(encoding, action, thought)
            action = alpha / alpha.sum(1, keepdim=True)
            means.append(action)
            values.append(value)
        return torch.stack(means), torch.stack(values)

    def final_belief(self, means, values):
        """Each image stops at its first negative value and keeps the belief it had. Returns beliefs, steps taken."""
        n_steps = ((values < 0).cumsum(0) == 0).sum(0)
        uniform = torch.full_like(means[:1], 1 / self.num_classes)
        return torch.cat([uniform, means])[n_steps, torch.arange(means.shape[1])], n_steps

    def forward(self, x):
        features = self.features(x)
        with torch.autocast("cuda", enabled=False):
            return self.final_belief(*self.trajectory(features))[0].log()


def WideResNet(depth=28, width=10, num_classes=10, dropout=0.0):
    """WRN-depth-width (Zagoruyko & Komodakis, 2016). WRN-28-10 is our (unconstrained) distillation teacher."""
    n = (depth - 4) // 6
    return PreActResNet([16 * width, 32 * width, 64 * width], [n, n, n], 16, num_classes, dropout)


class TransformerLayer(nn.Module):
    """Pre-norm transformer layer with stochastic depth (drop_path: probability of skipping the residual branch)."""

    def __init__(self, dim, heads, mlp_ratio, drop_path, attention_dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=attention_dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_ratio * dim), nn.GELU(), nn.Linear(mlp_ratio * dim, dim))
        self.drop_path = drop_path

    def residual(self, x, branch):
        if not self.training or self.drop_path == 0:
            return x + branch
        keep = (torch.rand(len(x), 1, 1, device=x.device) >= self.drop_path).to(x.dtype)
        return x + branch * keep / (1 - self.drop_path)

    def forward(self, x):
        h = self.norm1(x)
        x = self.residual(x, self.attention(h, h, h, need_weights=False)[0])
        return self.residual(x, self.mlp(self.norm2(x)))


def sinusoidal_embedding(length, dim):
    """Fixed sine/cosine position embedding (Vaswani et al., 2017), as in CCT's long-schedule models."""
    position = torch.arange(length).float()[:, None] / 10000 ** (2 * (torch.arange(dim) // 2) / dim)
    position[:, 0::2], position[:, 1::2] = position[:, 0::2].sin(), position[:, 1::2].cos()
    return position[None]


class CCT(nn.Module):
    """Compact Convolutional Transformer, CCT-7/3x1 by default (Hassani et al., 2021, SHI-Labs/Compact-Transformers):
    a 3x3 conv + max-pool tokenizer (16x16 = 256 tokens), 7 transformer layers and attention-based sequence pooling
    instead of a class token. Only used as a distillation teacher: it is not a ResNet."""

    def __init__(self, dim=256, layers=7, heads=4, mlp_ratio=2, num_classes=10, stochastic_depth=0.1,
                 positional="learnable"):
        super().__init__()
        self.tokenizer = nn.Sequential(nn.Conv2d(3, dim, 3, padding=1, bias=False), nn.ReLU(),
                                       nn.MaxPool2d(3, stride=2, padding=1))
        if positional == "sine":
            self.register_buffer("position", sinusoidal_embedding(16 * 16, dim))
        else:
            self.position = nn.Parameter(torch.zeros(1, 16 * 16, dim))
            nn.init.trunc_normal_(self.position, std=0.2)
        self.layers = nn.Sequential(*[TransformerLayer(dim, heads, mlp_ratio, stochastic_depth * i / (layers - 1))
                                      for i in range(layers)])
        self.norm = nn.LayerNorm(dim)
        self.pool = nn.Linear(dim, 1)
        self.head = nn.Linear(dim, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        x = self.tokenizer(x).flatten(2).transpose(1, 2) + self.position
        x = self.norm(self.layers(x))
        weights = self.pool(x).softmax(dim=1)
        return self.head((weights * x).sum(dim=1))


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return x + self.fn(x)


def ConvMixer(dim=384, depth=16, kernel_size=9, patch_size=1, num_classes=10):
    """ConvMixer-dim/depth (Trockman & Kolter, 2022): a patch embedding, then `depth` blocks of a residual depthwise
    conv (spatial mixing) followed by a pointwise conv (channel mixing), all at one resolution. Teacher only."""
    def block():
        return nn.Sequential(
            Residual(nn.Sequential(nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"), nn.GELU(),
                                   nn.BatchNorm2d(dim))),
            nn.Conv2d(dim, dim, 1), nn.GELU(), nn.BatchNorm2d(dim))
    return nn.Sequential(
        nn.Conv2d(3, dim, patch_size, stride=patch_size), nn.GELU(), nn.BatchNorm2d(dim),
        *[block() for _ in range(depth)],
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(dim, num_classes))


class Ensemble(nn.Module):
    """Averages the logits of several teachers (multi-teacher distillation)."""

    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):
        return torch.stack([m(x).float() for m in self.models]).mean(0)


def build_teacher(hyp):
    """Rebuilds a teacher from the hyperparameters saved in its checkpoint."""
    if hyp.get("arch") == "cct":
        return CCT(positional=hyp.get("positional", "learnable"))
    if hyp.get("arch") == "convmixer":
        return ConvMixer(hyp["dim"], hyp["depth"], hyp["kernel_size"], hyp["patch_size"])
    return WideResNet(hyp["depth"], hyp["width"])


# Students (all under 5M parameters). The ResNet candidates trade depth for width at ~4.8-5.0M parameters.
STUDENTS = {
    "airbench": lambda: AirBench(),
    "resnet18_w42": lambda: PreActResNet((42, 84, 168, 336), (2, 2, 2, 2), stem_width=42),
    "wrn28_w58": lambda: PreActResNet((58, 116, 232), (4, 4, 4)),
    "wrn22_w68": lambda: PreActResNet((68, 136, 272), (3, 3, 3)),
    "wrn16_w86": lambda: PreActResNet((86, 172, 344), (2, 2, 2)),
    "wrn28_w58_se": lambda: PreActResNet((58, 116, 232), (4, 4, 4), se=True),
    "wrn22_w68_se": lambda: PreActResNet((68, 136, 272), (3, 3, 3), se=True),
    "wrn16_w86_se": lambda: PreActResNet((86, 172, 344), (2, 2, 2), se=True),
    "wrn22_w68_ric": lambda: RIC(PreActResNet((68, 136, 272), (3, 3, 3)), feature_dim=272),
}


def count_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def load_checkpoint(name, teacher=False):
    """A trained model from checkpoints/<name>.pt, on the GPU, rebuilt from the hyperparameters saved with it."""
    ckpt = torch.load(f"checkpoints/{name}.pt", weights_only=True)
    model = build_teacher(ckpt["hyp"]) if teacher else STUDENTS[ckpt["hyp"].get("arch", "airbench")]()
    model = model.cuda().to(memory_format=torch.channels_last)
    model.load_state_dict(ckpt["model"])
    return model
