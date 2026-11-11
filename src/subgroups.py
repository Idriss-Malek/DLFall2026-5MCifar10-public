"""Subgroups inside each class (k-means on teacher features) and square-root rebalancing weights, to oversample
rare kinds of images within a class through a weighted sampler."""
import torch
import torch.nn.functional as F


@torch.no_grad()
def extract_features(model, images, batch_size=1000):
    """L2-normalized pooled features (model.features) of normalized images."""
    model.eval()
    out = []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for x in images.split(batch_size):
            out.append(model.features(x.cuda().contiguous(memory_format=torch.channels_last)).float())
    return F.normalize(torch.cat(out), dim=1)


def kmeans(x, k, iterations=100, seed=0):
    """Spherical k-means (cosine similarity) with k-means++ initialisation. x: (N, D) unit vectors.
    Returns (centers, assignment)."""
    g = torch.Generator(device=x.device).manual_seed(seed)
    centers = x[torch.randint(len(x), (1,), generator=g, device=x.device)]
    for _ in range(k - 1):
        distance = (1 - x @ centers.T).min(1).values.clamp(min=0)
        centers = torch.cat([centers, x[torch.multinomial(distance, 1, generator=g)]])
    for _ in range(iterations):
        assignment = (x @ centers.T).argmax(1)
        sums = torch.zeros_like(centers).index_add_(0, assignment, x)
        empty = sums.norm(dim=1) == 0
        sums[empty] = centers[empty]
        centers = F.normalize(sums, dim=1)
    return centers, (x @ centers.T).argmax(1)


def sqrt_weights(labels, subgroups, num_classes=10):
    """Sampling weight 1 / sqrt(subgroup size), rescaled so that every class keeps 1 / num_classes of the samples.
    subgroups: global subgroup ids (unique across classes)."""
    sizes = torch.bincount(subgroups).float()
    weights = sizes[subgroups].rsqrt()
    for c in range(num_classes):
        in_class = labels == c
        weights[in_class] *= (1 / num_classes) / weights[in_class].sum()
    return weights
