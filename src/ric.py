"""Reinforced Iterative Classification (Kallel et al., 2026, arXiv:2604.22110). Classification as an episode: starting
from a uniform belief, a recurrent policy outputs a new belief at every step and is rewarded by the change in
log-probability of the true class. Trained with SPO (Xie et al., 2025) and GAE, from a frozen snapshot of the policy."""
import copy
import math
import time

import torch
import torch.nn.functional as F
from torch.distributions import Gamma
from tqdm import tqdm

from .results import save_logs

RIC_HYP = {
    "optimizer": "adam",
    "epochs": 200,                 # RIC epochs after the head warmup (the paper: 2000 from scratch)
    "batch_size": 128,
    "lr": 3e-4,                    # constant, as in the paper; used for the new head
    "encoder_lr": 3e-6,            # ours: at 3e-4 Adam destroys the SGD-trained encoder (results/ric_encoder_lr_check)
    "weight_decay": 1e-3,          # not on the concentration layer: it decays its bias and drags the policy to c = 1
    "clip_norm": 0.5,
    "horizon": 20,
    "gamma": 0.8,
    "passes_per_snapshot": 5,
    "translate": 4,                # not given by the paper: same as the student being continued
    "cutout": 16,
    "gae_lambda": 0.95,            # not given by the paper: defaults of the official SPO code
    "spo_epsilon": 0.2,
    "value_coef": 0.5,
    "head_epochs": 5,              # ours: the new head starts random, so it is first fitted with the encoder frozen
    "min_probability": 1e-6,       # ours: floor on the true-class probability in the reward, not in the policy
    "max_log_ratio": 10.0,         # ours: keeps the probability ratio of the SPO loss finite
    "ece_bins": 15,                # not given by the paper
}


def true_class_score(log_action, y, hyp):
    return log_action.gather(1, y[:, None]).squeeze(1).clamp(min=math.log(hyp["min_probability"]))


def sample_log_dirichlet(alpha):
    """Log of a Dirichlet sample. Unlikely classes have alpha near 0.01, whose samples are too small for float32 and
    would round to 0, so the sample is drawn in log space, using Gamma(a) = Gamma(a + 1) * U^(1 / a)."""
    uniform = torch.rand_like(alpha).clamp(min=1e-30)
    log_gamma = Gamma(alpha + 1, 1.0).sample().log() + uniform.log() / alpha
    return log_gamma - log_gamma.logsumexp(1, keepdim=True)


def dirichlet_log_prob(alpha, log_action):
    return ((alpha - 1) * log_action).sum(1) + torch.lgamma(alpha.sum(1)) - torch.lgamma(alpha).sum(1)


def advantages_and_returns(rewards, values, next_value, gamma, lam):
    """Generalized advantage estimation, bootstrapped with the value after the last step."""
    advantages = torch.zeros_like(rewards)
    last = 0
    for t in reversed(range(len(rewards))):
        following = next_value if t == len(rewards) - 1 else values[t + 1]
        last = rewards[t] + gamma * following - values[t] + gamma * lam * last
        advantages[t] = last
    return advantages, advantages + values


@torch.no_grad()
def rollout(snapshot, x, y, hyp):
    """One sampled episode per image from the frozen policy."""
    encoding = snapshot.encode(snapshot.features(x))
    action, thought = snapshot.initial_state(len(x), x.device)
    log_action = action.log()
    score = true_class_score(log_action, y, hyp)
    log_actions, log_probs, rewards, values = [log_action], [], [], []
    for _ in range(hyp["horizon"]):
        thought, alpha, value = snapshot.step(encoding, action, thought)
        log_action = sample_log_dirichlet(alpha)
        action = log_action.exp()
        new_score = true_class_score(log_action, y, hyp)
        log_actions.append(log_action)
        log_probs.append(dirichlet_log_prob(alpha, log_action))
        rewards.append(new_score - score)
        values.append(value)
        score = new_score
    next_value = snapshot.step(encoding, action, thought)[2]
    values, rewards = torch.stack(values), torch.stack(rewards)
    advantages, returns = advantages_and_returns(rewards, values, next_value, hyp["gamma"], hyp["gae_lambda"])
    return dict(log_actions=log_actions, log_probs=torch.stack(log_probs), advantages=advantages, returns=returns,
                episode_return=rewards.sum(0).mean().item())


def ric_loss(model, x, episodes, hyp):
    """SPO policy loss plus value loss, replaying the sampled actions through the current parameters."""
    encoding = model.encode(model.features(x))
    thought = model.initial_state(len(x), x.device)[1]
    log_probs, values, concentrations = [], [], []
    for t in range(hyp["horizon"]):
        thought, alpha, value = model.step(encoding, episodes["log_actions"][t].exp(), thought)
        log_probs.append(dirichlet_log_prob(alpha, episodes["log_actions"][t + 1]))
        values.append(value)
        concentrations.append(alpha.sum(1).detach().mean())

    limit = hyp["max_log_ratio"]
    ratio = (torch.stack(log_probs) - episodes["log_probs"]).clamp(-limit, limit).exp()
    advantages = episodes["advantages"]
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    policy_loss = -(advantages * ratio - advantages.abs() * (ratio - 1) ** 2 / (2 * hyp["spo_epsilon"])).mean()
    value_loss = 0.5 * (torch.stack(values) - episodes["returns"]).pow(2).mean()
    stats = dict(policy_loss=policy_loss.item(), value_loss=value_loss.item(),
                 ratio_deviation=(ratio.detach() - 1).abs().mean().item(),
                 concentration=torch.stack(concentrations).mean().item())
    return policy_loss + hyp["value_coef"] * value_loss, stats


def head_loss(model, x, y, hyp):
    """Cross-entropy at every refinement step, without touching the encoder."""
    with torch.no_grad():
        features = model.features(x)
    means = model.trajectory(features, hyp["horizon"])[0]
    return F.nll_loss(means.log().flatten(0, 1), y.repeat(hyp["horizon"]))


def expected_calibration_error(probs, labels, bins):
    confidence, predicted = probs.max(1)
    correct = (predicted == labels).float()
    edges = torch.linspace(0, 1, bins + 1, device=probs.device)
    ece = torch.zeros(1, device=probs.device)
    for low, high in zip(edges[:-1], edges[1:]):
        inside = (confidence > low) & (confidence <= high)
        if inside.any():
            ece += inside.float().mean() * (correct[inside].mean() - confidence[inside].mean()).abs()
    return ece.item()


@torch.no_grad()
def evaluate_ric(model, images, labels, hyp, batch_size=2000):
    model.eval()
    device = next(model.parameters()).device
    trajectories = []
    for x in images.split(batch_size):
        x = x.to(device, torch.float16).contiguous(memory_format=torch.channels_last)
        trajectories.append(model.trajectory(model.features(x), hyp["horizon"]))
    means, values = torch.cat([t[0] for t in trajectories], 1), torch.cat([t[1] for t in trajectories], 1)
    probs, n_steps = model.final_belief(means, values)
    labels = labels.to(device)
    return dict(loss=F.nll_loss(probs.log(), labels).item(), acc=(probs.argmax(1) == labels).float().mean().item(),
                acc_first_step=(means[0].argmax(1) == labels).float().mean().item(),
                acc_last_step=(means[-1].argmax(1) == labels).float().mean().item(),
                ece=expected_calibration_error(probs, labels, hyp["ece_bins"]), mean_steps=n_steps.float().mean().item())


def train_ric(model, loader, test_images, test_labels, hyp, log_dir, last_path, resume=None):
    """Head warmup, then Algorithm 1 of the paper. Logs after every epoch; at the end of every round (snapshot), saves
    last_path so an interrupted run can continue. The test set is only logged: nothing is selected on it."""
    encoder_ids = {id(p) for p in model.encoder.parameters()}
    head_params = [p for p in model.parameters() if id(p) not in encoder_ids]
    head_optimizer = torch.optim.Adam(head_params, lr=hyp["lr"])
    concentration_ids = {id(p) for p in model.concentration.parameters()}
    optimizer = torch.optim.Adam([dict(params=[p for p in head_params if id(p) not in concentration_ids], lr=hyp["lr"]),
                                  dict(params=model.encoder.parameters(), lr=hyp["encoder_lr"]),
                                  dict(params=model.concentration.parameters(), lr=hyp["lr"], weight_decay=0.0)],
                                 weight_decay=hyp["weight_decay"])
    snapshot = copy.deepcopy(model)

    history = {k: [] for k in ["epoch", "phase", "train_loss", "policy_loss", "value_loss", "episode_return",
                               "ratio_deviation", "concentration", "val_loss", "val_acc", "val_acc_first_step",
                               "val_acc_last_step", "val_ece", "val_mean_steps", "lr", "time"]}
    steps = {"step": [], "epoch": [], "train_loss": [], "lr": []}
    first_epoch, elapsed = 0, 0.0
    if resume is not None:
        optimizer.load_state_dict(resume["optimizer"])
        history, steps, first_epoch = resume["history"], resume["steps"], resume["epoch"]
        elapsed = history["time"][-1]

    total_epochs = hyp["head_epochs"] + hyp["epochs"]
    start = time.time()
    bar = tqdm(total=(total_epochs - first_epoch) * len(loader), desc="training", unit="step", mininterval=10)
    for epoch in range(first_epoch, total_epochs):
        warmup = epoch < hyp["head_epochs"]
        ric_epoch = epoch - hyp["head_epochs"]
        if not warmup and ric_epoch % hyp["passes_per_snapshot"] == 0:
            snapshot.load_state_dict(model.state_dict())

        model.train()
        snapshot.train()
        if warmup:
            model.encoder.eval()
        epoch_stats, losses = [], []
        for x, y in loader:
            if warmup:
                loss, stats = head_loss(model, x, y, hyp), {}
                current = head_optimizer
            else:
                episodes = rollout(snapshot, x, y, hyp)
                loss, stats = ric_loss(model, x, episodes, hyp)
                stats["episode_return"] = episodes["episode_return"]
                current = optimizer
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {epoch + 1}")
            current.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), hyp["clip_norm"])
            current.step()
            losses.append(loss.item())
            epoch_stats.append(stats)
            steps["step"].append(len(steps["step"]) + 1)
            steps["epoch"].append(epoch + 1)
            steps["train_loss"].append(losses[-1])
            steps["lr"].append(hyp["lr"])
            bar.update()

        test = evaluate_ric(model, test_images, test_labels, hyp)
        history["epoch"].append(epoch + 1)
        history["phase"].append("head" if warmup else "ric")
        history["train_loss"].append(sum(losses) / len(losses))
        for key in ["policy_loss", "value_loss", "episode_return", "ratio_deviation", "concentration"]:
            history[key].append(float("nan") if warmup else sum(s[key] for s in epoch_stats) / len(epoch_stats))
        for key in ["loss", "acc", "acc_first_step", "acc_last_step", "ece", "mean_steps"]:
            history[f"val_{key}"].append(test[key])
        history["lr"].append(hyp["lr"])
        history["time"].append(elapsed + time.time() - start)
        save_logs(log_dir, history, steps)
        bar.set_postfix(loss=f"{history['train_loss'][-1]:.3f}", test_acc=f"{test['acc']:.4f}",
                        ret=f"{history['episode_return'][-1]:.2f}", ece=f"{test['ece']:.4f}")
        if not warmup and (ric_epoch + 1) % hyp["passes_per_snapshot"] == 0:
            torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch + 1,
                            history=history, steps=steps), last_path)
    bar.close()
    return history, steps
