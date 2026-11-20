"""
Losses for the uncertainty model.

gaussian_nll: per-beta heteroscedastic Gaussian negative log-likelihood
    0.5 * (logvar + (y - mu)^2 / exp(logvar)), averaged over betas and batch.
warmup_mse:   plain 0.5 * squared error on the mean. Used for the first few
    epochs so the mean head settles before the variance head starts to move
    (a common fix for NLL training getting stuck by inflating variance).

The old vertex loss is removed on purpose: it needed an SMPL forward pass per
batch (slow on this laptop) and it only constrains the mean. MPVE is measured
in Stage 3 instead.
"""
import torch


def gaussian_nll(mu: torch.Tensor, logvar: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (0.5 * (logvar + (y - mu) ** 2 * torch.exp(-logvar))).mean()


def warmup_mse(mu: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (0.5 * (y - mu) ** 2).mean()
