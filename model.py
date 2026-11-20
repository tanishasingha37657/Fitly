"""
MLP with two heads: measurements -> (mean betas, log-variance of betas).

The log-variance head is what gives us per-beta (aleatoric) uncertainty. It is
soft-clamped to [LOGVAR_MIN, LOGVAR_MAX] so the NLL cannot blow up, without
the dead gradient a hard clamp has at the boundaries.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import config

LOGVAR_MIN = getattr(config, "LOGVAR_MIN", -8.0)
LOGVAR_MAX = getattr(config, "LOGVAR_MAX", 4.0)


def soft_clamp(x, lo, hi):
    x = hi - F.softplus(hi - x)   # smooth upper bound
    x = lo + F.softplus(x - lo)   # smooth lower bound
    return x


class BetaGaussianMLP(nn.Module):
    def __init__(self, input_dim=None, output_dim=config.N_BETAS,
                 hidden_dims=None, dropout=None):
        super().__init__()
        input_dim = input_dim if input_dim is not None else len(config.INPUT_NAMES)
        hidden_dims = hidden_dims or config.HIDDEN_DIMS
        dropout = config.DROPOUT if dropout is None else dropout

        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.trunk = nn.Sequential(*layers)
        self.mean_head = nn.Linear(prev, output_dim)
        self.logvar_head = nn.Linear(prev, output_dim)

    def forward(self, x):
        h = self.trunk(x)
        mu = self.mean_head(h)
        logvar = soft_clamp(self.logvar_head(h), LOGVAR_MIN, LOGVAR_MAX)
        return mu, logvar
