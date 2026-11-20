"""
Deep-ensemble loader and predictor.

Each ensemble lives in  <ENSEMBLE_DIR>/<gender>_<tag>/  and contains:
    member_0.pt ... member_{K-1}.pt   state dicts
    scaler.joblib                     StandardScaler fitted on the training rows
    split.npz                         train/val/test indices + test arrays
    meta.json                         input names, hyper-params, filter info, input ranges

Predictive distribution per beta (mixture of K Gaussians, moment-matched):
    mean      = mean_k(mu_k)
    aleatoric = mean_k(var_k)        noise the inputs cannot explain
    epistemic = var_k(mu_k)          disagreement between members
    total     = aleatoric + epistemic
"""
import json
import os

import joblib
import numpy as np
import torch

import config
from model import BetaGaussianMLP


def default_tag():
    return "shoulder" if config.USE_SHOULDER_BREADTH else "base"


def ensemble_dir(gender, tag=None):
    return os.path.join(config.ENSEMBLE_DIR, f"{gender}_{tag or default_tag()}")


class BetaEnsemble:
    def __init__(self, gender, tag=None, device="cpu"):
        self.gender = gender
        self.tag = tag or default_tag()
        self.dir = ensemble_dir(gender, self.tag)
        meta_path = os.path.join(self.dir, "meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"No ensemble at {self.dir}. Train it with: "
                f"python train.py --gender {gender}")
        with open(meta_path) as f:
            self.meta = json.load(f)
        self.input_names = self.meta["input_names"]
        self.input_ranges = self.meta["input_ranges"]  # name -> [min, max] over training rows
        self.scaler = joblib.load(os.path.join(self.dir, "scaler.joblib"))

        self.device = device
        self.members = []
        for k in range(self.meta["n_members"]):
            m = BetaGaussianMLP(input_dim=len(self.input_names),
                                hidden_dims=self.meta["hidden_dims"],
                                dropout=self.meta["dropout"])
            m.load_state_dict(torch.load(os.path.join(self.dir, f"member_{k}.pt"),
                                         map_location=device))
            m.to(device).eval()
            self.members.append(m)

    @torch.no_grad()
    def predict(self, X):
        """X: (n, d) raw measurements in self.input_names order."""
        X = np.asarray(X, dtype=np.float32)
        xs = torch.tensor(self.scaler.transform(X).astype(np.float32), device=self.device)
        mus, vars_ = [], []
        for m in self.members:
            mu, lv = m(xs)
            mus.append(mu)
            vars_.append(torch.exp(lv))
        mus = torch.stack(mus)      # (K, n, B)
        vars_ = torch.stack(vars_)
        mean = mus.mean(0)
        aleatoric = vars_.mean(0)
        epistemic = mus.var(0, unbiased=False)
        total = aleatoric + epistemic
        to_np = lambda t: t.cpu().numpy()
        return {
            "mean": to_np(mean),
            "aleatoric_var": to_np(aleatoric),
            "epistemic_var": to_np(epistemic),
            "total_var": to_np(total),
            "total_std": to_np(total.sqrt()),
        }

    def predict_dict(self, measurements: dict):
        """measurements: dict containing every name in self.input_names."""
        x = np.array([[measurements[n] for n in self.input_names]], dtype=np.float32)
        return self.predict(x)

    def range_warnings(self, measurements: dict):
        """Inputs outside the range the ensemble was trained on (extrapolation)."""
        out = []
        for n in self.input_names:
            lo, hi = self.input_ranges[n]
            v = measurements[n]
            if v < lo or v > hi:
                out.append(f"{n}={v:g} is outside the training range [{lo:.1f}, {hi:.1f}]")
        return out
