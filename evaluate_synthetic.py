"""
Stage 3: synthetic evaluation.

    python evaluate_synthetic.py --gender male                 # default ensemble
    python evaluate_synthetic.py --gender male --tag shoulder  # ensemble trained with biacromial
    python evaluate_synthetic.py --gender male --sweep         # irreducible error vs number of inputs

Reports (on the held-out synthetic test split saved by train.py):
  - MPVE (mm) of the ensemble mean vs ground-truth betas, vs the mean-shape baseline (betas = 0)
  - calibration of the beta predictive distribution (nominal vs observed coverage)
  - mesh-level uncertainty: predicted MPVE (from sampled betas) vs actual MPVE
  - --sweep: MPVE as inputs are added one at a time (height, weight, chest, waist, hips, thigh, biacromial)

MPVE here is a shape-only comparison: both meshes use SMPL's default pose, so pose cancels out.
Results are also written to outputs\\stage3_*.json / *.png for the report.
"""
import smpl_compat  # noqa: F401  -- must be imported before smplx/chumpy

import argparse
import contextlib
import io
import json
import os

import numpy as np
import torch
from scipy.stats import norm, spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from ensemble import BetaEnsemble, default_tag, ensemble_dir

import smplx

BATCH = 100
LEVELS = [0.5, 0.68, 0.8, 0.9, 0.95, 0.99]
SWEEP_ORDER = ["height", "weight", "chest", "waist", "hips", "thigh", "biacromial"]


# ------------------------------------------------------------------ SMPL ----
class MeshMaker:
    def __init__(self, gender):
        self.smpl = smplx.create(model_path=config.SMPL_MODEL_DIR, model_type="smpl",
                                 gender=gender, num_betas=config.N_BETAS, batch_size=BATCH)

    @torch.no_grad()
    def verts(self, betas):
        """(m<=BATCH, 10) -> (m, 6890, 3) metres; pads the last partial batch."""
        m = len(betas)
        b = np.asarray(betas, dtype=np.float32)
        if m < BATCH:
            b = np.concatenate([b, np.repeat(b[-1:], BATCH - m, axis=0)])
        return self.smpl(betas=torch.tensor(b), return_verts=True).vertices.numpy()[:m]

    def per_body_dist(self, A, B):
        """Mean per-vertex distance (metres) between mesh(A[i]) and mesh(B[i]), one value per body."""
        out = []
        for i in range(0, len(A), BATCH):
            va, vb = self.verts(A[i:i + BATCH]), self.verts(B[i:i + BATCH])
            out.append(np.linalg.norm(va - vb, axis=2).mean(axis=1))
        return np.concatenate(out)


def summ(a):
    return {"mean": float(np.mean(a)), "median": float(np.median(a)), "p95": float(np.percentile(a, 95))}


# --------------------------------------------------------------- compute ----
def compute(gender, tag, maker, n_draws):
    ens = BetaEnsemble(gender, tag)
    sp = np.load(os.path.join(ens.dir, "split.npz"))
    X, y, ok = sp["X_test"], sp["y_test"], sp["test_plausible"].astype(bool)
    p = ens.predict(X)
    mu, std = p["mean"], p["total_std"]

    d_model = maker.per_body_dist(mu, y) * 1000
    d_base = maker.per_body_dist(np.zeros_like(y), y) * 1000
    err = mu - y

    res = {
        "gender": gender, "tag": tag, "inputs": ens.input_names,
        "n_test": int(len(y)), "n_plausible": int(ok.sum()),
        "mpve_mm_plausible": summ(d_model[ok]),
        "mpve_mm_all": summ(d_model),
        "mpve_mm_removed": summ(d_model[~ok]) if (~ok).any() else None,
        "baseline_mean_shape_mpve_mm_plausible": summ(d_base[ok]),
        "beta_mse_plausible": float(np.mean(err[ok] ** 2)),
        "beta_mse_per_beta_plausible": np.mean(err[ok] ** 2, axis=0).tolist(),
    }

    # --- calibration of the beta distribution (plausible rows) ---
    z = np.abs(err[ok]) / std[ok]
    cal = {"levels": LEVELS, "all": [], "beta_1_2": [], "beta_3_10": []}
    for lv in LEVELS:
        thr = norm.ppf(0.5 + lv / 2)
        cal["all"].append(float(np.mean(z < thr)))
        cal["beta_1_2"].append(float(np.mean(z[:, :2] < thr)))
        cal["beta_3_10"].append(float(np.mean(z[:, 2:] < thr)))
    res["calibration"] = cal

    # --- mesh-level uncertainty: predicted vs actual MPVE ---
    if n_draws > 0:
        rng = np.random.default_rng(config.RANDOM_SEED)
        pred_d = np.zeros(len(y))
        for _ in range(n_draws):
            draw = mu + std * rng.standard_normal(mu.shape).astype(np.float32)
            pred_d += maker.per_body_dist(draw, mu) * 1000
        pred_d /= n_draws
        rho = spearmanr(pred_d[ok], d_model[ok]).correlation
        q = np.quantile(pred_d[ok], [0.2, 0.4, 0.6, 0.8])
        bins = np.digitize(pred_d[ok], q)
        res["mesh_uncertainty"] = {
            "n_draws": n_draws,
            "mean_predicted_mm": float(pred_d[ok].mean()),
            "mean_actual_mm": float(d_model[ok].mean()),
            "ratio_actual_over_predicted": float(d_model[ok].mean() / pred_d[ok].mean()),
            "spearman_predicted_vs_actual": float(rho),
            "quintiles_predicted_mm": [float(pred_d[ok][bins == k].mean()) for k in range(5)],
            "quintiles_actual_mm": [float(d_model[ok][bins == k].mean()) for k in range(5)],
        }
    return res


def report(res):
    m, b = res["mpve_mm_plausible"], res["baseline_mean_shape_mpve_mm_plausible"]
    print(f"\n=== {res['gender']} / {res['tag']}  inputs: {', '.join(res['inputs'])} ===")
    print(f"test rows: {res['n_test']} ({res['n_plausible']} plausible)")
    print(f"MPVE plausible rows : mean {m['mean']:.2f} mm  median {m['median']:.2f}  p95 {m['p95']:.2f}")
    print(f"MPVE all rows       : mean {res['mpve_mm_all']['mean']:.2f} mm")
    if res["mpve_mm_removed"]:
        print(f"MPVE removed rows   : mean {res['mpve_mm_removed']['mean']:.2f} mm")
    print(f"mean-shape baseline : mean {b['mean']:.2f} mm  median {b['median']:.2f}  p95 {b['p95']:.2f}")
    print(f"beta MSE (plausible): {res['beta_mse_plausible']:.4f}")
    c = res["calibration"]
    print("\ncalibration (observed coverage at nominal level)")
    print("  nominal   all    b1-2   b3-10")
    for i, lv in enumerate(c["levels"]):
        print(f"  {lv:7.2f}  {c['all'][i]:.3f}  {c['beta_1_2'][i]:.3f}  {c['beta_3_10'][i]:.3f}")
    u = res.get("mesh_uncertainty")
    if u:
        print(f"\nmesh-level uncertainty ({u['n_draws']} draws): predicted {u['mean_predicted_mm']:.2f} mm vs "
              f"actual {u['mean_actual_mm']:.2f} mm (ratio {u['ratio_actual_over_predicted']:.2f}), "
              f"Spearman {u['spearman_predicted_vs_actual']:.3f}")
        print("  quintile of predicted MPVE:   " + "  ".join(f"{v:6.2f}" for v in u["quintiles_predicted_mm"]))
        print("  actual MPVE in that quintile: " + "  ".join(f"{v:6.2f}" for v in u["quintiles_actual_mm"]))


def plot_calibration(res, path):
    c = res["calibration"]
    plt.figure(figsize=(4.5, 4.5))
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="ideal")
    plt.plot(c["levels"], c["all"], "o-", label="all betas")
    plt.plot(c["levels"], c["beta_1_2"], "s-", label="betas 1-2")
    plt.plot(c["levels"], c["beta_3_10"], "^-", label="betas 3-10")
    plt.xlabel("nominal coverage")
    plt.ylabel("observed coverage")
    plt.title(f"Calibration: {res['gender']} / {res['tag']}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ----------------------------------------------------------------- sweep ----
def sweep(gender, maker, members):
    import train as trainmod   # train.py in the project root

    rows = []
    for k in range(1, len(SWEEP_ORDER) + 1):
        inputs = SWEEP_ORDER[:k]
        tag = f"sweep_n{k}"
        if not os.path.exists(os.path.join(ensemble_dir(gender, tag), "meta.json")):
            print(f"training {gender} {tag}: {inputs}")
            with contextlib.redirect_stdout(io.StringIO()):
                trainmod.train(gender, inputs, tag, members, use_filter=True)
        res = compute(gender, tag, maker, n_draws=0)
        rows.append({"n_inputs": k, "added": inputs[-1],
                     "mpve_mm": res["mpve_mm_plausible"]["mean"],
                     "beta_mse": res["beta_mse_plausible"],
                     "baseline_mm": res["baseline_mean_shape_mpve_mm_plausible"]["mean"]})
        print(f"  {k} inputs (+{inputs[-1]:10s}) MPVE {rows[-1]['mpve_mm']:6.2f} mm   beta MSE {rows[-1]['beta_mse']:.4f}")

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(config.OUTPUT_DIR, f"stage3_sweep_{gender}.json"), "w") as f:
        json.dump(rows, f, indent=2)
    plt.figure(figsize=(6, 4))
    plt.plot([r["n_inputs"] for r in rows], [r["mpve_mm"] for r in rows], "o-", label="model")
    plt.axhline(rows[0]["baseline_mm"], color="k", ls="--", lw=1, label="mean-shape baseline")
    plt.xticks([r["n_inputs"] for r in rows], [r["added"] for r in rows], rotation=30)
    plt.ylabel("MPVE (mm)")
    plt.title(f"Error vs number of inputs ({gender}, {members} member(s))")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_DIR, f"stage3_sweep_{gender}.png"), dpi=150)
    plt.close()


# ------------------------------------------------------------------ main ----
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gender", choices=["male", "female", "neutral"], required=True)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--draws", type=int, default=8, help="beta samples per body for mesh-level uncertainty")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--sweep-members", type=int, default=1)
    a = ap.parse_args()

    maker = MeshMaker(a.gender)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    if a.sweep:
        sweep(a.gender, maker, a.sweep_members)
    else:
        tag = a.tag or default_tag()
        res = compute(a.gender, tag, maker, a.draws)
        report(res)
        stem = os.path.join(config.OUTPUT_DIR, f"stage3_{a.gender}_{tag}")
        with open(stem + ".json", "w") as f:
            json.dump(res, f, indent=2)
        plot_calibration(res, stem + "_calibration.png")
        print(f"\nSaved {stem}.json and {stem}_calibration.png")
