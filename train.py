"""
Stage 2: train a deep ensemble (mean + log-variance per beta, Gaussian NLL).

Usage (venv active):
    python train.py --gender male
    python train.py --gender female

Ablation / Stage 3 variants (same rows and split, different inputs):
    python train.py --gender male --inputs height weight chest waist hips thigh biacromial --tag shoulder
    python train.py --gender male --inputs height weight --tag h_w

Other switches:
    --members N     ensemble size (default config.ENSEMBLE_SIZE)
    --no-filter     skip the plausibility filter (tag gets a _nofilter suffix)

No SMPL forward pass is needed here, so this runs fine on the laptop CPU.
"""
import argparse
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

import config
from ensemble import BetaEnsemble, default_tag, ensemble_dir
from losses import gaussian_nll, warmup_mse
from model import BetaGaussianMLP

BETA_COLS = [f"beta_{i + 1}" for i in range(config.N_BETAS)]


# ---------------------------------------------------------------- data ----
def load_dataframe(gender):
    path = config.SYNTHETIC_CSV[gender]
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run generate_synthetic_data.py --gender {gender} first.")
    return pd.read_csv(path)


def plausibility_mask(df, gender):
    """True = row kept. Uses BodyM-train ranges on config.PLAUSIBILITY_COLUMNS only
    (never biacromial: the synthetic proxy is offset from BodyM's definition)."""
    if not os.path.exists(config.PLAUSIBILITY_RANGES_PATH):
        raise FileNotFoundError(
            f"{config.PLAUSIBILITY_RANGES_PATH} missing. Run make_plausibility_ranges.py first "
            f"(or train with --no-filter).")
    with open(config.PLAUSIBILITY_RANGES_PATH) as f:
        ranges = json.load(f)[gender]
    keep = np.ones(len(df), dtype=bool)
    per_col = {}
    for col in config.PLAUSIBILITY_COLUMNS:
        bad = ((df[col] < ranges[col]["lo"]) | (df[col] > ranges[col]["hi"])).values
        per_col[col] = float(bad.mean())
        keep &= ~bad
    return keep, per_col


def split_indices(n):
    rng = np.random.default_rng(config.RANDOM_SEED)
    idx = rng.permutation(n)
    tr, va, _ = config.TRAIN_VAL_TEST_SPLIT
    n_tr, n_va = int(n * tr), int(n * va)
    return idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]


# ------------------------------------------------------------ training ----
def train_member(seed, Xtr, ytr, Xva, yva, input_dim, tag):
    device = config.DEVICE
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)

    model = BetaGaussianMLP(input_dim=input_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE,
                           weight_decay=config.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=10)
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=config.BATCH_SIZE,
                        shuffle=True, generator=g)
    Xva_d, yva_d = Xva.to(device), yva.to(device)

    warmup = config.NLL_WARMUP_EPOCHS
    patience = config.EARLY_STOP_PATIENCE
    best, best_state, bad, best_epoch = float("inf"), None, 0, -1

    for epoch in range(config.EPOCHS):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            mu, lv = model(xb)
            loss = warmup_mse(mu, yb) if epoch < warmup else gaussian_nll(mu, lv, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            mu, lv = model(Xva_d)
            val_nll = gaussian_nll(mu, lv, yva_d).item()
            val_mse = ((mu - yva_d) ** 2).mean().item()

        if epoch < warmup:          # variance head not trained yet: don't select on NLL
            continue
        sched.step(val_nll)
        if val_nll < best - 1e-5:
            best, bad, best_epoch = val_nll, 0, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if (epoch + 1) % 10 == 0:
            print(f"  [{tag}] epoch {epoch + 1:4d}  val_nll={val_nll:.4f}  val_mse={val_mse:.4f}  best={best:.4f}")
        if bad >= patience:
            print(f"  [{tag}] early stop at epoch {epoch + 1} (best epoch {best_epoch + 1})")
            break

    return best_state, best, best_epoch + 1


# ------------------------------------------------------------- summary ----
def summarize(ens, X, y, label):
    p = ens.predict(X)
    err = p["mean"] - y
    z = np.abs(err) / p["total_std"]
    print(f"\n--- {label} (n={len(X)}) ---")
    print(f"beta MSE overall: {np.mean(err ** 2):.4f}")
    print(f"coverage of +-1 std: {np.mean(z < 1):.3f} (ideal 0.683)   +-1.96 std: {np.mean(z < 1.96):.3f} (ideal 0.950)")
    print("beta   mse     mean_std  aleat_var  epist_var")
    for b in range(config.N_BETAS):
        print(f"  {b + 1:2d}  {np.mean(err[:, b] ** 2):7.4f}  {p['total_std'][:, b].mean():7.4f}  "
              f"{p['aleatoric_var'][:, b].mean():8.4f}  {p['epistemic_var'][:, b].mean():8.5f}")


# ---------------------------------------------------------------- main ----
def train(gender, input_names, tag, n_members, use_filter):
    t0 = time.time()
    df = load_dataframe(gender)
    n = len(df)

    keep, per_col_bad = (np.ones(n, dtype=bool), {})
    if use_filter:
        keep, per_col_bad = plausibility_mask(df, gender)
        print(f"Plausibility filter: keeping {keep.sum()}/{n} rows ({100 * (1 - keep.mean()):.1f}% removed)")
        for c, f in per_col_bad.items():
            print(f"   {c:7s} outside BodyM range: {100 * f:.1f}%")

    tr_idx, va_idx, te_idx = split_indices(n)
    # test set keeps ALL rows (flagged), so Stage 3 can report filtered and unfiltered
    tr_idx, va_idx = tr_idx[keep[tr_idx]], va_idx[keep[va_idx]]
    print(f"Split: train {len(tr_idx)}  val {len(va_idx)}  test {len(te_idx)} "
          f"({int(keep[te_idx].sum())} plausible)")

    X_all = df[input_names].values.astype(np.float32)
    y_all = df[BETA_COLS].values.astype(np.float32)

    scaler = StandardScaler().fit(X_all[tr_idx])
    f32 = lambda a: torch.tensor(scaler.transform(a).astype(np.float32))
    Xtr, Xva = f32(X_all[tr_idx]), f32(X_all[va_idx])
    ytr, yva = torch.tensor(y_all[tr_idx]), torch.tensor(y_all[va_idx])

    out_dir = ensemble_dir(gender, tag)
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(out_dir, "scaler.joblib"))

    member_info = []
    for k in range(n_members):
        print(f"\n=== member {k + 1}/{n_members} (seed {config.RANDOM_SEED + k}) ===")
        state, best, best_epoch = train_member(config.RANDOM_SEED + k, Xtr, ytr, Xva, yva,
                                               len(input_names), f"{gender}/{tag}/m{k}")
        torch.save(state, os.path.join(out_dir, f"member_{k}.pt"))
        member_info.append({"seed": config.RANDOM_SEED + k, "best_val_nll": best, "best_epoch": best_epoch})

    meta = {
        "gender": gender, "tag": tag, "input_names": input_names,
        "n_members": n_members, "n_betas": config.N_BETAS,
        "hidden_dims": list(config.HIDDEN_DIMS), "dropout": config.DROPOUT,
        "plausibility_filter": use_filter, "filter_bad_fraction_per_column": per_col_bad,
        "n_train": int(len(tr_idx)), "n_val": int(len(va_idx)), "n_test": int(len(te_idx)),
        "input_ranges": {nm: [float(df.loc[tr_idx, nm].min()), float(df.loc[tr_idx, nm].max())]
                         for nm in input_names},
        "members": member_info,
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    np.savez(os.path.join(out_dir, "split.npz"),
             train_idx=tr_idx, val_idx=va_idx, test_idx=te_idx,
             test_plausible=keep[te_idx],
             X_test=X_all[te_idx], y_test=y_all[te_idx])

    # quick sanity look (formal evaluation is Stage 3)
    ens = BetaEnsemble(gender, tag)
    Xte, yte, ok = X_all[te_idx], y_all[te_idx], keep[te_idx]
    summarize(ens, Xte[ok], yte[ok], "test, plausible rows")
    if use_filter:
        summarize(ens, Xte[~ok], yte[~ok], "test, rows REMOVED by filter (out-of-range bodies)")
    print(f"\nSaved ensemble to {out_dir}  ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gender", choices=["male", "female", "neutral"], required=True)
    ap.add_argument("--members", type=int, default=config.ENSEMBLE_SIZE)
    ap.add_argument("--inputs", nargs="+", default=None,
                    help="override input columns (requires --tag)")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--no-filter", action="store_true")
    a = ap.parse_args()

    if a.inputs and not a.tag:
        ap.error("--inputs needs an explicit --tag")
    inputs = a.inputs or list(config.INPUT_NAMES)
    tag = a.tag or default_tag()
    if a.no_filter and not a.tag:
        tag += "_nofilter"
    train(a.gender, inputs, tag, a.members, use_filter=not a.no_filter)
