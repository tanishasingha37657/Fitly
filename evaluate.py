"""
Step 4 of the pipeline: evaluate the trained MLP on the held-out test split.

Reports:
  - MPVE (Mean Per-Vertex Error) -- primary metric
  - MSE on betas -- secondary metric, also broken down per-beta index
  - Per-measurement reconstruction error -- most interpretable signal
"""

import smpl_compat  # noqa: F401  -- must be imported before smplx/chumpy

import numpy as np
import torch
import joblib

import config
from model import MeasurementToBetaMLP
from measurements import extract_measurements

try:
    import smplx
except ImportError as e:
    raise ImportError(
        "smplx is required for evaluation. Install with:\n"
        "  pip install smplx --break-system-packages"
    ) from e


def load_test_set(gender):
    data = np.load(config.TEST_SPLITS[gender])
    return data["X_test"], data["y_test"]


def load_model_and_scaler(gender):
    model = MeasurementToBetaMLP()
    model.load_state_dict(torch.load(config.MODEL_CHECKPOINTS[gender], map_location="cpu"))
    model.eval()
    scaler = joblib.load(config.SCALERS[gender])
    return model, scaler


def build_smpl_model(gender, batch_size):
    return smplx.create(
        model_path=config.SMPL_MODEL_DIR,
        model_type="smpl",
        gender=gender,
        num_betas=config.N_BETAS,
        batch_size=batch_size,
    )


def evaluate(gender):
    X_test, y_test = load_test_set(gender)
    model, scaler = load_model_and_scaler(gender)

    X_test_s = scaler.transform(X_test).astype(np.float32)
    with torch.no_grad():
        pred_betas = model(torch.tensor(X_test_s)).numpy()

    # --- Beta MSE (overall + per-beta) ---
    beta_sq_err = (pred_betas - y_test) ** 2
    overall_beta_mse = beta_sq_err.mean()
    per_beta_mse = beta_sq_err.mean(axis=0)

    print("=== Beta MSE ===")
    print(f"Overall: {overall_beta_mse:.5f}")
    for i, mse in enumerate(per_beta_mse):
        print(f"  beta_{i + 1}: {mse:.5f}")

    # --- MPVE ---
    n_test = X_test.shape[0]
    smpl_model = build_smpl_model(gender, batch_size=n_test)
    with torch.no_grad():
        pred_verts = smpl_model(betas=torch.tensor(pred_betas), return_verts=True).vertices.numpy()
        true_verts = smpl_model(betas=torch.tensor(y_test), return_verts=True).vertices.numpy()

    per_vertex_dist = np.linalg.norm(pred_verts - true_verts, axis=2)  # (n_test, n_vertices)
    mpve = per_vertex_dist.mean() * 1000  # convert m -> mm
    print(f"\n=== MPVE ===\n{mpve:.3f} mm")

    # --- Per-measurement reconstruction error ---
    faces = smpl_model.faces
    meas_errors = {name: [] for name in config.MEASUREMENT_NAMES}
    for i in range(n_test):
        pred_meas = extract_measurements(pred_verts[i], faces)
        true_meas = extract_measurements(true_verts[i], faces)
        for name in config.MEASUREMENT_NAMES:
            meas_errors[name].append(abs(pred_meas[name] - true_meas[name]))

    print("\n=== Per-measurement reconstruction error (mean absolute, cm/kg) ===")
    for name in config.MEASUREMENT_NAMES:
        print(f"  {name}: {np.mean(meas_errors[name]):.3f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gender",
        choices=["male", "female", "neutral"],
        required=True,
    )

    args = parser.parse_args()

    evaluate(args.gender)
