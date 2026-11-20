"""
Synthetic data generator (Stage 0).

Changes from the previous version:
  * betas ~ N(0,1) clipped to +-config.BETA_CLIP
  * posed in A-pose (or T-pose) via config.POSE_MODE
  * batched SMPL forward pass
  * chunked output (config.CHUNK_SIZE meshes per file) so Colab runs are
    resumable; chunks are merged into config.DATASETS[gender]

Usage:
  python generate_synthetic_data.py --gender neutral                 # all chunks + merge
  python generate_synthetic_data.py --gender neutral --chunk 3       # one chunk only
  python generate_synthetic_data.py --gender neutral --merge-only
  python generate_synthetic_data.py --gender neutral --pose-check    # writes pose_check.png
  python generate_synthetic_data.py --gender male --landmark-check   # writes landmarks_male*.png

Measurements now come from measurements_v2.py (fixed vertex-edge loops, no arms).
Run --landmark-check and inspect the overlays BEFORE generating the full dataset.
"""
import smpl_compat  # noqa: F401  -- must be imported before smplx/chumpy

import os
import csv
import glob
import numpy as np
import torch

import config
from measurements_v2 import MeasurementSet, plot_landmarks

try:
    import smplx
except ImportError as e:
    raise ImportError(
        "smplx is required. Install with: python -m pip install smplx\n"
        "and put the SMPL model files in config.SMPL_MODEL_DIR."
    ) from e

N_BODY_JOINTS = 23
# SMPL joint indices (pelvis=0). body_pose covers joints 1..23.
L_HIP, R_HIP = 1, 2
L_ELBOW, R_ELBOW = 18, 19
L_SHOULDER, R_SHOULDER = 16, 17


def build_smpl_model(gender, batch_size):
    return smplx.create(
        model_path=config.SMPL_MODEL_DIR,
        model_type="smpl",
        gender=gender,
        num_betas=config.N_BETAS,
        batch_size=batch_size,
    )


def build_measurement_set(gender: str):
    """Landmark loops defined once on the mean-shape body in the configured pose."""
    m1 = build_smpl_model(gender, 1)
    pose = torch.tensor(make_body_pose()).unsqueeze(0)
    with torch.no_grad():
        out = m1(betas=torch.zeros(1, config.N_BETAS), body_pose=pose, return_verts=True)
    v0 = out.vertices[0].numpy()
    joints = out.joints[0].numpy()
    return MeasurementSet.from_template(v0, m1.faces, joints), v0


def sample_betas(rng: np.random.Generator, n: int) -> np.ndarray:
    betas = rng.standard_normal((n, config.N_BETAS))
    return np.clip(betas, -config.BETA_CLIP, config.BETA_CLIP)


def make_body_pose(mode: str = None) -> np.ndarray:
    """(69,) axis-angle body pose. SMPL zero pose is a T-pose (arms horizontal).

    Convention assumed: +x is the subject's left, y up. Lowering the left arm
    is a NEGATIVE z-rotation at the left shoulder, the right arm POSITIVE.
    If the pose-check render shows arms rotating the wrong way (up / crossing),
    flip the signs below.
    """
    mode = mode or config.POSE_MODE
    pose = np.zeros((N_BODY_JOINTS, 3), dtype=np.float32)
    if mode == "t_pose":
        return pose.reshape(-1)
    if mode != "a_pose":
        raise ValueError(f"Unknown POSE_MODE {mode!r}")

    # arms: T-pose is 90 deg from the torso; want A_POSE_ARM_ANGLE_DEG from it.
    drop = np.deg2rad(90.0 - config.A_POSE_ARM_ANGLE_DEG)
    pose[L_SHOULDER - 1, 2] = -drop
    pose[R_SHOULDER - 1, 2] = +drop

    spread = np.deg2rad(config.A_POSE_LEG_SPREAD_DEG)
    pose[L_HIP - 1, 2] = +spread
    pose[R_HIP - 1, 2] = -spread

    bend = np.deg2rad(config.A_POSE_ELBOW_BEND_DEG)
    pose[L_ELBOW - 1, 1] = -bend
    pose[R_ELBOW - 1, 1] = +bend
    return pose.reshape(-1)


def add_measurement_noise(measurements: dict, rng: np.random.Generator) -> dict:
    noisy = {}
    for name, value in measurements.items():
        std = config.MEASUREMENT_NOISE_STD.get(name, 0.0)
        noisy[name] = value + (rng.normal(0.0, std) if std > 0 else 0.0)
    return noisy


def _chunk_path(gender: str, chunk: int) -> str:
    return os.path.join(config.DATA_DIR, f"synthetic_{gender}_chunk{chunk:03d}.csv")


def _fieldnames():
    return config.MEASUREMENT_NAMES + [f"beta_{i + 1}" for i in range(config.N_BETAS)]


def generate_chunk(gender: str, chunk: int, n_total: int = None):
    n_total = n_total or config.N_SAMPLES
    start = chunk * config.CHUNK_SIZE
    n = min(config.CHUNK_SIZE, n_total - start)
    if n <= 0:
        raise ValueError(f"chunk {chunk} is beyond N_SAMPLES={n_total}")

    # Independent, reproducible stream per chunk.
    rng = np.random.default_rng([config.RANDOM_SEED, chunk])
    bs = min(config.MESH_BATCH_SIZE, n)
    model = build_smpl_model(gender, bs)
    faces = model.faces
    body_pose_1 = torch.tensor(make_body_pose())
    mset, _ = build_measurement_set(gender)

    rows, done = [], 0
    while done < n:
        b = min(bs, n - done)
        if b != bs:  # last partial batch: rebuild model with matching batch size
            model = build_smpl_model(gender, b)
        betas_np = sample_betas(rng, b).astype(np.float32)
        betas = torch.tensor(betas_np)
        body_pose = body_pose_1.unsqueeze(0).repeat(b, 1)
        with torch.no_grad():
            out = model(betas=betas, body_pose=body_pose, return_verts=True)
        verts = out.vertices.numpy()

        for j in range(b):
            meas = mset.measure(verts[j], faces)
            meas = add_measurement_noise(meas, rng)
            row = {k: meas[k] for k in config.MEASUREMENT_NAMES}
            for bi in range(config.N_BETAS):
                row[f"beta_{bi + 1}"] = float(betas_np[j, bi])
            rows.append(row)
        done += b
        print(f"[chunk {chunk}] {done}/{n}")

    with open(_chunk_path(gender, chunk), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_fieldnames())
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {_chunk_path(gender, chunk)}")


def merge_chunks(gender: str):
    paths = sorted(glob.glob(os.path.join(config.DATA_DIR, f"synthetic_{gender}_chunk*.csv")))
    if not paths:
        raise FileNotFoundError("No chunk files found to merge.")
    n = 0
    with open(config.DATASETS[gender], "w", newline="") as out:
        w = csv.DictWriter(out, fieldnames=_fieldnames())
        w.writeheader()
        for p in paths:
            with open(p, newline="") as f:
                for row in csv.DictReader(f):
                    w.writerow(row)
                    n += 1
    print(f"Merged {len(paths)} chunks ({n} rows) -> {config.DATASETS[gender]}")


def landmark_check(gender: str):
    """Overlay the measurement loops on the mean-shape body and on a few random bodies."""
    mset, v0 = build_measurement_set(gender)
    print({k: round(float(y), 3) for k, y in mset.levels.items()})
    print("mean-shape measurements:", {k: round(x, 1) for k, x in mset.measure(v0, build_smpl_model(gender, 1).faces).items()})
    out = os.path.join(config.OUTPUT_DIR, f"landmarks_{gender}.png")
    plot_landmarks(mset, v0, out)
    print(f"Saved {out}")

    # extreme bodies (beta_1 = +-3; the sign of beta_1 differs between the male and
    # female models, so these are NOT 'heavy'/'slim'): loops should stay on the right body part
    model = build_smpl_model(gender, 1)
    pose = torch.tensor(make_body_pose()).unsqueeze(0)
    for name, b in (("b1plus", 3.0), ("b1minus", -3.0)):
        betas = torch.zeros(1, config.N_BETAS)
        betas[0, 0] = b
        with torch.no_grad():
            v = model(betas=betas, body_pose=pose, return_verts=True).vertices[0].numpy()
        plot_landmarks(mset, v, os.path.join(config.OUTPUT_DIR, f"landmarks_{gender}_{name}.png"))
        print(name, {k: round(x, 1) for k, x in mset.measure(v, model.faces).items()})


def pose_check(gender: str, out_path: str = None):
    """Render the mean-shape body in the configured pose (front + side) to verify the A-pose."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model = build_smpl_model(gender, 1)
    betas = torch.zeros(1, config.N_BETAS)
    body_pose = torch.tensor(make_body_pose()).unsqueeze(0)
    with torch.no_grad():
        v = model(betas=betas, body_pose=body_pose, return_verts=True).vertices[0].numpy()

    fig, axes = plt.subplots(1, 2, figsize=(8, 6))
    for ax, (hx, title) in zip(axes, [(0, "front (x-y)"), (2, "side (z-y)")]):
        ax.scatter(v[:, hx], v[:, 1], s=1, color="gray")
        ax.set_aspect("equal")
        ax.set_title(title)
    out_path = out_path or os.path.join(config.OUTPUT_DIR, "pose_check.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--gender", choices=["male", "female", "neutral"], required=True)
    p.add_argument("--chunk", type=int, default=None, help="generate only this chunk index")
    p.add_argument("--merge-only", action="store_true")
    p.add_argument("--pose-check", action="store_true")
    p.add_argument("--landmark-check", action="store_true")
    args = p.parse_args()

    if args.pose_check:
        pose_check(args.gender)
    elif args.landmark_check:
        landmark_check(args.gender)
    elif args.merge_only:
        merge_chunks(args.gender)
    elif args.chunk is not None:
        generate_chunk(args.gender, args.chunk)
    else:
        n_chunks = -(-config.N_SAMPLES // config.CHUNK_SIZE)
        for c in range(n_chunks):
            generate_chunk(args.gender, c)
        merge_chunks(args.gender)
