"""
End-to-end inference: measurements -> ensemble -> betas (+ uncertainty) -> SMPL -> mesh (.obj).

Betas are only ever fed to the SMPL model of the SAME gender as the ensemble
that predicted them (beta signs/meaning differ between male/female/neutral).
"""
import smpl_compat  # noqa: F401  -- must be imported before smplx/chumpy

import argparse
from functools import lru_cache

import numpy as np
import torch

import config
from ensemble import BetaEnsemble

try:
    import smplx
except ImportError as e:
    raise ImportError("smplx is required for mesh generation: python -m pip install smplx") from e


@lru_cache(maxsize=None)
def get_ensemble(gender, tag=None):
    return BetaEnsemble(gender, tag)


@lru_cache(maxsize=None)
def get_smpl(gender):
    return smplx.create(model_path=config.SMPL_MODEL_DIR, model_type="smpl",
                        gender=gender, num_betas=config.N_BETAS, batch_size=1)


def measurements_to_mesh(measurements: dict, gender, tag=None):
    """
    measurements: dict with keys covering the ensemble's input names
                  (config.INPUT_NAMES for the default ensemble).
    Returns (vertices, faces, info) where info has:
        betas, beta_std (total), beta_std_aleatoric, beta_std_epistemic, warnings
    """
    ens = get_ensemble(gender, tag)
    smpl = get_smpl(gender)

    pred = ens.predict_dict(measurements)
    pred["mean"] = np.clip(pred["mean"], -config.BETA_CLIP, config.BETA_CLIP)

    betas = torch.tensor(pred["mean"], dtype=torch.float32)
    with torch.no_grad():
        verts = smpl(betas=betas, return_verts=True).vertices[0].numpy()

    info = {
        "betas": pred["mean"][0],
        "beta_std": pred["total_std"][0],
        "beta_std_aleatoric": np.sqrt(pred["aleatoric_var"][0]),
        "beta_std_epistemic": np.sqrt(pred["epistemic_var"][0]),
        "warnings": ens.range_warnings(measurements),
    }
    return verts, smpl.faces, info


def save_obj(vertices: np.ndarray, faces: np.ndarray, path: str):
    with open(path, "w") as f:
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")  # OBJ is 1-indexed


def main():
    parser = argparse.ArgumentParser(description="Generate a body mesh from measurements.")
    for name in config.INPUT_NAMES:
        parser.add_argument(f"--{name}", type=float, required=True)
    parser.add_argument("--gender", choices=["male", "female", "neutral"], required=True)
    parser.add_argument("--tag", default=None, help="ensemble tag (default: base/shoulder from config)")
    parser.add_argument("--out", type=str, default=f"{config.OUTPUT_DIR}/generated_mesh.obj")
    args = parser.parse_args()

    measurements = {n: getattr(args, n) for n in config.INPUT_NAMES}
    verts, faces, info = measurements_to_mesh(measurements, args.gender, args.tag)
    save_obj(verts, faces, args.out)

    np.set_printoptions(precision=3, suppress=True)
    print(f"Gender: {args.gender}")
    print(f"Predicted betas: {info['betas']}")
    print(f"Beta std (total): {info['beta_std']}")
    for w in info["warnings"]:
        print(f"WARNING: {w}")
    print(f"Mesh saved to {args.out}")


if __name__ == "__main__":
    main()
