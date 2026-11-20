"""
Calibrate the silhouette RENDER pose on BodyM *train* subjects (never on test data).

Why: the overlays show the predicted body matches torso/hips/thighs well, but the mesh arms and legs
are posed differently from BodyM's A-pose (arm angle, leg spread, shoulder height). That pose
mismatch dominates the silhouette IoU. SMPL betas do not depend on pose, so we can draw the
predicted body in a pose fitted to BodyM without touching the trained models or the synthetic data.
(Round-trip measurements still use the original training pose.)

Method: coordinate descent over arm angle, leg spread and collar rotation, maximising the mean
FRONT-view IoU of the base ensemble's predictions on a sample of train subjects (both genders,
one pose shared by everyone).

Run from the project root (venv active):
  python calibrate_pose.py                 # ~20 subjects per gender, a few minutes
  python calibrate_pose.py --n 30
Output: outputs/render_pose.json   (then: python evaluate_real.py ... --render-pose outputs/render_pose.json)
"""
import smpl_compat  # noqa: F401  -- must be imported before smplx/chumpy

import argparse
import gc
import json
import os

import numpy as np

import config
from bodym_data import load_mask
from evaluate_real import subject_table, predict_models, betas_to_verts
from silhouette import largest_component, render_silhouette, iou, make_render_pose

CANDIDATES = {
    "arm": [10, 15, 20, 25, 30, 35],
    "leg": [-3, -1.5, 0, 1.5, 3, 5],
    "collar": [-10, -5, 0, 5, 10],
}
START = {"arm": 30.0, "leg": 5.0, "collar": 0.0}      # = the training pose


def build_sample(n_per_gender, tag, seed):
    rng = np.random.default_rng(seed)
    subj, photos = subject_table("train")
    sample = {}
    for gender in ("male", "female"):
        sg = subj[(subj["gender"] == gender) & (~subj["flagged"])]
        sg = sg.iloc[rng.permutation(len(sg))[:n_per_gender]].reset_index(drop=True)
        models = predict_models(gender, sg, [tag])
        if tag not in models:
            raise SystemExit(f"No ensemble {gender}_{tag}")
        masks, keep = [], []
        for i in range(len(sg)):
            fp, _ = photos[sg.loc[i, "subject_id"]][0]
            try:
                m = largest_component(load_mask(fp))
            except FileNotFoundError:
                continue
            if m.any():
                masks.append(m)
                keep.append(i)
        sample[gender] = dict(betas=models[tag]["betas"][keep], masks=masks,
                              height_cm=sg.loc[keep, "height_cm"].to_numpy(dtype=float))
        print(f"  {gender}: {len(keep)} calibration subjects")
    return sample


def objective(sample, params):
    pose = make_render_pose(params["arm"], params["leg"], params["collar"])
    vals = []
    for gender, d in sample.items():
        verts, faces = betas_to_verts(gender, d["betas"], pose)
        for v, m, h in zip(verts, d["masks"], d["height_cm"]):
            vals.append(iou(m, render_silhouette(v, faces, "front", m, float(h))))
    return float(np.nanmean(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="subjects per gender")
    ap.add_argument("--tag", default="base")
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("building calibration sample from BodyM train ...")
    sample = build_sample(args.n, args.tag, args.seed)
    gc.collect()

    params = dict(START)
    best = objective(sample, params)
    print(f"start (training pose) {params}: front IoU {best:.4f}")
    baseline = best
    for p in range(args.passes):
        for key, cands in CANDIDATES.items():
            scores = {}
            for c in cands:
                trial = dict(params, **{key: float(c)})
                scores[c] = objective(sample, trial)
                print(f"  pass {p + 1}  {key}={c:>5}: {scores[c]:.4f}")
            c_best = max(scores, key=scores.get)
            if scores[c_best] > best:
                best = scores[c_best]
                params[key] = float(c_best)
            print(f"-> {key} = {params[key]}  (IoU {best:.4f})")

    out = dict(params, front_iou_calibration=best, front_iou_training_pose=baseline,
               n_per_gender=args.n, tag=args.tag, split="train")
    path = os.path.join(config.OUTPUT_DIR, "render_pose.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nbest render pose {params}: front IoU {baseline:.4f} -> {best:.4f}")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
