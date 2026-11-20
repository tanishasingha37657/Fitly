"""
Synthetic silhouette-IoU reference: what IoU does the pipeline reach when ONLY the shape
prediction can be wrong?

Fresh synthetic bodies (new random betas, same measurement noise as training, inputs inside the
first tag's training range) are drawn; the TRUE mesh silhouette plays the role of the BodyM mask.
For every model (mean-shape body + each ensemble tag) we compute IoU(model body, true body)
with exactly the same renderer, render pose and centroid alignment as evaluate_real.py.

Comparing with the real-data IoU shows how much of the real-data gap to 1.0 is NOT due to the
shape prediction (hair, head posture, perspective, registration, pose), and whether the IoU gain
of the 6-input model over a height-only model (sweep_n1) is as large on real data as it is here.

Run from the project root (venv active):
  python synthetic_ceiling.py --render-pose outputs\\render_pose.json
  python synthetic_ceiling.py --render-pose outputs\\render_pose.json --tags base sweep_n1 sweep_n7
Output: printed table + outputs/synthetic_iou_reference.json
"""
import smpl_compat  # noqa: F401  -- must be imported before smplx/chumpy

import argparse
import gc
import json
import os

import numpy as np

import config
from ensemble import BetaEnsemble
from generate_synthetic_data import build_measurement_set, sample_betas, add_measurement_noise
from evaluate_real import betas_to_verts
from silhouette import make_render_pose, render_silhouette, iou

H, W = 960, 720


def fake_mask():
    """A one-pixel-wide full-height 'mask': only its bbox and centre column are used for scaling."""
    m = np.zeros((H, W), dtype=bool)
    m[20:941, W // 2] = True
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="bodies per gender")
    ap.add_argument("--tags", nargs="+", default=["base", "sweep_n1", "sweep_n7"])
    ap.add_argument("--render-pose", default=None)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    pose_r = None
    if args.render_pose:
        with open(args.render_pose) as f:
            rp = json.load(f)
        pose_r = make_render_pose(rp["arm"], rp["leg"], rp["collar"])
        print("render pose:", {k: rp[k] for k in ("arm", "leg", "collar")})

    mask = fake_mask()
    results = {}
    for gender in ("male", "female"):
        rng = np.random.default_rng([args.seed, 0 if gender == "male" else 1])
        mset, _ = build_measurement_set(gender)
        ens = {}
        for t in args.tags:
            try:
                ens[t] = BetaEnsemble(gender, t)
            except FileNotFoundError:
                print(f"  (no ensemble {gender}_{t}, skipped)")
        first = ens[next(iter(ens))]          # its input ranges define the in-range filter

        betas = sample_betas(rng, 4 * args.n).astype(np.float32)
        v_train, faces = betas_to_verts(gender, betas)                 # training pose -> measurements
        meas = [add_measurement_noise(mset.measure(v, faces), rng) for v in v_train]
        del v_train
        gc.collect()

        def inputs(e):
            return np.array([[m[k] for k in e.input_names] for m in meas], dtype=np.float32)

        Xf = inputs(first)
        keep = [i for i in range(len(Xf))
                if not first.range_warnings(dict(zip(first.input_names, Xf[i])))][:args.n]
        print(f"{gender}: {len(keep)} in-range synthetic bodies")

        v_true, _ = betas_to_verts(gender, betas[keep], pose_r)
        verts_by_model = {"mean_shape": betas_to_verts(gender, np.zeros((len(keep), config.N_BETAS), np.float32), pose_r)[0]}
        for t, e in ens.items():
            pred = np.clip(e.predict(inputs(e)[keep])["mean"], -config.BETA_CLIP, config.BETA_CLIP)
            verts_by_model[t] = betas_to_verts(gender, pred, pose_r)[0]
        heights = (v_true[:, 1].max(axis=1) - v_true[:, 1].min(axis=1)) * 100.0

        acc = {(name, view): [] for name in verts_by_model for view in ("front", "side")}
        for i in range(len(keep)):
            for view in ("front", "side"):
                s_true = render_silhouette(v_true[i], faces, view, mask, float(heights[i]))
                for name, vm in verts_by_model.items():
                    s = render_silhouette(vm[i], faces, view, mask, float(heights[i]))
                    acc[(name, view)].append(iou(s_true, s))
        results[gender] = {"n": len(keep)}
        for name in verts_by_model:
            fr = float(np.nanmean(acc[(name, "front")]))
            sd = float(np.nanmean(acc[(name, "side")]))
            results[gender][name] = {"front": fr, "side": sd, "mean": (fr + sd) / 2}

    print("\n=== Synthetic IoU vs the TRUE synthetic silhouette (shape error only) ===")
    print(f"{'gender':8s} {'model':12s} {'front':>7s} {'side':>7s} {'mean':>7s}")
    for g, r in results.items():
        for name, v in r.items():
            if name == "n":
                continue
            print(f"{g:8s} {name:12s} {v['front']:7.3f} {v['side']:7.3f} {v['mean']:7.3f}")
    path = os.path.join(config.OUTPUT_DIR, "synthetic_iou_reference.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
