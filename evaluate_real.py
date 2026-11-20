"""
Stage 4: evaluate the measurement -> beta -> mesh pipeline on real BodyM subjects.

For every subject (aggregated over photos) of the chosen split(s):
  1. inputs = BodyM measurements (config.BODYM_COLUMN_MAP; weight from hwg_metadata)
  2. route to the male/female ensemble (+ matching SMPL), predict betas (clipped to +-BETA_CLIP)
  3. build the A-posed mesh (same pose as the training data)
  4. round-trip measurement error: measurements_v2 on the predicted mesh vs BodyM values
  5. silhouette IoU (front + side) vs the BodyM masks, averaged over the subject's photos
  6. uncertainty analysis (does predicted beta uncertainty rank the errors? is epistemic std an
     out-of-range flag?)
Baselines: "mean_shape" (betas = 0) always; extra ensembles via --tags (e.g. sweep_n1 = height only,
sweep_n7 / shoulder = with biacromial).

Run from the project root (venv active):
  python evaluate_real.py --splits testA --max-subjects 20 --overlay 3     # quick check first
  python evaluate_real.py --splits testA testB --tags base sweep_n1
  python evaluate_real.py --no-silhouette                                  # measurements only (fast)

Outputs: outputs/stage4_<splits>_<tags>_subjects.csv / _summary.json, outputs/overlays/*.png
"""
import smpl_compat  # noqa: F401  -- must be imported before smplx/chumpy

import argparse
import gc
import json
import os
import time
from functools import lru_cache

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

import config
from bodym_data import load_split, load_mask
from ensemble import BetaEnsemble
from generate_synthetic_data import make_body_pose, build_measurement_set
from silhouette import (largest_component, render_silhouette, iou, overlay_image,
                        make_render_pose)

import smplx

MEAS = ["height", "weight", "chest", "waist", "hips", "thigh", "biacromial"]
GIRTHS = ["chest", "waist", "hips", "thigh"]
MEAN_SHAPE = "mean_shape"


# ----------------------------------------------------------------------
# data
# ----------------------------------------------------------------------
def subject_table(split):
    df = load_split(split, root=config.BODYM_DIR)
    subj = df.drop_duplicates("subject_id").copy()
    subj["weight"] = subj["weight_kg"]
    for name, col in config.BODYM_COLUMN_MAP.items():
        if col is not None:
            subj[name] = subj[col]
    n0 = len(subj)
    subj = subj.dropna(subset=MEAS + ["height_cm"])
    subj = subj[subj["gender"].isin(["male", "female"])].copy()
    if len(subj) < n0:
        print(f"[{split}] dropped {n0 - len(subj)} subjects with missing data")
    subj["height_diff"] = (subj["height"] - subj["height_cm"]).abs()
    subj["flagged"] = subj["height_diff"] > config.HEIGHT_MISMATCH_FLAG_CM
    photos = {sid: list(zip(g["front_path"], g["side_path"])) for sid, g in df.groupby("subject_id")}
    print(f"[{split}] {len(subj)} subjects ({subj['gender'].value_counts().to_dict()}), "
          f"{int(subj['flagged'].sum())} flagged with |height - height_cm| > "
          f"{config.HEIGHT_MISMATCH_FLAG_CM} cm")
    return subj.reset_index(drop=True), photos


# ----------------------------------------------------------------------
# prediction + meshes
# ----------------------------------------------------------------------
@lru_cache(maxsize=None)
def smpl_for(gender, batch):
    return smplx.create(model_path=config.SMPL_MODEL_DIR, model_type="smpl",
                        gender=gender, num_betas=config.N_BETAS, batch_size=batch)


def betas_to_verts(gender, betas, body_pose=None, bs=20):
    """One SMPL model per gender (fixed batch size, last batch padded) to keep RAM low.
    body_pose: (69,) pose; default = the training/measurement pose from config."""
    if body_pose is None:
        body_pose = make_body_pose()
    pose = torch.tensor(np.asarray(body_pose, dtype=np.float32)).float()
    m = smpl_for(gender, bs)
    out = []
    for i in range(0, len(betas), bs):
        b = np.asarray(betas[i:i + bs], dtype=np.float32)
        n = len(b)
        if n < bs:
            b = np.concatenate([b, np.zeros((bs - n, b.shape[1]), dtype=np.float32)])
        with torch.no_grad():
            o = m(betas=torch.tensor(b), body_pose=pose.unsqueeze(0).repeat(bs, 1),
                  return_verts=True)
        out.append(o.vertices.numpy()[:n])
    return np.concatenate(out), m.faces


def predict_models(gender, subj_g, tags):
    """name -> dict(betas, unc, epi, out_of_range) for every model that exists for this gender."""
    n = len(subj_g)
    models = {MEAN_SHAPE: dict(betas=np.zeros((n, config.N_BETAS)),
                               unc=np.full(n, np.nan), epi=np.full(n, np.nan),
                               oor=np.zeros(n, dtype=bool))}
    for tag in tags:
        try:
            ens = BetaEnsemble(gender, tag)
        except FileNotFoundError:
            print(f"  (no ensemble {gender}_{tag}, skipped)")
            continue
        X = subj_g[ens.input_names].to_numpy(dtype=np.float32)
        p = ens.predict(X)
        oor = np.array([len(ens.range_warnings(dict(zip(ens.input_names, x)))) > 0 for x in X])
        models[tag] = dict(
            betas=np.clip(p["mean"], -config.BETA_CLIP, config.BETA_CLIP),
            unc=np.sqrt(p["total_var"].sum(1)),
            epi=np.sqrt(p["epistemic_var"].sum(1)),
            oor=oor)
    return models


# ----------------------------------------------------------------------
# main evaluation
# ----------------------------------------------------------------------
def evaluate_split(split, tags, do_sil, max_subjects, n_overlay, overlay_dir, render_pose=None,
                   max_photos=None):
    subj, photos = subject_table(split)
    if max_subjects:
        subj = subj.groupby("gender", group_keys=False).head(max_subjects).reset_index(drop=True)
    records = []
    for gender in ("male", "female"):
        sg = subj[subj["gender"] == gender].reset_index(drop=True)
        if sg.empty:
            continue
        print(f"[{split}] {gender}: {len(sg)} subjects")
        smpl_for.cache_clear()      # free the previous gender's SMPL model
        gc.collect()
        mset, _ = build_measurement_set(gender)
        gc.collect()
        models = predict_models(gender, sg, tags)
        meshes, meshes_r, meas_err = {}, {}, {}
        for name, md in models.items():
            verts, faces = betas_to_verts(gender, md["betas"])      # training pose: measurements
            meshes[name] = verts
            meshes_r[name] = (verts if render_pose is None           # render pose: silhouettes
                              else betas_to_verts(gender, md["betas"], render_pose)[0])
            errs = []
            for i in range(len(sg)):
                m = mset.measure(verts[i], faces)
                errs.append({f"err_{k}": m[k] - float(sg.loc[i, k]) for k in MEAS})
            meas_err[name] = errs

        sil = {name: {"iou_front": np.full(len(sg), np.nan), "iou_side": np.full(len(sg), np.nan)}
               for name in models}
        if do_sil:
            t0 = time.time()
            for i in range(len(sg)):
                acc = {name: {"front": [], "side": []} for name in models}
                for k, (fp, sp) in enumerate(photos[sg.loc[i, "subject_id"]][:max_photos]):
                    try:
                        masks = {"front": largest_component(load_mask(fp)),
                                 "side": largest_component(load_mask(sp))}
                    except FileNotFoundError:
                        continue
                    if not masks["front"].any() or not masks["side"].any():
                        continue
                    for name in models:
                        for view in ("front", "side"):
                            s = render_silhouette(meshes_r[name][i], faces, view, masks[view],
                                                  float(sg.loc[i, "height_cm"]))
                            acc[name][view].append(iou(masks[view], s))
                            if k == 0 and i < n_overlay and name != MEAN_SHAPE:
                                os.makedirs(overlay_dir, exist_ok=True)
                                overlay_image(masks[view], s).save(os.path.join(
                                    overlay_dir, f"{split}_{gender}_{sg.loc[i, 'subject_id']}_{name}_{view}.png"))
                for name in models:
                    for view in ("front", "side"):
                        if acc[name][view]:
                            sil[name][f"iou_{view}"][i] = float(np.mean(acc[name][view]))
                if (i + 1) % 25 == 0:
                    print(f"  silhouettes {i + 1}/{len(sg)} ({time.time() - t0:.0f}s)")

        for name, md in models.items():
            for i in range(len(sg)):
                rec = dict(split=split, gender=gender, model=name,
                           subject_id=sg.loc[i, "subject_id"],
                           flagged=bool(sg.loc[i, "flagged"]),
                           height_diff=float(sg.loc[i, "height_diff"]),
                           unc=md["unc"][i], epi=md["epi"][i], out_of_range=bool(md["oor"][i]),
                           iou_front=sil[name]["iou_front"][i], iou_side=sil[name]["iou_side"][i])
                rec.update(meas_err[name][i])
                records.append(rec)
    res = pd.DataFrame(records)
    res["iou_mean"] = res[["iou_front", "iou_side"]].mean(axis=1)
    res["girth_mae"] = res[[f"err_{k}" for k in GIRTHS]].abs().mean(axis=1)
    return res


# ----------------------------------------------------------------------
# summaries
# ----------------------------------------------------------------------
def summarize(res):
    rows = []
    for (split, model), g0 in res.groupby(["split", "model"]):
        for gname, g1 in list(g0.groupby("gender")) + [("all", g0)]:
            for subset, g in (("all", g1), ("excl_flagged", g1[~g1["flagged"]])):
                if g.empty:
                    continue
                row = dict(split=split, model=model, gender=gname, subset=subset, n=len(g),
                           iou_front=g["iou_front"].mean(), iou_side=g["iou_side"].mean(),
                           iou_mean=g["iou_mean"].mean(), girth_mae=g["girth_mae"].mean())
                for k in MEAS:
                    row[f"mae_{k}"] = g[f"err_{k}"].abs().mean()
                    row[f"bias_{k}"] = g[f"err_{k}"].mean()
                rows.append(row)
    return pd.DataFrame(rows)


def uncertainty_report(res):
    out = []
    for (split, model, gender), g in res[res["model"] != MEAN_SHAPE].groupby(["split", "model", "gender"]):
        g = g[~g["flagged"]]
        d = dict(split=split, model=model, gender=gender, n=len(g))
        if len(g) > 10:
            d["spearman_unc_vs_1-iou"] = spearmanr(g["unc"], 1 - g["iou_mean"], nan_policy="omit")[0]
            d["spearman_unc_vs_girth_mae"] = spearmanr(g["unc"], g["girth_mae"], nan_policy="omit")[0]
            try:
                q = pd.qcut(g["unc"], 5, labels=False, duplicates="drop")
                d["iou_by_unc_quintile"] = [round(float(x), 4) for x in g.groupby(q)["iou_mean"].mean()]
            except ValueError:
                pass
        d["n_out_of_range"] = int(g["out_of_range"].sum())
        d["epi_in_range"] = float(g.loc[~g["out_of_range"], "epi"].mean())
        d["epi_out_of_range"] = (float(g.loc[g["out_of_range"], "epi"].mean())
                                 if g["out_of_range"].any() else None)
        out.append(d)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["testA", "testB"])
    ap.add_argument("--tags", nargs="+", default=["base"])
    ap.add_argument("--max-subjects", type=int, default=None, help="per gender, for quick tests")
    ap.add_argument("--no-silhouette", action="store_true")
    ap.add_argument("--overlay", type=int, default=0, help="save overlays for the first N subjects per gender")
    ap.add_argument("--max-photos", type=int, default=None,
                    help="use at most this many photos per subject for the silhouette IoU (speed)")
    ap.add_argument("--render-pose", default=None,
                    help="JSON from calibrate_pose.py: pose used to DRAW silhouettes "
                         "(default: the training pose from config)")
    args = ap.parse_args()

    render_pose = None
    if args.render_pose:
        with open(args.render_pose) as f:
            rp = json.load(f)
        print("render pose:", {k: rp[k] for k in ("arm", "leg", "collar")})
        render_pose = make_render_pose(rp["arm"], rp["leg"], rp["collar"])

    overlay_dir = os.path.join(config.OUTPUT_DIR, "overlays")
    res = pd.concat([evaluate_split(s, args.tags, not args.no_silhouette, args.max_subjects,
                                    args.overlay, overlay_dir, render_pose,
                                    args.max_photos) for s in args.splits],
                    ignore_index=True)
    summ = summarize(res)
    unc = uncertainty_report(res)

    stem = f"stage4_{'-'.join(args.splits)}_{'-'.join(args.tags)}"
    res.to_csv(os.path.join(config.OUTPUT_DIR, stem + "_subjects.csv"), index=False)
    with open(os.path.join(config.OUTPUT_DIR, stem + "_summary.json"), "w") as f:
        json.dump(dict(summary=json.loads(summ.to_json(orient="records")), uncertainty=unc), f, indent=2)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 50)
    cols = ["split", "model", "gender", "subset", "n", "iou_front", "iou_side", "iou_mean", "girth_mae"]
    print("\n=== Silhouette IoU and mean girth error (cm) ===")
    print(summ[cols].round(3).to_string(index=False))
    mcols = ["split", "model", "gender"] + [f"mae_{k}" for k in MEAS]
    print("\n=== Round-trip MAE (cm / kg), all subjects ===")
    print(summ[summ["subset"] == "all"][mcols].round(2).to_string(index=False))
    bcols = ["split", "model", "gender"] + [f"bias_{k}" for k in MEAS]
    print("\n=== Round-trip signed bias (predicted - BodyM) ===")
    print(summ[summ["subset"] == "all"][bcols].round(2).to_string(index=False))
    print("\n=== Uncertainty analysis (flagged subjects excluded) ===")
    for d in unc:
        print(d)
    print(f"\nSaved outputs/{stem}_subjects.csv and _summary.json")


if __name__ == "__main__":
    main()
