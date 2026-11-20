"""
Paired comparison of each model against the mean-shape baseline, from a Stage 4 subjects CSV.

For every subject the difference (model - mean_shape) in IoU is computed, so subject difficulty
cancels out. Reports the mean paired difference, a bootstrap 95% CI (over subjects) and the share
of subjects where the model wins. For girth error the sign is flipped (positive = model better).

Run from the project root:
  python analyze_paired.py                       # latest outputs/stage4_*_subjects.csv
  python analyze_paired.py --csv outputs\\stage4_testA-testB_base_subjects.csv --include-flagged
Output: printed table + outputs/stage4_paired.csv
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

import config

METRICS = [("iou_front", +1), ("iou_side", +1), ("iou_mean", +1), ("girth_mae", -1)]


def boot_ci(d, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    d = np.asarray(d, dtype=float)
    d = d[~np.isnan(d)]
    if len(d) < 2:
        return (np.nan, np.nan)
    means = rng.choice(d, (n, len(d))).mean(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--include-flagged", action="store_true")
    ap.add_argument("--baseline", default="mean_shape",
                    help="model to compare against (e.g. sweep_n1 = height-only)")
    args = ap.parse_args()

    path = args.csv
    if path is None:
        files = sorted(glob.glob(os.path.join(config.OUTPUT_DIR, "stage4_*_subjects.csv")),
                       key=os.path.getmtime)
        if not files:
            raise SystemExit("No stage4_*_subjects.csv found in outputs/")
        path = files[-1]
    print("reading", path)
    res = pd.read_csv(path)
    if not args.include_flagged:
        res = res[~res["flagged"]]

    base = res[res["model"] == args.baseline]
    if base.empty:
        raise SystemExit(f"baseline {args.baseline!r} not in the CSV (models: {sorted(set(res['model']))})")
    keys = ["split", "gender", "subject_id"]
    rows = []
    for model in sorted(set(res["model"]) - {args.baseline}):
        m = res[res["model"] == model].merge(base, on=keys, suffixes=("", "_ms"))
        for gname, g in list(m.groupby(["split", "gender"])) + [
                ((s, "all"), gg) for s, gg in m.groupby("split")]:
            split, gender = gname
            row = dict(split=split, model=model, gender=gender, n=len(g))
            for metric, sign in METRICS:
                d = sign * (g[metric] - g[metric + "_ms"])
                lo, hi = boot_ci(d)
                row[f"{metric}_gain"] = d.mean()
                row[f"{metric}_ci"] = f"[{lo:+.3f}, {hi:+.3f}]"
                row[f"{metric}_win%"] = 100 * float((d > 0).mean())
            rows.append(row)
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 50)
    show = ["split", "model", "gender", "n"] + [f"{m}_{s}" for m, _ in METRICS for s in ("gain", "ci", "win%")]
    print(out[show].round(3).to_string(index=False))
    name = "stage4_paired.csv" if args.baseline == "mean_shape" else f"stage4_paired_vs_{args.baseline}.csv"
    out.to_csv(os.path.join(config.OUTPUT_DIR, name), index=False)
    print(f"\nSaved outputs/{name}  (gain > 0 = better than {args.baseline})")


if __name__ == "__main__":
    main()
