"""
Compute per-gender plausibility ranges from BodyM *train* and write them to
config.PLAUSIBILITY_RANGES_PATH (JSON). train.py reads that file.

Run once:   python make_plausibility_ranges.py

Assumes  data\\bodym\\train\\measurements.csv  and  hwg_metadata.csv  both have a
`subject_id` column (as in the handoff). If a column name differs, the script
stops and prints the columns it found.
"""
import json
import os

import numpy as np
import pandas as pd

import config

# synthetic column -> BodyM column (weight comes from hwg_metadata.weight_kg)
COLUMN_MAP = {
    "height": "height",
    "weight": "weight_kg",
    "chest": "chest",
    "waist": "waist",
    "hips": "hip",
    "thigh": "thigh",
}


def main():
    d = os.path.join(config.BODYM_DIR, "train")
    meas = pd.read_csv(os.path.join(d, "measurements.csv"))
    hwg = pd.read_csv(os.path.join(d, "hwg_metadata.csv"))
    for need, df, nm in [("subject_id", meas, "measurements.csv"), ("subject_id", hwg, "hwg_metadata.csv")]:
        if need not in df.columns:
            raise SystemExit(f"{nm} has no '{need}' column. Columns: {list(df.columns)}")

    meas = meas.drop_duplicates("subject_id")
    hwg = hwg.drop_duplicates("subject_id")
    df = meas.merge(hwg[["subject_id", "gender", "weight_kg"]], on="subject_id", how="inner")
    df["gender"] = df["gender"].astype(str).str.lower()
    print(f"BodyM train subjects joined: {len(df)}")

    q_lo, q_hi = config.PLAUSIBILITY_QUANTILES
    out = {}
    for gender in ("male", "female"):
        g = df[df["gender"] == gender]
        out[gender] = {}
        print(f"\n{gender}: {len(g)} subjects (quantiles {q_lo}, {q_hi})")
        print(f"  {'col':8s} {'min':>8s} {'lo':>8s} {'median':>8s} {'hi':>8s} {'max':>8s}")
        for col in config.PLAUSIBILITY_COLUMNS:
            v = g[COLUMN_MAP[col]].astype(float).values
            lo, hi = float(np.quantile(v, q_lo)), float(np.quantile(v, q_hi))
            out[gender][col] = {"lo": lo, "hi": hi, "min": float(v.min()), "max": float(v.max())}
            print(f"  {col:8s} {v.min():8.1f} {lo:8.1f} {np.median(v):8.1f} {hi:8.1f} {v.max():8.1f}")

    os.makedirs(os.path.dirname(config.PLAUSIBILITY_RANGES_PATH), exist_ok=True)
    with open(config.PLAUSIBILITY_RANGES_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {config.PLAUSIBILITY_RANGES_PATH}")


if __name__ == "__main__":
    main()
