"""
Compare synthetic datasets with BodyM (train split) per gender.

Reports, per measurement:
  - percentiles (1 / 50 / 99) of both datasets
  - bodym_outside_synth_%: share of real subjects outside the synthetic [min, max]
    (the important one: values the model never saw during training)
  - synth_outside_bodym_%: share of synthetic rows outside BodyM's [0.5%, 99.5%]
    range (synthetic bodies more extreme than any real subject)

Run from the project root:  python check_coverage.py
"""
import os
import pandas as pd
import config

BODYM_DIR = os.path.join(config.DATA_DIR, "bodym", "train")
# our name -> (BodyM column, which BodyM file)
MAP = {
    "height": ("height", "m"), "weight": ("weight_kg", "h"), "chest": ("chest", "m"),
    "waist": ("waist", "m"), "hips": ("hip", "m"), "thigh": ("thigh", "m"),
}

m = pd.read_csv(os.path.join(BODYM_DIR, "measurements.csv"))
h = pd.read_csv(os.path.join(BODYM_DIR, "hwg_metadata.csv"))
real = m.merge(h, on="subject_id")

rows = []
for gender in ("male", "female"):
    syn = pd.read_csv(config.DATASETS[gender])
    r = real[real["gender"] == gender]
    for name, (col, _) in MAP.items():
        s, b = syn[name], r[col]
        lo, hi = b.quantile(0.005), b.quantile(0.995)
        rows.append({
            "gender": gender, "measure": name,
            "bodym_p1": b.quantile(0.01), "synth_p1": s.quantile(0.01),
            "bodym_p50": b.median(), "synth_p50": s.median(),
            "bodym_p99": b.quantile(0.99), "synth_p99": s.quantile(0.99),
            "bodym_std": b.std(), "synth_std": s.std(),
            "bodym_outside_synth_%": 100 * ((b < s.min()) | (b > s.max())).mean(),
            "synth_outside_bodym_%": 100 * ((s < lo) | (s > hi)).mean(),
        })
out = pd.DataFrame(rows).round(1)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 50)
print(out.to_string(index=False))
out.to_csv(os.path.join(config.OUTPUT_DIR, "coverage_check.csv"), index=False)
print("\nSaved outputs/coverage_check.csv")
