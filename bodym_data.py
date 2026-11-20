"""BodyM loader: joins metadata, measurements and front/side masks.

Run from the project root (venv active):  python bodym_data.py
Needs: numpy, pandas, pillow
"""
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path("data/bodym")
SPLITS = ("train", "testA", "testB")

# Columns in measurements.csv (14 body measurements, "height" = head-to-heel)
MEASUREMENT_COLS = [
    "ankle", "arm-length", "bicep", "calf", "chest", "forearm", "height",
    "hip", "leg-length", "shoulder-breadth", "shoulder-to-crotch",
    "thigh", "waist", "wrist",
]


def load_split(split, root=ROOT):
    """One row per photo: subject info, 14 measurements, front/side mask paths."""
    d = Path(root) / split
    photos = pd.read_csv(d / "subject_to_photo_map.csv")
    hwg = pd.read_csv(d / "hwg_metadata.csv")
    meas = pd.read_csv(d / "measurements.csv")

    df = (
        photos.merge(hwg, on="subject_id", how="left")
        .merge(meas, on="subject_id", how="left")
    )
    df["front_path"] = [d / "mask" / f"{p}.png" for p in df["photo_id"]]
    df["side_path"] = [d / "mask_left" / f"{p}.png" for p in df["photo_id"]]
    df["split"] = split
    return df


def load_mask(path):
    """Binary (H, W) bool array. Masks are stored as 0/255 PNGs."""
    return np.array(Image.open(path).convert("L")) > 127


def check(split):
    """Print sanity checks for one split."""
    d = ROOT / split
    df = load_split(split)
    print(f"\n== {split} ==")
    print("photos:", len(df), "| subjects:", df["subject_id"].nunique())
    per_subject = df.drop_duplicates("subject_id")
    print("gender (per subject):", per_subject["gender"].value_counts().to_dict())
    print(
        "rows missing hwg metadata:", int(df["height_cm"].isna().sum()),
        "| rows missing measurements:", int(df["chest"].isna().sum()),
    )

    front_ok = df["front_path"].map(Path.exists)
    side_ok = df["side_path"].map(Path.exists)
    print("missing masks (front, side):", int((~front_ok).sum()), int((~side_ok).sum()))
    print(
        "mask files on disk (front, side):",
        len(list((d / "mask").glob("*.png"))),
        len(list((d / "mask_left").glob("*.png"))),
    )

    diff = (df["height_cm"] - df["height"]).abs()
    print(
        "height_cm (hwg) vs scan height: mean abs diff",
        round(float(diff.mean()), 2), "| max", round(float(diff.max()), 2),
    )

    shapes = set()
    for p in df.loc[front_ok & side_ok, "front_path"].head(50):
        shapes.add(load_mask(p).shape)
    for p in df.loc[front_ok & side_ok, "side_path"].head(50):
        shapes.add(load_mask(p).shape)
    print("mask shapes seen in first 50 pairs (H, W):", shapes)


if __name__ == "__main__":
    for s in SPLITS:
        check(s)
