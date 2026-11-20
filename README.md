# Body Measurement → SMPL Beta Pipeline

Implements the pipeline: synthetic data generation (via SMPL) → MLP regressor
(measurements → betas) → SMPL mesh reconstruction, with vertex-loss training,
weighted beta loss, and MPVE/beta/measurement evaluation.

## What you need to supply

This code assumes you already have:
1. A pretrained SMPL model file (e.g. `SMPL_NEUTRAL.pkl` / `.npz`), obtained
   from https://smpl.is.tue.mpg.de/ under their license. **Not included here**
   — you must register and download it yourself.
2. The `smplx` Python package (a maintained wrapper around SMPL/SMPL-X) to
   load and run the body model. Install with:
   ```
   pip install smplx torch numpy scikit-learn --break-system-packages
   ```

## Files

- `src/measurements.py` — extracts the 8 anthropometric measurements from a
  SMPL mesh's vertices, given landmark vertex indices.
- `src/generate_synthetic_data.py` — samples betas, runs them through SMPL,
  extracts measurements, saves (measurements, betas) pairs to CSV.
- `src/model.py` — the MLP (measurements → betas) architecture.
- `src/losses.py` — weighted beta MSE loss + SMPL vertex loss.
- `src/train.py` — training loop combining both losses.
- `src/evaluate.py` — MPVE, beta MSE, per-measurement error, per-beta error
  breakdown.
- `src/inference.py` — end-to-end demo: measurements in → mesh out.
- `src/config.py` — central config (paths, hyperparameters, landmark indices).

## IMPORTANT: landmark vertex indices

`src/measurements.py` contains a `LANDMARKS` dict of SMPL vertex indices used
to compute circumferences/lengths. The indices included are commonly cited
approximate landmarks from SMPL-measurement literature, but **you should
verify them yourself** against your actual SMPL topology (visualize the
vertices with the provided `debug_plot_landmarks()` helper) before trusting
the numbers for your report. This is exactly the kind of thing to mention
explicitly if asked in a viva — the measurement extraction depends on landmark
calibration, which is a modeling choice, not something SMPL gives you for free.

## Running the pipeline

```bash
cd src
python generate_synthetic_data.py     # writes data/synthetic_dataset.csv
python train.py                        # trains MLP, saves models/mlp.pt
python evaluate.py                     # prints MPVE / beta MSE / per-measurement error
python inference.py --chest 96 --waist 82 --hips 100 --height 172 \
                     --weight 68 --biacromial 38 --bicristal 27 --thigh 56
```
