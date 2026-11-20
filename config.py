"""
Central configuration for the measurement -> beta -> mesh pipeline.
"""
import os

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Path to the pretrained SMPL model folder (must contain SMPL_NEUTRAL.pkl or
# similar, downloaded separately from https://smpl.is.tue.mpg.de/).
SMPL_MODEL_DIR = os.environ.get(
    "SMPL_MODEL_DIR",
    os.path.join(PROJECT_ROOT, "smpl_models")
)

# SMPL_MODEL_DIR = os.path.join(PROJECT_ROOT, "smpl_models", "smpl")

DEFAULT_GENDER = "neutral"

MODEL_CHECKPOINTS = {
    "male": os.path.join(MODEL_DIR, "mlp_male.pt"),
    "female": os.path.join(MODEL_DIR, "mlp_female.pt"),
    "neutral": os.path.join(MODEL_DIR, "mlp_neutral.pt"),
}

DATASETS = {
    "male": os.path.join(DATA_DIR, "synthetic_male.csv"),
    "female": os.path.join(DATA_DIR, "synthetic_female.csv"),
    "neutral": os.path.join(DATA_DIR, "synthetic_neutral.csv"),
}

SCALERS = {
    "male": os.path.join(MODEL_DIR, "mlp_male_scaler.pkl"),
    "female": os.path.join(MODEL_DIR, "mlp_female_scaler.pkl"),
    "neutral": os.path.join(MODEL_DIR, "mlp_neutral_scaler.pkl"),
}

TEST_SPLITS = {
    "male": os.path.join(MODEL_DIR, "mlp_male_test.npz"),
    "female": os.path.join(MODEL_DIR, "mlp_female_test.npz"),
    "neutral": os.path.join(MODEL_DIR, "mlp_neutral_test.npz"),
}

# ----------------------------------------------------------------------
# Synthetic data generation
# ----------------------------------------------------------------------
N_BETAS = 10
N_SAMPLES = 20000

# Beta sampling: standard normal per component, clipped to +-BETA_CLIP.
# (SMPL betas are PCA coefficients in units of std-dev, so N(0,1) is the
# natural prior.)
BETA_CLIP = 3.0

# LEGACY (no longer used by the generator): old decaying per-beta std-devs.
BETA_STD_DECAY = [2.0, 1.6, 1.3, 1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.25]
assert len(BETA_STD_DECAY) == N_BETAS

# Pose used when generating synthetic meshes. BodyM subjects stand in an
# A-pose (arms ~30 deg from the body, feet apart). "t_pose" = SMPL zero pose.
POSE_MODE = "a_pose"          # "a_pose" or "t_pose"
A_POSE_ARM_ANGLE_DEG = 30.0   # angle of each arm away from the torso
A_POSE_LEG_SPREAD_DEG = 5.0   # extra hip abduction per leg; tune after render check
A_POSE_ELBOW_BEND_DEG = 0.0   # optional; leave 0 unless the render looks stiff

# Generation in chunks (for Colab / resumable runs).
CHUNK_SIZE = 2000
MESH_BATCH_SIZE = 100         # meshes per SMPL forward pass

# Gaussian noise added to synthetic measurements (cm / kg). 0.0 disables.
MEASUREMENT_NOISE_STD = {
    "chest": 0.5,
    "waist": 0.5,
    "hips": 0.5,
    "height": 0.3,
    "weight": 0.3,
    "biacromial": 0.3,
    "thigh": 0.4,
}

# Everything the generator writes to the CSV (bicristal dropped: no BodyM
# equivalent).
MEASUREMENT_NAMES = [
    "height", "weight", "chest", "waist", "hips", "thigh", "biacromial",
]

# ----- Common input set (what the MLP actually receives) -----
# biacromial <-> BodyM "shoulder-breadth" is only a proxy, so it is a flag.
# Train with and without it and compare.
USE_SHOULDER_BREADTH = False
INPUT_NAMES = ["height", "weight", "chest", "waist", "hips", "thigh"] + (
    ["biacromial"] if USE_SHOULDER_BREADTH else []
)

# Mapping from our names to BodyM columns (measurements.csv unless noted).
BODYM_COLUMN_MAP = {
    "height": "height",          # scan head-to-heel height = model input
    "weight": None,              # hwg_metadata.csv: weight_kg
    "chest": "chest",
    "waist": "waist",
    "hips": "hip",
    "thigh": "thigh",
    "biacromial": "shoulder-breadth",
}
# height_cm (hwg_metadata.csv) is used only for silhouette scaling.
HEIGHT_MISMATCH_FLAG_CM = 5.0  # flag subjects where |scan height - height_cm| exceeds this

# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------
TRAIN_VAL_TEST_SPLIT = (0.8, 0.1, 0.1)
BATCH_SIZE = 128
EPOCHS = 200
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
DROPOUT = 0.1
HIDDEN_DIMS = [64, 128, 64]

# Relative weighting between vertex loss and beta loss in the combined
# training objective. Vertex loss reflects visual/volumetric accuracy;
# beta loss stabilizes training early on.
VERTEX_LOSS_WEIGHT = 1.0
BETA_LOSS_WEIGHT = 0.1

# Per-beta weights for the weighted beta MSE (dominant/early betas weighted
# higher, since they move more vertices). Uniform now that all betas share
# the same N(0,1) prior; revisit in Stage 2 if desired.
BETA_LOSS_WEIGHTS = [1.0] * N_BETAS  # betas are N(0,1) now, so equal weights

DEVICE = "cpu"  # set to "cuda" if a GPU is available
RANDOM_SEED = 42


# ===== Stage 2: paste at the END of config.py =====
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))

# Ensemble / uncertainty
ENSEMBLE_SIZE = 5
ENSEMBLE_DIR = _os.path.join(_HERE, "models", "ensembles")
LOGVAR_MIN = -8.0            # soft clamp on predicted log-variance
LOGVAR_MAX = 4.0
NLL_WARMUP_EPOCHS = 5        # MSE-only epochs before the variance head is trained
EARLY_STOP_PATIENCE = 30     # epochs without val-NLL improvement

# Data locations
SYNTHETIC_CSV = {g: _os.path.join(_HERE, "data", f"synthetic_{g}.csv")
                 for g in ("male", "female", "neutral")}
BODYM_DIR = _os.path.join(_HERE, "data", "bodym")

# Plausibility filter (drops synthetic rows outside BodyM-train ranges; never uses biacromial)
USE_PLAUSIBILITY_FILTER = True   # train.py uses --no-filter to override
PLAUSIBILITY_QUANTILES = (0.005, 0.995)
PLAUSIBILITY_COLUMNS = ["height", "weight", "chest", "waist", "hips", "thigh"]
PLAUSIBILITY_RANGES_PATH = _os.path.join(_HERE, "models", "plausibility_ranges.json")

# The UI needs a real default now that there is no single neutral model
DEFAULT_GENDER = "male"
