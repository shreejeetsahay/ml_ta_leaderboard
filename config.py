import os
from pathlib import Path
import torch
# Device
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_ROOT = BASE_DIR / "data"
SUBMISSION_DIR = BASE_DIR / "submissions"
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
MIN_LATENT_DIM = 8  # Minimum latent dimension for acceptance
# Limits / Rules
MAX_MODEL_SIZE = 23.0  # MB (hard limit)
RATE_LIMIT_MINUTES = 15  # minutes between submissions

# Admin
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "MySecretPassword")

# Evaluation
IMG_SIZE = 256
BATCH_SIZE = 64

CACHE_DIR =BASE_DIR/"cache"  # <-- NEW: preprocessed tensors & masks
os.makedirs(CACHE_DIR, exist_ok=True)



# Changes suggested
# --- Scoring caps (fixed, published) ---
SCORE_CAPS = {
    "latent_dim":   {"min": 8,     "max": 256},   # lower is better
    "full_mse":     {"min": 5e-4,  "max": 5e-2},  # lower is better
    "roi_mse":      {"min": 2e-3,  "max": 8e-2},  # lower is better
    "model_size":   {"min": 1.0,   "max": 23.0},  # MB, lower is better
}

# Use fixed caps (recommended) vs dynamic cohort min/max
USE_DYNAMIC_NORMALIZATION = False   # if True, compute min/max from current cohort each render

# We still rank by the tie-break order; composite is display-only
RANK_BY = "composite"   # "tie" (recommended) or "composite"

