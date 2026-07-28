from pathlib import Path
import os
import torch


# ---------------------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
CACHE_ROOT = PROJECT_ROOT / "persistent_cache"

# ---------------------------------------------------------------------
# EXPERIMENT SELECTION
# ---------------------------------------------------------------------

DATASET = os.environ.get("BRATS_DATASET", "2023")
MODEL_NAME = os.environ.get("BRATS_MODEL", "swin_unetr")


# ---------------------------------------------------------------------
# DATASET-SPECIFIC CONFIGURATION
# ---------------------------------------------------------------------

DATASET_CONFIGS = {
    "2023": {
        "train_dirs": [
            DATA_DIR / "brats_train_val_2023" / "BraTS-MEN-Train",
        ],
        "in_channels": 4,
        "out_channels": 4,
        "n_folds": 5,
        "inner_val_ratio": 0.10,
    },
    "2024": {
        "train_dirs": [
            DATA_DIR / "training_data1_v2",
            DATA_DIR / "training_data_additional",
        ],
        "in_channels": 4,
        "out_channels": 5,
        "n_folds": 5,
        "inner_val_ratio": 0.10,
    },
}


if DATASET not in DATASET_CONFIGS:
    raise ValueError(
        f"Unsupported dataset: {DATASET}. "
        f"Available datasets: {list(DATASET_CONFIGS)}"
    )


DATASET_CONFIG = DATASET_CONFIGS[DATASET]

TRAIN_DIRS = DATASET_CONFIG["train_dirs"]
IN_CHANNELS = DATASET_CONFIG["in_channels"]
OUT_CHANNELS = DATASET_CONFIG["out_channels"]

N_FOLDS = DATASET_CONFIG["n_folds"]
INNER_VAL_RATIO = DATASET_CONFIG["inner_val_ratio"]


# ---------------------------------------------------------------------
# HARDWARE AND REPRODUCIBILITY
# ---------------------------------------------------------------------

MODEL_NAME = MODEL_NAME.lower()

CUDA_REQUIRED_MODELS = {
    "segmamba",
    "segmambav2",
}

if MODEL_NAME in CUDA_REQUIRED_MODELS:
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"{MODEL_NAME} requires CUDA, but no GPU is available."
        )

    DEVICE = torch.device("cuda")

else:
    DEVICE = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


SEED = 42


# ---------------------------------------------------------------------
# DATASET CONFIGURATION
# ---------------------------------------------------------------------

LABEL_SUFFIX = "seg"

ROI_SIZE = (128, 128, 128)
SPACING = (1.0, 1.0, 1.0)


# ---------------------------------------------------------------------
# TRAINING HYPERPARAMETERS
# ---------------------------------------------------------------------

BATCH_SIZE = 1
VAL_BATCH_SIZE = 1

MAX_EPOCHS = 100
VAL_EVERY = 5

WEIGHT_DECAY = 1e-5


MODEL_TRAINING_CONFIGS = {
    "unet3d": {
        "learning_rate": 1e-4,
        "clip_grad": False,
        "grad_clip_max_norm": None,
        "num_workers": 4,
    },
    "resunet3d": {
        "learning_rate": 1e-4,
        "clip_grad": False,
        "grad_clip_max_norm": None,
        "num_workers": 4,
    },
    "segresnet": {
        "learning_rate": 1e-4,
        "clip_grad": False,
        "grad_clip_max_norm": None,
        "num_workers": 4,
    },
    "swin_unetr": {
        "learning_rate": 1e-4,
        "clip_grad": False,
        "grad_clip_max_norm": None,
        "num_workers": 4,
    },
    "segmamba": {
        "learning_rate": 1e-5,
        "clip_grad": True,
        "grad_clip_max_norm": 1.0,
        "num_workers": 2,
    },
    "segmambav2": {
        "learning_rate": 1e-5,
        "clip_grad": True,
        "grad_clip_max_norm": 1.0,
        "num_workers": 2,
    },
}


if MODEL_NAME not in MODEL_TRAINING_CONFIGS:
    raise ValueError(
        f"Unsupported model: {MODEL_NAME}. "
        f"Available models: {list(MODEL_TRAINING_CONFIGS)}"
    )


MODEL_CONFIG = MODEL_TRAINING_CONFIGS[MODEL_NAME]

LEARNING_RATE = MODEL_CONFIG["learning_rate"]
CLIP_GRAD = MODEL_CONFIG["clip_grad"]
GRAD_CLIP_MAX_NORM = MODEL_CONFIG["grad_clip_max_norm"]
NUM_WORKERS = MODEL_CONFIG["num_workers"]


# ---------------------------------------------------------------------
# MEMORY AND PERFORMANCE
# ---------------------------------------------------------------------

SW_BATCH_SIZE = 2
USE_CHECKPOINT = True


# ---------------------------------------------------------------------
# CACHE
# ---------------------------------------------------------------------

CACHE_VERSION = "v1"

CACHE_NAME = (
    f"brats{DATASET}_{CACHE_VERSION}"
    f"_roi{ROI_SIZE[0]}x{ROI_SIZE[1]}x{ROI_SIZE[2]}"
    f"_sp{SPACING[0]}_{SPACING[1]}_{SPACING[2]}"
)

PERSISTENT_CACHE_DIR = CACHE_ROOT / CACHE_NAME


# ---------------------------------------------------------------------
# EXPERIMENT NAME
# ---------------------------------------------------------------------

EXPERIMENT_NAME = (
    f"brats{DATASET}_{MODEL_NAME}"
    f"_roi{ROI_SIZE[0]}"
    f"_bs{BATCH_SIZE}"
    f"_nworkers{NUM_WORKERS}"
    f"_cv{N_FOLDS}"
    f"_without_background"
)

EXPERIMENT_DIR = EXPERIMENTS_DIR / EXPERIMENT_NAME


