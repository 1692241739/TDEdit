"""Portable path configuration; set environment variables before importing TDEdit.

Defaults live under this checkout, not the caller's working directory. Model
weights, upstream depth repositories and datasets are obtained separately.
Algorithm settings deliberately do not belong in this module.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def configured_path(name, default):
    """Expand an explicit filesystem path without modifying the environment."""
    return str(Path(os.environ.get(name) or default).expanduser().resolve())


DATA_ROOT = configured_path("TDEDIT_DATA_ROOT", PROJECT_ROOT / "data")
OUTPUT_ROOT = configured_path("TDEDIT_OUTPUT_ROOT", PROJECT_ROOT / "outputs")
MODEL_PATH = os.environ.get("TDEDIT_MODEL_PATH") or str(PROJECT_ROOT / "checkpoints" / "LCM_Dreamshaper_v7")
DRAGBENCH_ROOT = configured_path("TDEDIT_DRAGBENCH_ROOT", Path(DATA_ROOT) / "DragBench")
PIEBENCH_ROOT = configured_path("TDEDIT_PIEBENCH_ROOT", Path(DATA_ROOT) / "PIE-Bench")
EVAL_SD_PATH = os.environ.get("TDEDIT_EVAL_SD_PATH") or str(PROJECT_ROOT / "checkpoints" / "stable-diffusion-2-1")
EVAL_SCRIPT_DIR = configured_path("TDEDIT_EVAL_SCRIPT_DIR", PROJECT_ROOT / "run_evaluations")
SAM2_CHECKPOINT = configured_path("TDEDIT_SAM2_CHECKPOINT", PROJECT_ROOT / "checkpoints" / "sam2" / "sam2.1_hiera_large.pt")
SAM2_CONFIG = configured_path("TDEDIT_SAM2_CONFIG", PROJECT_ROOT / "configs" / "sam2_hiera_l.yaml")
DEPTH_V2_REPO = configured_path("TDEDIT_DEPTH_V2_REPO", PROJECT_ROOT / "third_party" / "Depth-Anything-V2")
DEPTH_V2_CHECKPOINT_DIR = configured_path("TDEDIT_DEPTH_V2_CHECKPOINT_DIR", Path(DEPTH_V2_REPO) / "checkpoints")
DEPTH_V1_REPO = configured_path("TDEDIT_DEPTH_V1_REPO", PROJECT_ROOT / "third_party" / "Depth-Anything")
DEPTH_V1_CHECKPOINT = configured_path("TDEDIT_DEPTH_V1_CHECKPOINT", Path(DEPTH_V1_REPO) / "checkpoints" / "depth_anything_vitb14.pth")
META_DIR = configured_path("TDEDIT_META_DIR", Path(PIEBENCH_ROOT) / "meta_file")
PROCESS_ROOT = configured_path("TDEDIT_PROCESS_ROOT", Path(OUTPUT_ROOT) / "process_images")
