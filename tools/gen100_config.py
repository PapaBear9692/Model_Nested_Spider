"""
Configuration for 100 PTM Prototype Generation Pipeline.

Contains dataset definitions, model lists, paths, and utility functions
shared across the pipeline scripts.
"""

import os
import re
import json
from pathlib import Path
from datetime import datetime

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "required_files" / "data"
ORIGINAL_PROTO_DIR = DATA_DIR / "implclproto"
DEFAULT_OUTPUT_DIR = DATA_DIR / "implclproto_100"

# =============================================================================
# TEST DATASETS (9 datasets used for evaluation)
# =============================================================================

# Mapping: dataset short name -> (display_name, num_classes, torchvision_dataset_class, auto_download)
TEST_DATASETS = {
    "CIFAR10":    {"num_classes": 10,  "auto_download": True,  "input_size": 32},
    "CIFAR100":   {"num_classes": 100, "auto_download": True,  "input_size": 32},
    "Caltech101": {"num_classes": 101, "auto_download": True,  "input_size": 224},
    "DTD":        {"num_classes": 47,  "auto_download": True,  "input_size": 224},
    "Pet":        {"num_classes": 37,  "auto_download": True,  "input_size": 224},
    "Aircraft":   {"num_classes": 100, "auto_download": False, "input_size": 224},
    "Cars":       {"num_classes": 196, "auto_download": False, "input_size": 224},
    "SUN397":     {"num_classes": 397, "auto_download": True,  "input_size": 224},
    "dSprites":   {"num_classes": 1,   "auto_download": False, "input_size": 64},
}

# Original directory names for test datasets (in implclproto)
TEST_DATASET_DIRS = {
    "CIFAR10":    "CIFAR10_w10s5000_seed1_hh",
    "CIFAR100":   "CIFAR100_w100s500_seed1_hh",
    "Caltech101": "Caltech101_w101s69_seed1_hh",
    "DTD":        "DTD_w47s80_seed1_hh",
    "Pet":        "Pet_w37s99_seed1_hh",
    "Aircraft":   "Aircraft_w100s67_seed1_hh",
    "Cars":       "Cars_w196s42_seed1_hh",
    "SUN397":     "SUN397_w397s219_seed1_hh",
    "dSprites":   "dSprites_w100s500_seed1_hh",
}

# =============================================================================
# SOURCE DATASETS (13 datasets used for training)
# =============================================================================

# The 13 source datasets used in the actual training run
SOURCE_DATASET_SHORT_NAMES = [
    "c86", "c59", "c16", "c14", "c38", "c43", "c9",
    "c12", "c32", "c19", "c31", "c57", "c29",
]

# CN_DICT maps short names to directory names — load from learnware_info
# We'll build this dynamically
def _build_source_dataset_dirs():
    """Parse learnware_info.py CN_DICT to map source dataset names to dirs."""
    # Hardcoded from the file we read — these are the training source dirs
    cn_dict = {
        "c86": "EuroSAT_OfficeHome_SmallNORB_VLCS_w100s50_hh_c86",
        "c59": "STL10_SmallNORB_VLCS_w100s50_hh_c59",
        "c16": "OfficeHome_PACS_w100s50_hh_c16",
        "c14": "STL10_SmallNORB_w100s50_hh_c14",
        "c38": "OfficeHome_PACS_STL10_w100s50_hh_c38",
        "c43": "OfficeHome_SmallNORB_VLCS_w100s50_hh_c43",
        "c9":  "OfficeHome_SmallNORB_w100s50_hh_c9",
        "c12": "PACS_SmallNORB_w100s50_hh_c12",
        "c32": "EuroSAT_OfficeHome_VLCS_w100s50_hh_c32",
        "c19": "OfficeHome_VLCS_w100s50_hh_c19",
        "c31": "SmallNORB_VLCS_w100s50_hh_c31",
        "c57": "EuroSAT_OfficeHome_PACS_w100s50_hh_c57",
        "c29": "OfficeHome_STL10_w100s50_hh_c29",
    }
    return cn_dict

SOURCE_DATASET_DIRS = _build_source_dataset_dirs()

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

def _load_model_config():
    """Load the 100 PTM model list and feature dimensions."""
    sys_path = str(PROJECT_ROOT)
    if sys_path not in os.sys.path:
        os.sys.path.insert(0, sys_path)

    from learnware.learnware_info_100 import (
        BKB_100_SPECIFIC_RANK,
        MODEL_100_2FEAT_DIM,
        TORCHVISION_MODELS_100,
        TIMM_MODELS_100,
    )
    return BKB_100_SPECIFIC_RANK, MODEL_100_2FEAT_DIM, TORCHVISION_MODELS_100, TIMM_MODELS_100


# =============================================================================
# MANIFEST (progress tracking)
# =============================================================================

def load_manifest(output_dir: Path) -> dict:
    """Load or create manifest.json for progress tracking."""
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "r") as f:
            return json.load(f)

    model_list, _, _, _ = _load_model_config()
    all_datasets = list(TEST_DATASET_DIRS.keys()) + SOURCE_DATASET_SHORT_NAMES

    manifest = {
        "start_time": datetime.now().isoformat(),
        "last_update": datetime.now().isoformat(),
        "models_total": len(model_list),
        "datasets": {},
        "errors": [],
    }
    for ds in all_datasets:
        manifest["datasets"][ds] = {
            "models_done": 0,
            "models_total": len(model_list),
            "status": "pending",
        }
    return manifest


def save_manifest(manifest: dict, output_dir: Path):
    """Save manifest.json."""
    manifest["last_update"] = datetime.now().isoformat()
    manifest_path = output_dir / "manifest.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)


def update_manifest_model_done(manifest: dict, ds_name: str, model_name: str, output_dir: Path):
    """Update manifest after a model-dataset pair is completed."""
    ds_info = manifest["datasets"].get(ds_name)
    if ds_info is None:
        return
    ds_info["models_done"] += 1
    if ds_info["models_done"] >= ds_info["models_total"]:
        ds_info["status"] = "done"
    else:
        ds_info["status"] = "in_progress"
    save_manifest(manifest, output_dir)


def count_completed(feature_dir: Path, ds_name: str, model_list: list) -> int:
    """Count how many models already have feature files for a dataset."""
    ds_dir = feature_dir / ds_name
    if not ds_dir.exists():
        return 0
    return sum(1 for m in model_list if (ds_dir / f"{m}.pt").exists())


def print_status(output_dir: Path):
    """Print current pipeline status."""
    manifest = load_manifest(output_dir)
    model_list = None
    try:
        model_list, _, _, _ = _load_model_config()
    except Exception:
        pass

    feature_dir = output_dir / "features"

    print("=" * 70)
    print(f"100 PTM Pipeline Status — {manifest.get('last_update', 'N/A')}")
    print("=" * 70)

    total_done = 0
    total_all = 0

    for ds_name, ds_info in manifest["datasets"].items():
        # Recount from disk
        if model_list:
            actual_done = count_completed(feature_dir, ds_name, model_list)
        else:
            actual_done = ds_info["models_done"]
        total = ds_info["models_total"]
        total_done += actual_done
        total_all += total

        bar_len = 30
        filled = int(bar_len * actual_done / total) if total > 0 else 0
        bar = "█" * filled + "░" * (bar_len - filled)
        pct = 100 * actual_done / total if total > 0 else 0
        print(f"  {ds_name:<15} [{bar}] {actual_done:>3}/{total} ({pct:.0f}%)")

    print("-" * 70)
    overall_pct = 100 * total_done / total_all if total_all > 0 else 0
    print(f"  Overall: {total_done}/{total_all} model-dataset pairs ({overall_pct:.1f}%)")

    if manifest.get("errors"):
        print(f"\n  Errors ({len(manifest['errors'])}):")
        for err in manifest["errors"][-10:]:
            print(f"    - {err}")
    print()
