"""
100 PTM Prototype Generation Pipeline.

Main orchestrator that handles three phases:
  1. extract  — Extract features from each model on each dataset (resume-safe)
  2. merge    — Merge individual .pt files into training-ready pkl files
  3. status   — Show current progress

Usage:
  python tools/gen100_pipeline.py --phase extract  [--raw_data PATH] [--output_dir PATH]
  python tools/gen100_pipeline.py --phase merge    [--output_dir PATH]
  python tools/gen100_pipeline.py --phase all      [--raw_data PATH] [--output_dir PATH]
  python tools/gen100_pipeline.py --phase status   [--output_dir PATH]

Features:
  - Resume-safe: skips existing .pt files, survives power outages
  - Progress tracking via manifest.json
  - One model loaded at a time (fits 8 GB VRAM)
  - Immediate per-file saves (no data loss on crash)
  - Source datasets handled via zero-placeholder expansion
"""

import os
import sys
import argparse
import pickle
import time
import hashlib
from pathlib import Path
from datetime import datetime

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.gen100_config import (
    PROJECT_ROOT as _PROOT,
    ORIGINAL_PROTO_DIR,
    DEFAULT_OUTPUT_DIR,
    TEST_DATASETS,
    TEST_DATASET_DIRS,
    SOURCE_DATASET_SHORT_NAMES,
    SOURCE_DATASET_DIRS,
    load_manifest,
    save_manifest,
    update_manifest_model_done,
    count_completed,
    print_status,
)
from learnware.learnware_info_100 import (
    BKB_100_SPECIFIC_RANK,
    MODEL_100_2FEAT_DIM,
)


# =============================================================================
# PHASE 1: EXTRACT
# =============================================================================

def phase_extract(output_dir: Path, raw_data_path: str, batch_size: int = 32,
                   models: list = None, datasets: list = None):
    """Extract features from models on datasets. Fully resumable."""
    import torch
    from tools.gen100_extract import extract_one_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    feature_dir = output_dir / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(output_dir)

    # Build dataset list: (ds_name, input_size, num_classes)
    all_datasets = []
    if datasets:
        # Filter to requested datasets
        for ds in datasets:
            if ds in TEST_DATASETS:
                all_datasets.append((ds, TEST_DATASETS[ds]["input_size"],
                                      TEST_DATASETS[ds]["num_classes"]))
            elif ds in SOURCE_DATASET_DIRS:
                # Source datasets need raw component data — skip for now
                print(f"  [SKIP] Source dataset '{ds}' requires raw component data (handled in merge)")
    else:
        # Default: all test datasets
        for ds_name, ds_info in TEST_DATASETS.items():
            all_datasets.append((ds_name, ds_info["input_size"], ds_info["num_classes"]))

    model_list = models if models else BKB_100_SPECIFIC_RANK

    print(f"\nDatasets to process: {len(all_datasets)}")
    print(f"Models to process: {len(model_list)}")
    print(f"Output: {feature_dir}")
    print(f"Raw data path: {raw_data_path}\n")

    # Count what's already done
    total_pairs = len(model_list) * len(all_datasets)
    already_done = sum(
        count_completed(feature_dir, ds_name, model_list)
        for ds_name, _, _ in all_datasets
    )
    print(f"Already completed: {already_done}/{total_pairs} ({100*already_done/total_pairs:.1f}%)")
    print("=" * 70)

    start_time = time.time()
    models_done = 0

    for model_idx, model_name in enumerate(model_list):
        # Count how many datasets this model still needs
        pending = sum(
            1 for ds_name, _, _ in all_datasets
            if not (feature_dir / ds_name / f"{model_name}.pt").exists()
        )
        if pending == 0:
            continue

        elapsed = time.time() - start_time
        if models_done > 0:
            rate = elapsed / models_done
            remaining = rate * (len(model_list) - model_idx)
            eta = f"ETA: {remaining/60:.0f}min"
        else:
            eta = "ETA: calculating..."

        print(f"[{model_idx+1}/{len(model_list)}] {model_name} "
              f"({pending} datasets pending) {eta}")

        try:
            results = extract_one_model(
                model_name=model_name,
                datasets_to_process=all_datasets,
                feature_dir=feature_dir,
                device=device,
                batch_size=batch_size,
                data_path=raw_data_path,
            )

            for ds_name, status in results.items():
                if status == "ok":
                    update_manifest_model_done(manifest, ds_name, model_name, output_dir)
                elif status == "skip":
                    pass  # already counted in previous runs
                else:
                    manifest.setdefault("errors", []).append(
                        f"{model_name}@{ds_name}: {status}"
                    )
                    save_manifest(manifest, output_dir)

            models_done += 1

        except Exception as e:
            print(f"    FATAL: {e}")
            manifest.setdefault("errors", []).append(f"{model_name}: FATAL {e}")
            save_manifest(manifest, output_dir)
            torch.cuda.empty_cache()
            continue

    total_time = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"Extraction complete. Total time: {total_time/60:.1f} minutes")
    print_status(output_dir)


# =============================================================================
# PHASE 2: MERGE
# =============================================================================

def phase_merge(output_dir: Path):
    """Merge individual .pt feature files into training-ready pkl files."""
    feature_dir = output_dir / "features"
    merged_dir = output_dir / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Phase 2: Merging features into pkl files")
    print("=" * 70)

    # Process test datasets
    for ds_name, ds_dir_name in TEST_DATASET_DIRS.items():
        print(f"\n  Merging {ds_name}...")
        _merge_one_dataset(
            ds_name=ds_name,
            ds_dir_name=ds_dir_name,
            feature_dir=feature_dir,
            merged_dir=merged_dir,
            is_test=True,
        )

    # Process source datasets
    for ds_short in SOURCE_DATASET_SHORT_NAMES:
        ds_dir_name = SOURCE_DATASET_DIRS[ds_short]
        print(f"\n  Merging {ds_short} ({ds_dir_name})...")
        _merge_one_dataset(
            ds_name=ds_short,
            ds_dir_name=ds_dir_name,
            feature_dir=feature_dir,
            merged_dir=merged_dir,
            is_test=False,
        )

    print(f"\nMerge complete. Output: {merged_dir}")


def _merge_one_dataset(ds_name, ds_dir_name, feature_dir, merged_dir, is_test):
    """Merge features for a single dataset."""
    import torch

    # Find original pkl file(s)
    orig_dir = ORIGINAL_PROTO_DIR / ds_dir_name
    if not orig_dir.exists():
        print(f"    [SKIP] Original dir not found: {orig_dir}")
        return

    pkl_files = list(orig_dir.glob("z_*.pkl")) if is_test else list(orig_dir.glob("*.pkl"))
    if not pkl_files:
        # Try any pkl for source datasets
        pkl_files = list(orig_dir.glob("*.pkl"))

    if not pkl_files:
        print(f"    [SKIP] No pkl files in {orig_dir}")
        return

    for pkl_file in pkl_files:
        try:
            original = pickle.load(open(pkl_file, "rb"))
        except Exception as e:
            print(f"    [ERROR] Cannot load {pkl_file.name}: {e}")
            continue

        uniform = original[0]  # Swin features — keep unchanged
        labels = original[2] if len(original) > 2 else None
        num_classes = uniform.shape[0]

        # Build heterogeneous dict from individual .pt files + existing data
        hete_dict = {}

        # First, copy existing 10-model features if they exist
        existing_hete = original[1] if len(original) > 1 else {}
        for k, v in existing_hete.items():
            hete_dict[k] = v

        # Then load new .pt files (these override existing if both exist)
        real_count = 0
        zero_count = 0
        for model_name in BKB_100_SPECIFIC_RANK:
            pt_file = feature_dir / ds_name / f"{model_name}.pt"
            if pt_file.exists():
                hete_dict[model_name] = torch.load(pt_file, weights_only=True)
                real_count += 1
            elif model_name not in hete_dict:
                # Zero placeholder for models without features
                feat_dim = MODEL_100_2FEAT_DIM[model_name]
                hete_dict[model_name] = torch.zeros(num_classes, feat_dim)
                zero_count += 1

        # Build output
        if labels is not None:
            result = [uniform, hete_dict, labels]
        else:
            result = [uniform, hete_dict]

        # Save with same naming convention
        out_dir = merged_dir / ds_dir_name
        out_dir.mkdir(parents=True, exist_ok=True)

        if is_test:
            # Test datasets use z_ prefix
            out_file = out_dir / f"z_{_hash_name(ds_name)}.pkl"
        else:
            out_file = out_dir / pkl_file.name

        with open(out_file, "wb") as f:
            pickle.dump(result, f)

        file_size = os.path.getsize(out_file) / 1024 / 1024
        print(f"    {pkl_file.name} -> {out_file.name} "
              f"({len(hete_dict)} models, {real_count} real, {zero_count} zeros, {file_size:.1f} MB)")


def _hash_name(name: str) -> str:
    """Generate a deterministic hash for a dataset name."""
    return hashlib.sha1(name.encode()).hexdigest()[:40]


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="100 PTM Prototype Generation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check current status
  python tools/gen100_pipeline.py --phase status

  # Extract features for all test datasets (resumable)
  python tools/gen100_pipeline.py --phase extract --raw_data ./raw_data

  # Extract features for specific models and datasets
  python tools/gen100_pipeline.py --phase extract --models resnet50 vgg16 --datasets CIFAR10 CIFAR100

  # Merge extracted features into training pkl files
  python tools/gen100_pipeline.py --phase merge

  # Full pipeline
  python tools/gen100_pipeline.py --phase all --raw_data ./raw_data
        """,
    )

    parser.add_argument("--phase", type=str, required=True,
                        choices=["extract", "merge", "all", "status"],
                        help="Pipeline phase to run")
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
                        help="Output directory (default: required_files/data/implclproto_100)")
    parser.add_argument("--raw_data", type=str, default="./raw_data",
                        help="Path to raw dataset downloads (for CIFAR10, etc.)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for feature extraction (default: 32)")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Specific models to process (default: all 100)")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Specific datasets to process (default: all test datasets)")

    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    if args.phase == "status":
        print_status(output_dir)
        return

    if args.phase in ("extract", "all"):
        phase_extract(
            output_dir=output_dir,
            raw_data_path=args.raw_data,
            batch_size=args.batch_size,
            models=args.models,
            datasets=args.datasets,
        )

    if args.phase in ("merge", "all"):
        phase_merge(output_dir)

    if args.phase == "status":
        print_status(output_dir)


if __name__ == "__main__":
    main()
