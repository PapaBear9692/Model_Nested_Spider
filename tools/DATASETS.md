# 100 PTM Prototype Generation Pipeline

## What It Does

This pipeline generates prototype features from **100 pre-trained models** across your training and test datasets. It produces pkl files in the same format as the existing `implclproto/` directory, but with 100 model features instead of 10.

The pipeline has three phases:
1. **Extract** — Load each model, extract class-wise mean features, save as individual `.pt` files
2. **Merge** — Combine individual features into the pkl format used by Model Spider
3. **Status** — Show progress (how many model-dataset pairs are done)

## Key Features

- **Resume-safe**: Skips existing files. If power cuts out, just re-run the same command — it picks up where it left off.
- **Immediate saves**: Each model's features are saved as soon as extracted — no data lost on crash.
- **One model at a time**: Fits within 8 GB VRAM (RTX 3050).
- **Progress tracking**: `manifest.json` tracks which model-dataset pairs are done.

## Quick Start

### Step 1: Install Dependencies

```bash
conda activate spider
pip install timm tqdm
```

### Step 2: Create Raw Data Directory

```bash
mkdir raw_data
```

### Step 3: Download Datasets

The 9 test datasets need raw images for feature extraction. Some auto-download:

| Dataset | Auto-Download | How to Get |
|---------|:---:|-----------|
| CIFAR10 | Yes | Automatic via torchvision |
| CIFAR100 | Yes | Automatic via torchvision |
| Caltech101 | Yes | Automatic via torchvision |
| DTD | Yes | Automatic via torchvision |
| Pet | Yes | Automatic via torchvision |
| SUN397 | Yes | Automatic via torchvision |
| Aircraft | No | Download from [FGVC-Aircraft](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/) and place in `raw_data/aircraft/` |
| Cars | No | Download from [Stanford Cars](https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset) and place in `raw_data/cars/` |
| dSprites | No | Download from [dSprites](https://github.com/deepmind/dsprites-dataset) and place in `raw_data/dsprites/` |

### Step 4: Run the Pipeline

```bash
# Check status at any time
python tools/gen100_pipeline.py --phase status

# Extract features for all test datasets (5-6 hours, resumable)
python tools/gen100_pipeline.py --phase extract --raw_data ./raw_data

# Or extract just specific models/datasets for testing
python tools/gen100_pipeline.py --phase extract --models resnet50 efficientnet_b0 --datasets CIFAR10 CIFAR100 --raw_data ./raw_data

# After extraction, merge into training-ready pkl files
python tools/gen100_pipeline.py --phase merge

# Or run everything at once
python tools/gen100_pipeline.py --phase all --raw_data ./raw_data
```

## How It Works

### Directory Structure

```
required_files/data/implclproto_100/
├── features/                          # Phase 1: individual .pt files
│   ├── CIFAR10/
│   │   ├── resnet18.pt               # [10, 512] tensor
│   │   ├── resnet50.pt               # [10, 2048] tensor
│   │   ├── efficientnet_b0.pt        # [10, 1280] tensor
│   │   └── ... (100 files)
│   ├── Caltech101/
│   │   └── ... (100 files)
│   └── ...
├── merged/                            # Phase 2: final pkl files
│   ├── CIFAR10_w10s5000_seed1_hh/
│   │   └── z_hash.pkl               # [uniform(10,3072), {100 model tensors}]
│   └── ...
└── manifest.json                      # Progress tracker
```

### Feature Extraction Process

For each model:
1. Load model to GPU (torchvision or timm)
2. Register a forward hook on the penultimate layer
3. Run inference on dataset batches
4. Collect features per class
5. Compute class-wise mean (prototype)
6. Save as `.pt` file immediately
7. Unload model, clear GPU memory

### Source Datasets

The 13 source training datasets (c86, c59, etc.) are combinations of multiple public datasets. Since their raw data requires specific splits, the pipeline handles them differently:

1. Copies existing 10-model features from `implclproto/`
2. Adds zero placeholders for the 90 new models
3. This allows training to start immediately with all 100 model slots

To replace zero placeholders with real features later, you'd need to download the component datasets (EuroSAT, OfficeHome, SmallNORB, VLCS, PACS, STL10, etc.) and run extraction with `--datasets c86 c59 ...`

## Estimated Time

| Task | Time | Notes |
|------|------|-------|
| CIFAR10 (10 classes) | ~8 min / 100 models | 5000 samples, auto-download |
| CIFAR100 (100 classes) | ~15 min / 100 models | 10000 samples, auto-download |
| SUN397 (397 classes) | ~50 min / 100 models | 40000 samples, auto-download |
| **All 9 test datasets** | **~5-6 hours** | Fully resumable |
| Merge phase | ~2 min | Just reads/writes files |

## Resuming After Interruption

Simply re-run the same command:

```bash
python tools/gen100_pipeline.py --phase extract --raw_data ./raw_data
```

The pipeline checks for existing `.pt` files and skips them. No wasted work.

Check progress anytime:

```bash
python tools/gen100_pipeline.py --phase status
```

## Training with 100 PTMs

After merge, train with:

```bash
set PTM100=yes
python trainer.py --seed 0 --num_learnware 100 --batch_size 4 --heterogeneous --use_hierarchy ^
  --train_dataset c86 c59 c16 c14 c38 c43 c9 c12 c32 c19 c31 c57 c29 ^
  --val_dataset c86 ^
  --test_dataset CIFAR10 Caltech101 DTD Pet Aircraft CIFAR100 Cars SUN397 dSprites ^
  --data_sub_url swin_base_7_checkpoint ^
  --data_url "D:\Study\Thesis\Model_Nested_Spider\required_files\data\implclproto_100\merged" ^
  --log_url "D:\Study\Thesis\Model_Nested_Spider\required_files\log_100" ^
  --num_workers 0
```

Note: You will also need to update `DATASET2DIR` in `learnware_info_100.py` to point to the new `merged/` directory structure.
