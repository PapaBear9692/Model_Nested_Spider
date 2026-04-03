# Vanilla Model-Spider Training Results (Explained for Dummies)

## What Did We Just Do?

We trained a **PTM Ranker** — a model that looks at a new task and decides which pre-trained model (PTM) will work best for that task.

**Analogy:** Imagine you're a chef choosing the right knife for a dish. You have 10 different knives (PTMs) in your kitchen. For each new dish (task), the model predicts which knife will work best — without actually testing all of them.

---

## What Datasets Did We Use?

### Training Datasets (The "Practice Problems")

We used **13 combined datasets** (named c86, c59, c16, etc.) to teach the model. Each is a mix of two benchmark datasets:

| Code | Combination |
|------|-------------|
| c86 | Dogs + Flowers |
| c59 | OfficeHome + SmallNORB |
| c16 | OfficeHome + PACS |
| c14 | STL10 + SmallNORB |
| c38 | OfficeHome + PACS + STL10 |
| c43 | OfficeHome + SmallNORB + VLCS |
| c9 | OfficeHome + SmallNORB |
| c12 | PACS + SmallNORB |
| c32 | EuroSAT + OfficeHome + VLCS |
| c19 | OfficeHome + VLCS |
| c31 | SmallNORB + VLCS |
| c57 | ? (combination dataset) |
| c29 | OfficeHome + STL10 |

These are "meta-datasets" — combinations that help the model learn general patterns about which PTMs work well on which types of tasks.

### Test Datasets (The "Real World Problems")

We tested on **9 popular benchmark datasets**:

| Dataset | What It Contains | Why It Matters |
|---------|------------------|----------------|
| **CIFAR10** | 10 classes of small images (planes, cars, birds, etc.) | Basic image classification benchmark |
| **CIFAR100** | 100 classes of small images | Harder version of CIFAR10 |
| **Caltech101** | 101 object categories | Classic object recognition dataset |
| **DTD** | Describable textures | Tests texture understanding |
| **Pet** | 37 pet breeds (cats & dogs) | Fine-grained classification |
| **Aircraft** | 100 aircraft variants | Fine-grained object recognition |
| **Cars** | 196 car models | Fine-grained vehicle recognition |
| **SUN397** | 397 scene categories | Large-scale scene understanding |
| **dSprites** | 2D shapes with latent factors | Tests disentangled representations |

---

## What Do The Results Mean?

### The Metric: Weighted Tau (τ)

**Weighted Tau** measures how well our rankings match the true rankings. 

- **Score ranges from -1 to 1**
  - **1.0** = Perfect ranking (we predicted the exact order of best to worst PTMs)
  - **0.0** = Random guessing
  - **-1.0** = Perfectly wrong (we ranked everything backwards)

**Example:** If the true best PTM for CIFAR10 is ResNet50, and we ranked it #1, that's good! If we ranked it #10, that's bad.

### What is "k"?

**k = number of additional PTM features used**

- **k=0**: We only use the base features (Swin Transformer) — simplest prediction
- **k=5**: We use features from 5 additional PTMs (like ResNet, DenseNet, etc.) — more information
- **k=10**: We use features from all 10 PTMs — most information, but potentially noisy

Think of it like asking for second opinions:
- k=0 = make a decision alone
- k=5 = ask 5 friends for advice
- k=10 = ask 10 friends for advice (might be too many conflicting opinions!)

---

## The Results: How Well Did We Do?

### Best Performance Per Dataset

| Dataset | Best Score | Best k | What This Means |
|---------|------------|-------|-----------------|
| **CIFAR100** | 0.871 | ≥4 | Excellent! Extra PTM info really helps |
| **CIFAR10** | 0.854 | ≥4 | Great! Similar to CIFAR100 |
| **SUN397** | 0.864 | ≥5 | Excellent! Scene recognition benefits from diverse PTMs |
| **Cars** | 0.775 | k=5 | Good! Moderate extra info helps |
| **Pet** | 0.721 | ≥4 | Good! Fine-grained pets need some PTM diversity |
| **Caltech101** | 0.637 | ≥4 | Decent. General objects benefit from extra info |
| **DTD** | 0.661 | k<4 | Decent. Textures prefer simpler predictions |
| **dSprites** | 0.625 | all | Decent. Unaffected by k (simple shapes) |
| **Aircraft** | 0.551 | k<5 | Mediocre. Too much info hurts! |

### What Patterns Do We See?

1. **Simple datasets (CIFAR10, CIFAR100)** → More PTM info helps a lot ✅
2. **Complex datasets (Aircraft, DTD)** → Less is more; too much info confuses the model ⚠️
3. **SUN397** → Sweet spot at k=5; more info helps but not too much 🎯

---

## Training Configuration (The Settings We Used)

```
Epochs: 30 (we trained for 30 full passes through the data)
Batch Size: 16 (we processed 16 samples at a time)
Learning Rate: 0.00025 (how fast the model learns)
Optimizer: Adam (the algorithm that updates the model)
PTMs to Rank: 10 (ResNet50, DenseNet, MobileNet, etc.)
Base Model: Swin Transformer (extracts the main features)
```

---

## Where Are The Files?

```
required_files/log/<long_folder_name>/0404-02-41-30-102/
├── 1.pth, 2.pth, ..., 30.pth    # Saved models after each epoch
├── configs.json                  # All settings used
├── heterogeneous_sampled_acc.csv # Detailed metrics
├── train.log                     # Console output
└── tflogger/                     # TensorBoard visualization
```

### How to Use a Saved Model

Pick any checkpoint (e.g., `30.pth` = final model) and run:

```bash
python trainer.py ... --pretrained_url "D:\Study\Thesis\Model_Nested_Spider\required_files\log\...\30.pth"
```

This skips training and goes straight to testing!

---

## Summary: What Did We Learn?

### The Good News ✅
- Our model achieves **0.85+ weightedtau** on CIFAR datasets — very reliable rankings
- SUN397 (397 classes!) also gets **0.86+** — scales well to large datasets
- The model learned meaningful patterns from training data

### The Challenges ⚠️
- **Aircraft dataset** (0.55) is tough — fine-grained aircraft variants are hard to rank
- **dSprites** (0.62) is mediocre — synthetic shapes don't benefit from real-world PTM knowledge
- Adding more PTM features (higher k) isn't always better — can add noise

### Takeaway 🎯
The Model-Spider approach works well for **natural image datasets** where pre-trained knowledge transfers. For **specialized domains** (aircraft, synthetic data), the rankings are less reliable.

---

## Next Steps?

1. **Try different k values** for your specific use case
2. **Fine-tune on domain-specific data** if working with specialized images
3. **Compare with baseline methods** (LEEP, LogME, H-Score) in the `mptms/` folder

---

## 66 PTM Mode (Extended)

### What Changed?

The 66 PTM mode expands the model zoo from **10 PTMs** to **66 PTMs**:

- **10 PTM Mode**: googlenet, inception_v3, resnet50, resnet101, resnet152, densenet121, densenet169, densenet201, mobilenet_v2, mnasnet1_0
- **66 PTM Mode**: DenseNet-201, Inception-V3, ResNet-50 — each fine-tuned on 22 datasets (AID, Aircraft, CIFAR10, CIFAR100, CUB2011, Caltech101, Cars, DTD, Dogs, EuroSAT, Flowers, Food, ImageNet, NABirds, PACS, Pet, Resisc45, STL10, SUN397, SVHN, SmallNORB, VLCS)

### How to Enable

Set environment variable `MODELS42=yes` and use `--num_learnware 66`:

```powershell
# Windows PowerShell
$env:MODELS42="yes"; python trainer.py --seed 0 --train_dataset c86 c59 c16 c14 c38 c43 c9 c12 c32 c19 c31 c57 c29 --val_dataset c86 --test_dataset CIFAR10 Caltech101 DTD Pet Aircraft CIFAR100 Cars SUN397 dSprites --test_size_threshold 0 --data_sub_url swin_base_7_checkpoint --heterogeneous --lr 0.00025 --weight_decay 0.0005 --momentum 0.5 --max_epoch 30 --optimizer Adam --num_learnware 66 --batch_size 8 --dataset_size_threshold 0 --lr_scheduler cosine --val_ratio 0.05 --fixed_gt_size_threshold 0 --heterogeneous_sampled_maxnum 10 --data_url "D:\Study\Thesis\Model_Nested_Spider\required_files\data\implclproto" --log_url "D:\Study\Thesis\Model_Nested_Spider\required_files\log" --num_workers 0
```

### ⚠️ Data Requirements

**Important:** The 66 PTM mode requires prototype data files (.pkl) that contain features for all 66 PTMs. If your data only has 10 PTM features, you need to either:

1. **Generate new prototype data** using `tools/feature_extractor.py`
2. **Or stick with 10 PTM mode** until you data is regenerated

### Key Differences

| Aspect | 10 PTM Mode | 66 PTM Mode |
|-------|-------------|-------------|
| PTMs to Rank | 10 | 66 |
| Model Zoo | Standard architectures | Fine-tuned variants |
| Memory Usage | Lower | Higher |
| Training Time | Faster | ~6-7x slower |
| Batch Size | 16 | 8 (recommended) |
| Data Required | 10 PTM features | 66 PTM features |

> **Warning:** If you get `KeyError` for PTM names, your prototype data doesn't have features for all 66 PTMs. You need to regenerate the data first.
