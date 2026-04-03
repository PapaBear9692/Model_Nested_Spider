# Model_Nested_Spider
## An Update To the original Model_Spider PTM ranker for efficient ranking from larger Model-Zoo.


### How to Run
0. Clone the Repository, download the required_files.zip, unzip it, copy the ```folders: data, log``` and ```file: best.pth``` into the required_files folder in codebase root.
It should look like this:

```
Model_Nested_Spider/
├── .gitignore
├── README.md
├── requirements.txt
├── trainer.py                 # Main training/testing pipeline
├── utils.py                   # Helpers (metrics, logging, CLI parsing)
├── original_flow.md           # Project documentation
│
├── datasets/                  # Dataset loaders for various benchmarks
│   ├── __init__.py
│   ├── load_dataset.py
│   ├── aircraft.py
│   ├── caltech101.py
│   ├── cars.py
│   ├── cub2011.py
│   ├── dogs.py
│   ├── domainnet.py
│   ├── dsprites.py
│   ├── dtd.py
│   ├── eurosat.py
│   ├── flowers.py
│   ├── nabirds.py
│   ├── officehome.py
│   ├── oxford_iiit_pet.py
│   ├── pacs.py
│   ├── pcam.py
│   ├── resisc45.py
│   ├── smallnorb.py
│   ├── sun397.py
│   ├── udomain.py
│   ├── utkface.py
│   ├── vlcs.py
│   └── aid.py
│
├── learnware/                 # Core model components
│   ├── model.py               # LearnwareCAHeterogeneous transformer
│   ├── loss.py                # HierarchicalCE, Top1CE, ListMLE losses
│   ├── dataset.py             # LearnwareDataset + collate_fn
│   └── learnware_info.py      # Constants (PTM names, feature dims, ranks)
│
├── mptms/                     # Baseline PTM selection methods
│   ├── DEPARA.py
│   ├── GBC.py
│   ├── H_Score.py
│   ├── LEEP.py
│   ├── LFC.py
│   ├── LogME.py
│   ├── NCE.py
│   ├── OTCE.py
│   └── PACTran.py
│
├── scripts/                   # Shell scripts
│   ├── test-model-spider.sh
│   ├── reproduce-baseline-methods.sh
│   └── modify-path.sh
│
├── tools/                     # Utility scripts
│   └── feature_extractor.py
│
└── required_files/            # Data & checkpoints (not in git)
    ├── .gitkeep
    ├── best.pth               # Pre-trained model weights
    ├── data/
    │   └── implclproto/       # Precomputed prototype .pkl files
    └── log/                   # Training logs
```

1. First create a virtual environment
For conda: ```conda create --name spider python=3.11```
For venv: ```python3 -m venv spider```

2. Activate the virtual environment
For conda: ```conda activate spider```
For venv:: source ```spider/bin/activate```

3. Install requirements
For conda and venv both: ```pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu118```

4. Run the application using command


### Please change these directories in the command according to your workspace path.
```
--data_url "D:\Study\Thesis\modelspider_store\data\implclproto" 
--log_url "D:\Study\Thesis\modelspider_store\log" 
--pretrained_url "D:\Study\Thesis\modelspider_store\best.pth"
```

---

## 10 PTM Mode (Default)

The default mode ranks **10 PTMs**: googlenet, inception_v3, resnet50, resnet101, resnet152, densenet121, densenet169, densenet201, mobilenet_v2, mnasnet1_0

### Inference Mode (10 PTMs):

Use this to first train then do inference. (change the name of .pth file)
```
python trainer.py --seed 0 --train_dataset c86 c59 c16 c14 c38 c43 c9 c12 c32 c19 c31 c57 c29 --val_dataset c86 --test_dataset CIFAR10 Caltech101 DTD Pet Aircraft CIFAR100 Cars SUN397 dSprites --test_size_threshold 0 --data_sub_url swin_base_7_checkpoint --heterogeneous --lr 0.00025 --weight_decay 0.0005 --momentum 0.5 --max_epoch 30 --optimizer Adam --num_learnware 10 --batch_size 16 --dataset_size_threshold 0 --lr_scheduler cosine --val_ratio 0.05 --fixed_gt_size_threshold 0 --heterogeneous_sampled_maxnum 10 --data_url "D:\Study\Thesis\Model_Nested_Spider\required_files\data\implclproto" --log_url "D:\Study\Thesis\Model_Nested_Spider\required_files\log" --pretrained_url "D:\Study\Thesis\Model_Nested_Spider\required_files\best.pth" --num_workers 0
```

### Training Mode (10 PTMs):

```
python trainer.py --seed 0 --train_dataset c86 c59 c16 c14 c38 c43 c9 c12 c32 c19 c31 c57 c29 --val_dataset c86 --test_dataset CIFAR10 Caltech101 DTD Pet Aircraft CIFAR100 Cars SUN397 dSprites --test_size_threshold 0 --data_sub_url swin_base_7_checkpoint --heterogeneous --lr 0.00025 --weight_decay 0.0005 --momentum 0.5 --max_epoch 30 --optimizer Adam --num_learnware 10 --batch_size 16 --dataset_size_threshold 0 --lr_scheduler cosine --val_ratio 0.05 --fixed_gt_size_threshold 0 --heterogeneous_sampled_maxnum 10 --data_url "D:\Study\Thesis\Model_Nested_Spider\required_files\data\implclproto" --log_url "D:\Study\Thesis\Model_Nested_Spider\required_files\log" --num_workers 0
```

> **Note:** Inference mode = same command but WITH `--pretrained_url` flag (skips training)

---

## 66 PTM Mode (Extended)

This mode ranks **66 PTMs** — combinations of 3 architectures (DenseNet-201, Inception-V3, ResNet-50) fine-tuned on 22 datasets each.

### Important:
To run 66 PTM mode, your current prototype data needs to be regenerated with features for all 66 PTMs. The ```tools/feature_extractor.py``` script can help with that, or you need to provide the 66 fine-tuned model weights yourself. let me know if you need help with that.

### The 66 PTMs:

| Architecture | Datasets |
|--------------|----------|
| DenseNet-201 | AID, Aircraft, CIFAR10, CIFAR100, CUB2011, Caltech101, Cars, DTD, Dogs, EuroSAT, Flowers, Food, ImageNet, NABirds, PACS, Pet, Resisc45, STL10, SUN397, SVHN, SmallNORB, VLCS |
| Inception-V3 | Same 22 datasets |
| ResNet-50 | Same 22 datasets |

### How to Enable:

Set environment variable `MODELS42=yes` and use `--num_learnware 66`

### Training Mode (66 PTMs):

**Windows PowerShell:**
```powershell
$env:MODELS42="yes"; python trainer.py --seed 0 --train_dataset c86 c59 c16 c14 c38 c43 c9 c12 c32 c19 c31 c57 c29 --val_dataset c86 --test_dataset CIFAR10 Caltech101 DTD Pet Aircraft CIFAR100 Cars SUN397 dSprites --test_size_threshold 0 --data_sub_url swin_base_7_checkpoint --heterogeneous --lr 0.00025 --weight_decay 0.0005 --momentum 0.5 --max_epoch 30 --optimizer Adam --num_learnware 66 --batch_size 8 --dataset_size_threshold 0 --lr_scheduler cosine --val_ratio 0.05 --fixed_gt_size_threshold 0 --heterogeneous_sampled_maxnum 10 --data_url "D:\Study\Thesis\Model_Nested_Spider\required_files\data\implclproto" --log_url "D:\Study\Thesis\Model_Nested_Spider\required_files\log" --num_workers 0
```

**Windows CMD:**
```cmd
set MODELS42=yes && python trainer.py --seed 0 --train_dataset c86 c59 c16 c14 c38 c43 c9 c12 c32 c19 c31 c57 c29 --val_dataset c86 --test_dataset CIFAR10 Caltech101 DTD Pet Aircraft CIFAR100 Cars SUN397 dSprites --test_size_threshold 0 --data_sub_url swin_base_7_checkpoint --heterogeneous --lr 0.00025 --weight_decay 0.0005 --momentum 0.5 --max_epoch 30 --optimizer Adam --num_learnware 66 --batch_size 8 --dataset_size_threshold 0 --lr_scheduler cosine --val_ratio 0.05 --fixed_gt_size_threshold 0 --heterogeneous_sampled_maxnum 10 --data_url "D:\Study\Thesis\Model_Nested_Spider\required_files\data\implclproto" --log_url "D:\Study\Thesis\Model_Nested_Spider\required_files\log" --num_workers 0
```

**Git Bash / Linux:**
```bash
MODELS42=yes python trainer.py --seed 0 --train_dataset c86 c59 c16 c14 c38 c43 c9 c12 c32 c19 c31 c57 c29 --val_dataset c86 --test_dataset CIFAR10 Caltech101 DTD Pet Aircraft CIFAR100 Cars SUN397 dSprites --test_size_threshold 0 --data_sub_url swin_base_7_checkpoint --heterogeneous --lr 0.00025 --weight_decay 0.0005 --momentum 0.5 --max_epoch 30 --optimizer Adam --num_learnware 66 --batch_size 8 --dataset_size_threshold 0 --lr_scheduler cosine --val_ratio 0.05 --fixed_gt_size_threshold 0 --heterogeneous_sampled_maxnum 10 --data_url "D:\Study\Thesis\Model_Nested_Spider\required_files\data\implclproto" --log_url "D:\Study\Thesis\Model_Nested_Spider\required_files\log" --num_workers 0
```

### Inference Mode (66 PTMs):

Add `--pretrained_url` to skip training:

**Windows PowerShell:**
```powershell
$env:MODELS42="yes"; python trainer.py --seed 0 --train_dataset c86 c59 c16 c14 c38 c43 c9 c12 c32 c19 c31 c57 c29 --val_dataset c86 --test_dataset CIFAR10 Caltech101 DTD Pet Aircraft CIFAR100 Cars SUN397 dSprites --test_size_threshold 0 --data_sub_url swin_base_7_checkpoint --heterogeneous --lr 0.00025 --weight_decay 0.0005 --momentum 0.5 --max_epoch 30 --optimizer Adam --num_learnware 66 --batch_size 8 --dataset_size_threshold 0 --lr_scheduler cosine --val_ratio 0.05 --fixed_gt_size_threshold 0 --heterogeneous_sampled_maxnum 10 --data_url "D:\Study\Thesis\Model_Nested_Spider\required_files\data\implclproto" --log_url "D:\Study\Thesis\Model_Nested_Spider\required_files\log" --pretrained_url "D:\Study\Thesis\Model_Nested_Spider\required_files\best_66.pth" --num_workers 0
```

### Key Differences: 10 vs 66 PTM Mode

| Parameter | 10 PTM Mode | 66 PTM Mode |
|-----------|-------------|-------------|
| `MODELS42` | not set | `yes` |
| `--num_learnware` | 10 | 66 |
| `--batch_size` | 16 | 8 (recommended) |
| Memory Usage | Lower | Higher |
| Training Time | Faster | ~6-7x slower |

> **Warning:** 66 PTM mode requires more GPU memory. If you get OOM errors, reduce `--batch_size` to 4.

---

## Command Line Flags Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--seed` | 0 | Random seed for reproducibility |
| `--train_dataset` | - | List of dataset codes for training (e.g., c86, c59) |
| `--val_dataset` | - | Dataset for validation |
| `--test_dataset` | - | List of downstream datasets to evaluate on |
| `--test_size_threshold` | 0 | Max samples per test dataset (0 = all) |
| `--data_sub_url` | swin_base_7_checkpoint | Base PTM feature extractor name |
| `--heterogeneous` | False | Enable multi-source model zoo mode |
| `--lr` | 0.01 | Learning rate |
| `--weight_decay` | 0.00005 | L2 regularization weight |
| `--momentum` | 0.8 | Momentum for optimizer |
| `--max_epoch` | 50 | Number of training epochs |
| `--optimizer` | Adam | Optimizer type (Adam, SGD) |
| `--num_learnware` | 72 | Number of PTMs to rank |
| `--batch_size` | 128 | Batch size for training/testing |
| `--dataset_size_threshold` | 0 | Max training samples (0 = all) |
| `--lr_scheduler` | cosine | Learning rate scheduler type |
| `--val_ratio` | 0.2 | Validation split ratio |
| `--fixed_gt_size_threshold` | 128 | Max fixed ground-truth samples |
| `--heterogeneous_sampled_maxnum` | 10 | Max heterogeneous backbones to sample |
| `--data_url` | - | Path to prototype data folder |
| `--log_url` | - | Path to save logs |
| `--pretrained_url` | None | Path to pretrained weights (skips training if set) |
| `--num_workers` | 8 | DataLoader worker processes |
| `--gpu` | 0 | GPU device ID |
| `--attn_pool` | cls | Attention pooling method (cls/mean) |

---


