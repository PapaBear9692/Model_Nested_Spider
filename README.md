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

#### Inference Mode Run Command:
```
python trainer.py --seed 0 --train_dataset c86 c59 c16 c14 c38 c43 c9 c12 c32 c19 c31 c57 c29 --val_dataset c86 --test_dataset CIFAR10 Caltech101 DTD Pet Aircraft CIFAR100 Cars SUN397 dSprites --test_size_threshold 0 --data_sub_url swin_base_7_checkpoint --heterogeneous --lr 0.00025 --weight_decay 0.0005 --momentum 0.5 --max_epoch 30 --optimizer Adam --num_learnware 10 --batch_size 16 --dataset_size_threshold 0 --lr_scheduler cosine --val_ratio 0.05 --fixed_gt_size_threshold 0 --heterogeneous_sampled_maxnum 10 --data_url "D:\Study\Thesis\Model_Nested_Spider\required_filesdata\implclproto" --log_url "D:\Study\Thesis\Model_Nested_Spider\required_files\log" --pretrained_url "D:\Study\Thesis\Model_Nested_Spider\required_files\best.pth" --num_workers 0
```
#### Training Mode Run Command: 

```
python trainer.py --seed 0 --train_dataset c86 c59 c16 c14 c38 c43 c9 c12 c32 c19 c31 c57 c29 --val_dataset c86 --test_dataset CIFAR10 Caltech101 DTD Pet Aircraft CIFAR100 Cars SUN397 dSprites --test_size_threshold 0 --data_sub_url swin_base_7_checkpoint --heterogeneous --lr 0.00025 --weight_decay 0.0005 --momentum 0.5 --max_epoch 30 --optimizer Adam --num_learnware 10 --batch_size 16 --dataset_size_threshold 0 --lr_scheduler cosine --val_ratio 0.05 --fixed_gt_size_threshold 0 --heterogeneous_sampled_maxnum 10 --data_url "D:\Study\Thesis\Model_Nested_Spider\required_files\data\implclproto" --log_url "D:\Study\Thesis\Model_Nested_Spider\required_files\log" --num_workers 0
```
#### *just dropped --pretrained_url "D:\Study\Thesis\Model_Nested_Spider\required_files\best.pth" from command of inference mode
---
