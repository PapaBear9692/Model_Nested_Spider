# Testing Commands

## Run 1: Original Model Spider (Baseline)

### Training
```
python trainer.py --seed 0 --train_dataset c86 c59 c16 c14 c38 c43 c9 c12 c32 c19 c31 c57 c29 --val_dataset c86 --test_dataset CIFAR10 Caltech101 DTD Pet Aircraft CIFAR100 Cars SUN397 dSprites --test_size_threshold 0 --data_sub_url swin_base_7_checkpoint --heterogeneous --lr 0.00025 --weight_decay 0.0005 --momentum 0.5 --max_epoch 30 --optimizer Adam --num_learnware 10 --batch_size 16 --dataset_size_threshold 0 --lr_scheduler cosine --val_ratio 0.05 --fixed_gt_size_threshold 0 --heterogeneous_sampled_maxnum 10 --data_url "D:\Study\Thesis\Model_Nested_Spider\required_files\data\implclproto" --log_url "D:\Study\Thesis\Model_Nested_Spider\required_files\log" --num_workers 0
```

### Inference
Replace `<TIMESTAMP>` with the folder name from the log directory after training.
```
python trainer.py --seed 0 --train_dataset c86 c59 c16 c14 c38 c43 c9 c12 c32 c19 c31 c57 c29 --val_dataset c86 --test_dataset CIFAR10 Caltech101 DTD Pet Aircraft CIFAR100 Cars SUN397 dSprites --test_size_threshold 0 --data_sub_url swin_base_7_checkpoint --heterogeneous --lr 0.00025 --weight_decay 0.0005 --momentum 0.5 --max_epoch 30 --optimizer Adam --num_learnware 10 --batch_size 16 --dataset_size_threshold 0 --lr_scheduler cosine --val_ratio 0.05 --fixed_gt_size_threshold 0 --heterogeneous_sampled_maxnum 10 --data_url "D:\Study\Thesis\Model_Nested_Spider\required_files\data\implclproto" --log_url "D:\Study\Thesis\Model_Nested_Spider\required_files\log" --pretrained_url "D:\Study\Thesis\Model_Nested_Spider\required_files\log\c12_c14_c16_c19_c29_c31_c32_c38_c43_c57_c59_c86_c9__c86__Aircraft_CIFAR10_CIFAR100_Caltech101_Cars_DTD_Pet_SUN397_dSprites\<TIMESTAMP>\30.pth" --num_workers 0
```

---

## Run 2: Hierarchical Clustering Mode

### Training
```
python trainer.py --seed 0 --train_dataset c86 c59 c16 c14 c38 c43 c9 c12 c32 c19 c31 c57 c29 --val_dataset c86 --test_dataset CIFAR10 Caltech101 DTD Pet Aircraft CIFAR100 Cars SUN397 dSprites --test_size_threshold 0 --data_sub_url swin_base_7_checkpoint --heterogeneous --lr 0.00025 --weight_decay 0.0005 --momentum 0.5 --max_epoch 30 --optimizer Adam --num_learnware 10 --batch_size 16 --dataset_size_threshold 0 --lr_scheduler cosine --val_ratio 0.05 --fixed_gt_size_threshold 0 --heterogeneous_sampled_maxnum 10 --data_url "D:\Study\Thesis\Model_Nested_Spider\required_files\data\implclproto" --log_url "D:\Study\Thesis\Model_Nested_Spider\required_files\log" --num_workers 0 --use_hierarchy --hier_top_k_L1 1 --hier_top_k_families 2 --hier_alpha 0.3 --hier_beta 0.2
```

### Inference
Replace `<TIMESTAMP>` with the folder name from the log directory after training.
```
python trainer.py --seed 0 --train_dataset c86 c59 c16 c14 c38 c43 c9 c12 c32 c19 c31 c57 c29 --val_dataset c86 --test_dataset CIFAR10 Caltech101 DTD Pet Aircraft CIFAR100 Cars SUN397 dSprites --test_size_threshold 0 --data_sub_url swin_base_7_checkpoint --heterogeneous --lr 0.00025 --weight_decay 0.0005 --momentum 0.5 --max_epoch 30 --optimizer Adam --num_learnware 10 --batch_size 16 --dataset_size_threshold 0 --lr_scheduler cosine --val_ratio 0.05 --fixed_gt_size_threshold 0 --heterogeneous_sampled_maxnum 10 --data_url "D:\Study\Thesis\Model_Nested_Spider\required_files\data\implclproto" --log_url "D:\Study\Thesis\Model_Nested_Spider\required_files\log" --pretrained_url "D:\Study\Thesis\Model_Nested_Spider\required_files\log\c12_c14_c16_c19_c29_c31_c32_c38_c43_c57_c59_c86_c9__c86__Aircraft_CIFAR10_CIFAR100_Caltech101_Cars_DTD_Pet_SUN397_dSprites\<TIMESTAMP>\30.pth" --num_workers 0 --use_hierarchy --hier_top_k_L1 1 --hier_top_k_families 2 --hier_alpha 0.3 --hier_beta 0.2
```

---

## Results Location
- Log directory: `D:\Study\Thesis\Model_Nested_Spider\required_files\log\c12_c14_c16_c19_c29_c31_c32_c38_c43_c57_c59_c86_c9__c86__Aircraft_CIFAR10_CIFAR100_Caltech101_Cars_DTD_Pet_SUN397_dSprites\<TIMESTAMP>\`
- Per-epoch metrics: `heterogeneous_sampled_acc.csv`
- Model checkpoints: `<epoch>.pth`

---

## How to Read the Results

### During Training (console output)

After each epoch, you'll see lines like:
```
[15] CIFAR10 test   weightedtau  0.6521  pearsonr  0.7103
```
- **weightedtau** — Primary metric. Measures how well the predicted PTM ranking matches the ground-truth ranking. Range: -1 to 1. **Higher = better**. This is the main number to compare between original and hierarchical.
- **pearsonr** — Secondary metric. Measures linear correlation between predicted scores and ground-truth ranks. Range: -1 to 1. **Higher = better**.

These only appear when a new best score is achieved for that dataset, so you won't see them every epoch.

### During Inference (console output)

At the end of inference, you'll see the final PTM scores per dataset:
```
Model Spider's scores on ['googlenet', 'inception_v3', 'resnet50', ...]
CIFAR10: [0.85, 0.12, 0.92, ...]
```
These are the raw scores assigned to each PTM for that dataset. Higher score = model thinks that PTM is better for the task.

Then the final weightedtau results:
```
wtau of CIFAR10: 0.7234
wtau of Caltech101: 0.5891
...
```
Plus the best `heterogeneous_sample_num` — the number of heterogeneous PTMs to sample that gave the best average result.

### `heterogeneous_sampled_acc.csv` (in log folder)

Each row: `epoch,dataset_name,metric_value`
- Contains the weightedtau for each dataset at each epoch
- The last column is the metric for `k=10` heterogeneous samples
- Use this to plot training curves or compare final epoch results

### How to Compare Original vs Hierarchical

1. Run both training commands to completion (30 epochs each)
2. For each test dataset, note the **best weightedtau** across all epochs from the console output
3. Compare:

| Dataset | Original wtau | Hierarchical wtau | Delta |
|---------|--------------|-------------------|-------|
| CIFAR10 | ... | ... | ... |
| Caltech101 | ... | ... | ... |
| DTD | ... | ... | ... |
| Pet | ... | ... | ... |
| Aircraft | ... | ... | ... |
| CIFAR100 | ... | ... | ... |
| Cars | ... | ... | ... |
| SUN397 | ... | ... | ... |
| dSprites | ... | ... | ... |

- **Positive delta** = hierarchical mode is better
- **Negative delta** = original mode is better
- Look at the average across all datasets for the overall picture

---

## Resource Usage Output

At the end of training or inference, you'll see a summary like:

### Training
```
=== Training Summary ===
Total training time: 342.5s (5.7min)
Peak GPU memory: 1234 MB
GPU: NVIDIA GeForce RTX 3090
```

### Inference
```
=== Inference Summary ===
Total inference time: 45.2s
Peak GPU memory: 876 MB
GPU: NVIDIA GeForce RTX 3090
```

- **Total time** — wall-clock time from start to finish of the training/inference run
- **Peak GPU memory** — maximum GPU memory used during the run (lower is better, especially for 100 PTMs)
- **GPU** — the GPU device used (useful to confirm you're running on the right hardware)
