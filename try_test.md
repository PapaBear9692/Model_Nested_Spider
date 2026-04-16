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
- Console output shows weightedtau and pearsonr per dataset per epoch
