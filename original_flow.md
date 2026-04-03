# Model-Spider Code Flow

## 1. Entry Point: `scripts/test-model-spider.sh`

The shell script calls `python trainer.py` with a bunch of CLI args. Key ones:
- `--train_dataset c86 c59 c16 ...` — which dataset combinations to train on
- `--test_dataset CIFAR10 Caltech101 ...` — downstream tasks to evaluate on
- `--heterogeneous` — enables multi-source model zoo mode
- `--num_learnware 10` — number of PTMs to rank
- `--pretrained_url` — if set, skips training and goes straight to testing

## 2. Boot: `trainer.py:518-534` → `main()`

```
main()
  → get_command_line_parser()     # from utils.py: parses --gpu, --seed, --log_url
  → Trainer.parse_trainer_args()  # adds all model/dataset/training args
  → args = parser.parse_args()
  → set_gpu(), set_seed()         # reproducibility setup
  → Trainer(args)                 # construct the trainer
  → trainer.fit()                 # run the pipeline
```

## 3. Construction: `trainer.py:64-177` → `Trainer.__init__()`

This does **5 key things**:

### a) Compute config (lines 79-85)
Determines `prototype_maxnum` (max classes across all datasets) and `dim` (feature dimension).

### b) Build the model (lines 91-103)
```python
self.model = LearnwareCAHeterogeneous(
    num_learnware=10,      # how many PTMs to rank
    dim=...,               # feature dim
    uni_hete_proto_dim=(max_classes_trainval, max_classes_train),
    ...
)
```

### c) Build datasets (lines 105-138)
```python
LearnwareDataset(args, stype='train')  → loads .pkl prototype files from disk
LearnwareDataset(args, stype='val')
LearnwareDataset(args, stype='test')
```
Each wrapped in a `DataLoader`.

### d) Build optimizer + loss (lines 140-142)
```python
optimizer = Adam(lr=0.00025)
scheduler = CosineAnnealingLR
criterion = HierarchicalCE(num_learnware=10)
```

### e) Precompute attention masks (lines 148-168)
Padding masks for variable-length prototypes.

## 4. Training Loop: `trainer.py:302-401` → `fit()`

If `--pretrained_url` is set, it skips training and jumps to testing. Otherwise:

```
for epoch in 1..30:
    model.train()
    for batch in train_loader:
        ↓
```

### Each batch goes through:

#### Step A — Data loading (`learnware/dataset.py:75-108`)

Each `.pkl` file contains:
- `x[0]` — uniform features: `[num_prototypes, dim]` (prototype vectors from the base PTM)
- `x[1]` — heterogeneous features: `{backbone_name: [num_prototypes, dim]}` (prototype vectors from other PTMs)
- `x[2]` — ground-truth ranking labels: `[num_learnware]`

The dataset pads prototypes to `prototype_maxnum` and returns:
```
(ret_x, sample_hete), labels, dataset_name, pad_length
```

#### Step B — Collation (`learnware/dataset.py:114-148`)

The custom `collate_fn` (used in heterogeneous mode):
1. Randomly samples `k` heterogeneous backbones per sample (k ∈ [0, 10])
2. Organizes them by backbone key → `ret_hete_x`
3. Returns: `(uniform_features, (hete_features, hete_indices)), labels, dataset_names, pad_lengths`

#### Step C — Preprocess (`trainer.py:215-300`) → `preprocess_hete_inputs()`

1. Projects uniform features through `model.uni_linear` (e.g., 1024→1024)
2. Projects each heterogeneous backbone through its own `model.hete_linears[backbone]` (different input dims → 1024)
3. Aligns/pads heterogeneous prototypes to the same length
4. Organizes into `prompt2hete`: a dict mapping each of the 10 PTM slots to their heterogeneous features
5. Returns: `x_uni, prompt2hete, pad_lengths, num_hete_sampled`

#### Step D — Model forward (`learnware/model.py:129-159`)

The core of Model Spider. For each of the 10 PTM slots:

```python
for each PTM slot i:
    cur_prompt = learned_model_prompt[i]        # learnable token
    cur_x = concat([cur_prompt, x_uni, x_hete[i]])  # concatenate task + model features
    cur_x = MultiHeadAttention(cur_x, cur_x, cur_x) # self-attention
    cur_x = cur_x[:, 0]                         # take CLS token (the prompt)
    cur_x = MLP_head(cur_x)                     # → scalar score
    outputs.append(score)
```

Output: `[batch_size, 10]` — one score per PTM.

#### Step E — Loss & update (`learnware/loss.py:7-26`)

`HierarchicalCE` is a **learning-to-rank** loss:
- Takes predicted scores `[batch, 10]` and ground-truth ranks `[batch, 10]`
- Creates a triangular mask to compare pairwise rankings
- Computes negative log-likelihood of correct ranking order

```python
loss = HierarchicalCE(outputs, labels)
loss.backward()
optimizer.step()
```

## 5. Testing: `trainer.py:421-487` → `test()`

After each epoch (and at the end):

```python
model.eval()
for batch in test_loader:
    forward pass → outputs [batch, 10]
    rankings = argsort(outputs)         # convert scores to rankings
    measure_test(rankings, labels)      # compute weightedtau & pearsonr
```

### Two-phase heterogeneous testing (lines 369-397):
1. **k=0**: Test with only uniform (base) features → get initial PTM rankings
2. **k=1..10**: Use those rankings to intelligently select heterogeneous backbones, test again
3. Pick the `k` with best average weightedtau → final result

## 6. Metrics: `utils.py:270-276`

```python
weightedtau = scipy.stats.weightedtau(labels, outputs)  # primary metric
pearsonr = scipy.stats.pearsonr(outputs, labels)[0]      # secondary
```

---

## Visual Flow Summary

```
test-model-spider.sh
        │
        ▼
   trainer.py main()
        │
        ▼
   Trainer.__init__()
   ├── Build LearnwareCAHeterogeneous model
   ├── Load LearnwareDataset (train/val/test)
   ├── Setup Adam + CosineAnnealing + HierarchicalCE
   └── Precompute attention pad masks
        │
        ▼
   trainer.fit()
        │
        ├── TRAINING LOOP (if no --pretrained_url)
        │   for each epoch:
        │     for each batch from train_loader:
        │       │
        │       ▼ dataset.__getitem__()
        │       Load .pkl → (uniform_proto, hete_proto_dict), labels, pad_length
        │       │
        │       ▼ collate_fn()
        │       Sample k random heterogeneous backbones per sample
        │       │
        │       ▼ preprocess_hete_inputs()
        │       Project uniform + heterogeneous features → dim 1024
        │       Pad & align prototypes
        │       │
        │       ▼ model.forward()
        │       For each PTM slot:
        │         concat([learned_prompt, uni_features, hete_features])
        │         → MultiHeadAttention → CLS pool → MLP → scalar score
        │       Output: [batch, num_learnware] scores
        │       │
        │       ▼ HierarchicalCE loss
        │       Learning-to-rank: pairwise ranking NLL
        │       │
        │       ▼ backprop + optimizer.step()
        │
        └── TESTING (after each epoch)
            ├── k=0: rank PTMs using only base features
            ├── k=1..10: re-rank with heterogeneous features
            └── Report weightedtau + pearsonr per dataset
```

---

## Key Insight

**Prototype vectors** (class centroids extracted from each PTM on each dataset) serve as the "fingerprint" of both the PTM and the task. The cross-attention learns to match "which PTM fingerprint fits this task fingerprint best."
