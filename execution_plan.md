# Hierarchical Clustering Layer — Implementation Plan

## Overview

Add a hierarchical clustering layer that organizes 100 PTMs into a coarse-to-fine tree for faster inference.

**Design principle: ZERO changes to existing code paths.** All new logic is isolated in new files. The only change in existing files is a conditional `if args.use_hierarchy:` guard in `trainer.py.__init__()` that swaps the model and loss objects. The training loop, test loop, preprocessing, metrics, and data loading are completely untouched.

**How to compare:** Flip `--use_hierarchy` on/off. Identical behavior when off.

---

## What Changes Where

| File | Change Type | Lines Touched |
|------|-------------|---------------|
| `learnware/learnware_info.py` | APPEND `CLUSTER_TREE_10` dict at end | ~12 lines added |
| `learnware/learnware_info_100.py` | APPEND `CLUSTER_TREE_100` dict at end | ~60 lines added |
| `learnware/hierarchical_cluster.py` | **NEW FILE** | ~310 lines |
| `learnware/model.py` | APPEND new class + MODIFY `forward()` signature | ~100 lines added |
| `learnware/loss.py` | APPEND new class after existing ones | ~35 lines added |
| `trainer.py` | MODIFY `__init__` + ADD timing/resource output | ~25 lines changed |

**NOT touched:** `dataset.py`, `utils.py`, `preprocess_hete_inputs()`, `test()`, `measure_test()`, `collate_fn()`, `mptms/`, `datasets/`.

---

## Step 1: Define Cluster Tree Configs

### File: `learnware/learnware_info.py` — APPEND ~12 lines at end

Add a `CLUSTER_TREE_10` dictionary for the 10-PTM mode:

```python
CLUSTER_TREE_10 = {
    'cnn_classic': {
        'resnet': ['resnet50', 'resnet101', 'resnet152'],
        'densenet': ['densenet121', 'densenet169', 'densenet201'],
        'classical': ['googlenet', 'inception_v3'],
    },
    'cnn_lightweight': {
        'mobilenet': ['mobilenet_v2', 'mnasnet1_0'],
    },
}
```

Total: 2 L1 clusters, 4 L2 families, 10 leaf PTMs.

### File: `learnware/learnware_info_100.py` — APPEND ~60 lines at end (before `if is_100_mode():`)

Add a `CLUSTER_TREE_100` dictionary:

```python
CLUSTER_TREE_100 = {
    'cnn_classic': {
        'resnet': ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'],
        'densenet': ['densenet121', 'densenet161', 'densenet169', 'densenet201'],
        'vgg': ['vgg11', 'vgg13', 'vgg16', 'vgg19',
                'vgg11_bn', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn'],
        'classical': ['googlenet', 'inception_v3', 'alexnet'],
    },
    'cnn_modern': {
        'efficientnet': ['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2',
                         'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5',
                         'efficientnet_b6', 'efficientnet_b7'],
        'efficientnetv2': ['efficientnetv2_s', 'efficientnetv2_m',
                           'efficientnetv2_l', 'efficientnetv2_xl'],
        'regnet_y': ['regnet_y_400mf', 'regnet_y_800mf', 'regnet_y_1_6gf',
                      'regnet_y_3_2gf', 'regnet_y_8gf', 'regnet_y_16gf', 'regnet_y_32gf'],
        'regnet_x': ['regnet_x_400mf', 'regnet_x_800mf', 'regnet_x_1_6gf',
                      'regnet_x_3_2gf', 'regnet_x_8gf', 'regnet_x_16gf', 'regnet_x_32gf'],
    },
    'cnn_lightweight': {
        'mobilenet': ['mobilenet_v2', 'mobilenet_v3_small', 'mobilenet_v3_large',
                      'mnasnet0_5', 'mnasnet1_0'],
        'shufflenet': ['shufflenet_v2_x0_5', 'shufflenet_v2_x1_0',
                       'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0'],
        'squeezenet': ['squeezenet1_0', 'squeezenet1_1'],
    },
    'transformer': {
        'vit': ['vit_b_16', 'vit_b_32', 'vit_l_16', 'vit_l_32'],
        'swin': ['swin_t', 'swin_s', 'swin_b', 'swin_v2_t'],
        'deit': ['deit_tiny_patch16_224', 'deit_small_patch16_224',
                 'deit_base_patch16_224', 'deit_base_patch16_384'],
        'maxvit': ['maxvit_tiny_224', 'maxvit_small_224', 'maxvit_base_224'],
        'mobilevit': ['mobilevit_s', 'mobilevit_xs', 'mobilevit_xxs'],
        'coat': ['coat_lite_tiny', 'coat_lite_small', 'coat_lite_medium'],
        'levit': ['levit_128s', 'levit_128', 'levit_192'],
        'twins': ['twins_pcpvt_small', 'twins_pcpvt_base', 'twins_pcpvt_large'],
        'pit_cait_tnt': ['tnt_s_patch16_224', 'pit_b_224', 'pit_s_224', 'cait_s24_224'],
    },
    'hybrid_specialized': {
        'convnext': ['convnext_tiny', 'convnext_small', 'convnext_base', 'convnext_large'],
        'convnextv2': ['convnextv2_atto', 'convnextv2_femto', 'convnextv2_nano', 'convnextv2_tiny'],
        'hrnet': ['hrnet_w18_small', 'hrnet_w18', 'hrnet_w30'],
    },
}
```

Total: 5 L1 clusters, 23 L2 families, 100 leaf PTMs.
Assert `sum of all leaves == 100` and `all names in BKB_100_SPECIFIC_RANK`.

---

## Step 2: Create the Hierarchical Module (NEW FILE)

### File: `learnware/hierarchical_cluster.py` — NEW, ~300 lines

Contains 3 components:

### 2a. `ClusterTree` class (~120 lines)

Static tree data structure. No learnable parameters. Precomputes index tensors for vectorized inference.

```python
class ClusterTree:
    """Parses CLUSTER_TREE_10/100 into index structures."""

    def __init__(self, tree_config: dict, all_models: list):
        # Build ordered lists
        self.level1_names = list(tree_config.keys())
        self.level2_names = []                                  # flattened across all L1 clusters
        self.leaf_names = all_models                            # from BKB_SPECIFIC_RANK

        # Build index maps
        self.level1_to_idx = {n: i for i, n in enumerate(self.level1_names)}
        self.level2_to_idx = {}
        self.leaf_to_idx = {n: i for i, n in enumerate(self.leaf_names)}

        # leaf_to_path: PTM name → (L1_idx, L2_idx, leaf_idx_within_family)
        self.leaf_to_path = {}

        # l1_to_families: L1_idx → [L2_idx, ...]
        # family_to_leaves: L2_idx → [global_leaf_idx, ...]

        # ... populate from tree_config ...
        # ... then call _build_tensors() for precomputed padded tensors ...

    def get_families(self, l1_idx):
        """Return L2 family indices for a given L1 cluster."""
        ...

    def get_leaves(self, l2_idx):
        """Return global PTM indices for a given L2 family."""
        ...

    def get_best_cluster_labels(self, labels):
        """Derive L1 and L2 labels from ground-truth rankings.

        Args:
            labels: [batch, num_learnware] — higher value = better rank
        Returns:
            l1_labels: [batch] — index of L1 cluster containing best PTM
            l2_labels: [batch] — index of L2 family containing best PTM
        """
        best_ptm_idx = labels.argmax(dim=-1)
        paths = self._leaf_path_tensor.to(labels.device)[best_ptm_idx]
        return paths[:, 0], paths[:, 1]
```

### 2b. `HierarchicalClusterLayer(nn.Module)` (~150 lines)

Learnable cluster navigation. Contains cluster tokens, a task projection layer, and separate `_ClusterAttention` modules.

```python
class HierarchicalClusterLayer(nn.Module):
    def __init__(self, cluster_tree: ClusterTree, dim: int, heads: int = 1, dropout: float = 0.1):
        super().__init__()
        self.tree = cluster_tree
        self.dim = dim

        # Learnable tokens for each tree level
        self.cluster_tokens_L1 = nn.Parameter(torch.randn(1, cluster_tree.num_l1, dim))
        self.cluster_tokens_L2 = nn.Parameter(torch.randn(1, cluster_tree.num_l2, dim))

        # Task projection (projects task embedding before combining with cluster tokens)
        self.task_proj = nn.Linear(dim, dim)

        # Separate attention + scoring for L1 and L2
        # Uses custom _ClusterAttention (not MultiHeadAttention from base model)
        self.attn_L1 = _ClusterAttention(dim, heads, dropout)
        self.attn_L2 = _ClusterAttention(dim, heads, dropout)
        self.score_L1 = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))
        self.score_L2 = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))

        # Stored auxiliary outputs (set during training forward)
        self._aux_L1 = None
        self._aux_L2 = None
```

Key methods:

**`_score_level(tokens, task_emb, attn_module, score_head)`** — Generic scoring for one tree level:
```
For each token:
    task_proj_emb = task_proj(task_emb)  → project task embedding
    stack([tokens, task_proj_emb]) → [batch*num_tokens, 2, dim]
    attended = _ClusterAttention(combined)  → [batch*num_tokens, 2, dim]
    cls = attended[:, 0]  → [batch*num_tokens, dim]
    score = score_head(cls) → [batch, num_tokens]
Return: scores [batch, num_tokens]
```

**`forward_training(task_emb, base_model, x_uni, x_hete, attn_mask, attn_mask_func)`**:
```
1. Clear stale aux outputs
2. Score ALL L1 clusters → self._aux_L1 = [batch, num_l1]  (stored on self)
3. Score ALL L2 families → self._aux_L2 = [batch, num_l2]  (stored on self)
4. Score ALL PTMs via base_model.forward() → final_scores [batch, num_learnware]
Return: final_scores [batch, num_learnware]   ← SAME SHAPE AS ORIGINAL
```

**`forward_inference(task_emb, base_model, x_uni, x_hete, attn_mask, attn_mask_func, top_k_L1, top_k_families)`**:
```
1. Score all L1 clusters → select top_k_L1
2. Score families in selected clusters → select top_k_families per cluster
3. Build candidate leaf set per batch element
4. Find union of candidates across batch
5. Score only candidate PTMs via base_model.forward(candidate_indices=sorted_candidates)
6. Map candidate scores back to full [batch, num_learnware] tensor, fill -inf for pruned
Return: scores [batch, num_learnware]   ← SAME SHAPE AS ORIGINAL
```

**Critical:** Both methods return `[batch, num_learnware]` — identical to `LearnwareCAHeterogeneous.forward()`.

### 2c. `_ClusterAttention(nn.Module)` (~40 lines)

Custom self-attention module for cluster scoring (separate from the base model's `MultiHeadAttention`):

```python
class _ClusterAttention(nn.Module):
    def __init__(self, dim, heads=1, dropout=0.1):
        # Standard multi-head attention with LayerNorm + residual
        self.w_qs = nn.Linear(dim, heads * dim, bias=False)
        self.w_ks = nn.Linear(dim, heads * dim, bias=False)
        self.w_vs = nn.Linear(dim, heads * dim, bias=False)
        self.fc = nn.Linear(heads * dim, dim)
        # ... Xavier initialization ...
```

---

## Step 3: Add Hierarchical Model Wrapper + Modify Base Model Forward

### File: `learnware/model.py` — MODIFY `LearnwareCAHeterogeneous.forward()` + APPEND ~100 lines

#### 3a. Modify `LearnwareCAHeterogeneous.forward()` signature

Add an optional `candidate_indices` parameter. When provided (during hierarchical inference), the attention loop only iterates over those indices instead of all `num_learnware`:

```python
def forward(self, x_uni, x_hete, attn_mask, attn_mask_func=None,
            permute_indices=None, candidate_indices=None):
    # ...
    prompt_range = candidate_indices if candidate_indices is not None else range(model_prompt.shape[1])
    for i_prompt in prompt_range:
        # ... existing attention logic unchanged ...
```

This is the key mechanism for inference speedup — pruned PTMs are never scored.

#### 3b. Append `HierarchicalLearnwareCA` class

Add a new class after `LearnwareCAHeterogeneous`. The existing class is otherwise NOT modified.

```python
class HierarchicalLearnwareCA(nn.Module):
    """
    Drop-in replacement for LearnwareCAHeterogeneous.

    Same forward() signature. Same return shape [batch, num_learnware].
    During training: stores auxiliary outputs as _aux_L1, _aux_L2 on self
                     (used by HierarchicalClusterLoss via model reference).
    During eval:     prunes using tree traversal, returns [batch, num_learnware]
                     with -inf for pruned PTMs.
    """
    def __init__(self, *,
                 num_learnware, dim, hdim, heads,
                 uni_hete_proto_dim, data_sub_url,
                 cluster_tree_config, all_models,
                 top_k_L1=3, top_k_families=2,
                 pool='cls', dropout=0.1, emb_dropout=0.1,
                 heterogeneous_extra_prompt=False):

        super().__init__()

        # Base model — exact same construction as the original
        self.base_model = LearnwareCAHeterogeneous(
            num_learnware=num_learnware, dim=dim, hdim=hdim,
            heads=heads, uni_hete_proto_dim=uni_hete_proto_dim,
            data_sub_url=data_sub_url, pool=pool,
            dropout=dropout, emb_dropout=emb_dropout,
            heterogeneous_extra_prompt=heterogeneous_extra_prompt
        )

        # Cluster tree and layer
        from learnware.hierarchical_cluster import ClusterTree, HierarchicalClusterLayer
        self.cluster_tree = ClusterTree(cluster_tree_config, all_models)
        self.cluster_layer = HierarchicalClusterLayer(
            cluster_tree=self.cluster_tree, dim=dim, heads=heads
        )

        self.top_k_L1 = top_k_L1
        self.top_k_families = top_k_families

    @property
    def uni_linear(self):
        """Delegate to base_model so preprocess_hete_inputs() works unchanged."""
        return self.base_model.uni_linear

    @property
    def hete_linears(self):
        """Delegate to base_model so preprocess_hete_inputs() works unchanged."""
        return self.base_model.hete_linears

    def forward(self, x_uni, x_hete, attn_mask, attn_mask_func=None, permute_indices=None):
        """Same signature and return shape as LearnwareCAHeterogeneous.forward()."""
        task_emb = x_uni.mean(dim=1, keepdim=True)  # [batch, 1, dim]

        if self.training:
            # Full scoring — no pruning. Store aux outputs for loss.
            return self.cluster_layer.forward_training(
                task_emb, self.base_model, x_uni, x_hete,
                attn_mask, attn_mask_func
            )
            # _aux_L1 and _aux_L2 are set inside cluster_layer
        else:
            # Pruned inference — only score candidates
            return self.cluster_layer.forward_inference(
                task_emb, self.base_model, x_uni, x_hete,
                attn_mask, attn_mask_func,
                self.top_k_L1, self.top_k_families
            )
```

**Why this is safe:** The base_model (`LearnwareCAHeterogeneous`) is created with the exact same arguments as the original. Its weights, prompts, attention modules are all there. The `forward()` method takes the same arguments and returns `[batch, num_learnware]`.

---

## Step 4: Add Combined Loss

### File: `learnware/loss.py` — APPEND ~35 lines after existing loss classes

```python
class HierarchicalClusterLoss(nn.Module):
    """
    Drop-in replacement for HierarchicalCE.

    Same forward(logits, labels) signature.
    Reads auxiliary outputs from model.cluster_layer._aux_L1 and _aux_L2.
    During eval (aux outputs are None), computes only the main HierarchicalCE loss.
    """
    def __init__(self, num_learnware, cluster_tree, alpha=0.3, beta=0.2):
        super().__init__()
        self.main_loss = HierarchicalCE(num_learnware)
        self.cluster_tree = cluster_tree
        self.alpha = alpha
        self.beta = beta
        self.model_ref = None  # Set once by trainer

    def set_model(self, model):
        """Wire up model reference so loss can read cluster_layer._aux_L1, _aux_L2."""
        self.model_ref = model

    def forward(self, logits, labels):
        # Main ranking loss — always computed
        loss = self.main_loss(logits, labels)

        # Auxiliary cluster losses — only when model stored aux outputs (training)
        if (self.model_ref is not None
                and hasattr(self.model_ref, 'cluster_layer')
                and self.model_ref.cluster_layer._aux_L1 is not None):

            l1_labels, l2_labels = self.cluster_tree.get_best_cluster_labels(labels)
            loss = loss + self.alpha * F.cross_entropy(
                self.model_ref.cluster_layer._aux_L1, l1_labels)
            loss = loss + self.beta * F.cross_entropy(
                self.model_ref.cluster_layer._aux_L2, l2_labels)

        return loss
```

**Why this is safe:** `forward(logits, labels)` — same signature as `HierarchicalCE`. During eval, `model_ref.cluster_layer._aux_L1` is None, so it just computes the main loss (identical to `HierarchicalCE`). During training, it reads the aux outputs from `model_ref.cluster_layer` (where `HierarchicalClusterLayer.forward_training()` stored them).

---

## Step 5: Modify Trainer `__init__` + Add Timing Output

### File: `trainer.py` — MODIFY `__init__()` + ADD timing in `fit()`, ~25 lines changed

### 5a. New CLI flags in `parse_trainer_args` (append after line 61)

```python
# Hierarchical clustering config (all opt-in, no effect when not set)
parser.add_argument('--use_hierarchy', action='store_true', default=False)
parser.add_argument('--hier_top_k_L1', type=int, default=3)
parser.add_argument('--hier_top_k_families', type=int, default=2)
parser.add_argument('--hier_alpha', type=float, default=0.3)
parser.add_argument('--hier_beta', type=float, default=0.2)
```

### 5b. Conditional model + loss in `__init__` (replace lines 91-102)

**Before (original code, preserved):**
```python
self.model = LearnwareCAHeterogeneous(
    num_learnware=args.num_learnware, ...)
self.model = self.model.to(torch.device('cuda'))
```

**After:**
```python
if args.use_hierarchy:
    from learnware.model import HierarchicalLearnwareCA
    from learnware.learnware_info import CLUSTER_TREE_10

    # Pick cluster tree based on PTM count
    _cluster_tree = CLUSTER_TREE_10
    _all_models = list(BKB_SPECIFIC_RANK)
    if os.environ.get('PTM100', '').lower() == 'yes':
        from learnware.learnware_info_100 import CLUSTER_TREE_100
        _cluster_tree = CLUSTER_TREE_100

    self.model = HierarchicalLearnwareCA(
        num_learnware=args.num_learnware,
        dim=args.dim, hdim=args.dim,
        uni_hete_proto_dim=(args.prototype_maxnum, args.prototype_maxnum_hete),
        data_sub_url=args.data_sub_url,
        cluster_tree_config=_cluster_tree,
        all_models=_all_models,
        top_k_L1=args.hier_top_k_L1,
        top_k_families=args.hier_top_k_families,
        pool=args.attn_pool, heads=1, dropout=0.1, emb_dropout=0.1,
        heterogeneous_extra_prompt=args.heterogeneous_extra_prompt
    )
    logging.info(f'[Hierarchical Mode] L1={self.model.cluster_tree.num_l1}, '
                 f'L2={self.model.cluster_tree.num_l2}, leaves={self.model.cluster_tree.num_leaves}')
else:
    self.model = LearnwareCAHeterogeneous(
        num_learnware=args.num_learnware,
        dim=args.dim, hdim=args.dim,
        uni_hete_proto_dim=(args.prototype_maxnum, args.prototype_maxnum_hete),
        data_sub_url=args.data_sub_url,
        pool=args.attn_pool, heads=1, dropout=0.1, emb_dropout=0.1,
        heterogeneous_extra_prompt=args.heterogeneous_extra_prompt
    )
self.model = self.model.to(torch.device('cuda'))
```

### 5c. Conditional loss in `__init__` (replace line 142)

**Before:**
```python
self.criterion = HierarchicalCE(args.num_learnware)
```

**After:**
```python
if args.use_hierarchy:
    from learnware.loss import HierarchicalClusterLoss
    self.criterion = HierarchicalClusterLoss(
        args.num_learnware, self.model.cluster_tree,
        alpha=args.hier_alpha, beta=args.hier_beta
    )
    self.criterion.set_model(self.model)
else:
    self.criterion = HierarchicalCE(args.num_learnware)
```

### THAT'S IT FOR TRAINER CHANGES.

The rest of `trainer.py`'s core logic is untouched:
- `test()` method — **NO CHANGES**
- `preprocess_hete_inputs()` — **NO CHANGES**
- `get_attn_pad_mask()`, `get_attn_pad_hete_mask()` — **NO CHANGES**

Minor additions to `fit()` (non-functional):
- `import time` at top
- Timestamp format: `{mode}_{phase}_{timestamp}` (e.g., `hier_train_20260416-193500`)
- Timing/resource output: overall training time, inference time, peak GPU memory, GPU name
- These are output-only and do not affect model behavior

---

## How the Pipeline Works (End-to-End)

### During Training (with `--use_hierarchy`)

```
Input: prototype .pkl files (uniform features + heterogeneous features + ground-truth rankings)
       ↓
DataLoader → collate_fn → (uniform, heterogeneous, labels)      ← UNCHANGED
       ↓
preprocess_hete_inputs() → project features to 1024-dim         ← UNCHANGED
       ↓
model.forward(x_uni, x_hete, attn_mask, ...)                    ← SAME CALL SIGNATURE
       │
       ├─ 1. Build task embedding:
       │      task_emb = mean(x_uni, dim=1) → [batch, 1, 1024]
       │
       ├─ 2. L1 cluster scoring (ALL clusters, no pruning):
       │      For each L1 token:
       │        stack([L1_token_i, task_proj(task_emb)]) → [batch*2, 2, 1024]
       │        → _ClusterAttention → CLS → score_L1 → scalar
       │      Store as cluster_layer._aux_L1 = [batch, num_l1]
       │
       ├─ 3. L2 family scoring (ALL families, no pruning):
       │      For each L2 token:
       │        stack([L2_token_j, task_proj(task_emb)]) → [batch*2, 2, 1024]
       │        → _ClusterAttention → CLS → score_L2 → scalar
       │      Store as cluster_layer._aux_L2 = [batch, num_l2]
       │
       ├─ 4. Leaf PTM scoring (ALL PTMs — same as original):
       │      For each model_prompt i in 0..num_learnware-1:
       │        concat([prompt_i, x_uni, x_hete[i]]) → [batch, seq_len, 1024]
       │        → base_model.transformer → CLS → mlp_head → scalar
       │      → final_scores [batch, num_learnware]
       │
       └─ Return: final_scores [batch, 100]   ← SAME SHAPE
       ↓
criterion(outputs, labels)                                       ← SAME CALL SIGNATURE
       │
       ├─ Main loss: HierarchicalCE(outputs, labels)             ← SAME AS ORIGINAL
       │
       ├─ Aux L1 loss: CE(model.cluster_layer._aux_L1, l1_labels) × 0.3  ← NEW
       │   (l1_labels from cluster_tree.get_best_cluster_labels(labels))
       │
       └─ Aux L2 loss: CE(model.cluster_layer._aux_L2, l2_labels) × 0.2  ← NEW
           (l2_labels from cluster_tree.get_best_cluster_labels(labels))
       ↓
Backprop → updates model_prompts + cluster_tokens + all attention weights
```

### During Inference (model.eval() with `--use_hierarchy`)

```
Input: test prototype .pkl files
       ↓
DataLoader → preprocess_hete_inputs()                            ← UNCHANGED
       ↓
model.forward(x_uni, x_hete, attn_mask, ...)                    ← SAME CALL SIGNATURE
       │
       ├─ 1. Build task embedding:
       │      task_emb = mean(x_uni, dim=1) → [batch, 1, 1024]
       │
       ├─ 2. L1 COARSE scoring (select top-K1 clusters):
       │      Score each L1 token against task_proj(task_emb)
       │      Cost: num_l1 attention passes (2 or 5)
       │
       ├─ 3. L2 FAMILY scoring (families in selected clusters → top-K2 each):
       │      Score family tokens within selected clusters
       │      Cost: varies by num_l2
       │
       ├─ 4. LEAF scoring (PTMs in selected families only):
       │      Score only candidates via base_model.forward(candidate_indices=...)
       │      Base model loop skips pruned PTMs entirely
       │
       └─ Return: scores [batch, num_learnware]                  ← SAME SHAPE
                   Real scores for selected PTMs
                   -inf for pruned PTMs
       ↓
test() continues with rankings = argsort(scores)                 ← UNCHANGED
       ↓
measure_test(rankings, labels) → weightedtau                     ← UNCHANGED
```

**Total inference (100 PTMs): 5 + 12 + 20 = ~37 attention passes** (vs 100 flat)
**With aggressive settings (top-2 L1, top-1 family): ~19 passes** (~5.3x faster)
**Total inference (10 PTMs): 2 + 2 + 5 = ~9 attention passes** (vs 10 flat — minimal gain)

### Two-Phase Testing (k=0, k=1..10)

Works unchanged. `fit()` calls `test()` → `model.eval()` → hierarchical pruning kicks in automatically. The k=0 phase scores only candidate PTMs. The k=1..10 phase adds heterogeneous features only to the survivors.

### Without `--use_hierarchy`

Everything is identical to the current code. The `else` branches in `__init__` create the exact same `LearnwareCAHeterogeneous` and `HierarchicalCE` objects as before. No other code paths are affected.

---

## Rollback

To revert, remove:
1. `--use_hierarchy` flag from your command
2. (Optional) delete `learnware/hierarchical_cluster.py`
3. (Optional) remove `CLUSTER_TREE_10` from `learnware_info.py`
4. (Optional) remove `CLUSTER_TREE_100` from `learnware_info_100.py`
5. (Optional) remove the appended classes in `model.py`, `loss.py`
6. (Optional) remove the `candidate_indices` parameter from `LearnwareCAHeterogeneous.forward()`
7. (Optional) revert the `if/else` in `trainer.py.__init__()` back to the original single-path
8. (Optional) remove timing/resource output from `trainer.py`

The appended code in `model.py`, `loss.py`, and `learnware_info*.py` is inert when `--use_hierarchy` is not set — it's never imported or executed.
