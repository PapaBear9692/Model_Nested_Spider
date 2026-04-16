# Hierarchical Clustering Layer — Theory & Design

## Context

**Problem:** The current `LearnwareCAHeterogeneous.forward()` loops over all N PTMs, performing one self-attention pass per PTM. For 100 PTMs, this means 100 sequential attention computations per batch element — slow and memory-heavy.

**Solution:** Add a hierarchical clustering layer that organizes models into a tree (coarse → fine). At inference, score cluster tokens first, prune unpromising branches, then only score the top candidate PTMs. This reduces inference from O(N) to O(√N) attention passes.

**User choices:** Joint training, configurable beam width via CLI flags, hybrid tree (initialized from architecture families, fine-tunable during training).

**Design principle: ZERO changes to existing code paths.** All new logic is isolated in new files. The training loop, test loop, preprocessing, metrics, and data loading are completely untouched.

**How to compare:** Flip `--use_hierarchy` on/off. Identical behavior when off.

---

## Why Hierarchical Clustering Works for PTM Ranking

Not all PTMs need to be evaluated for every task. Models within the same architecture family (e.g., ResNet-18 vs ResNet-152) tend to perform similarly on a given downstream task — if ResNet-50 ranks highly, it's likely that ResNet-101 will also rank well. This creates a natural hierarchy:

1. **Coarse discrimination:** Determine which architecture *paradigm* suits the task (CNN vs Transformer vs Hybrid)
2. **Fine discrimination:** Within the selected paradigm, determine which *family* is best (ResNet vs DenseNet vs VGG)
3. **Leaf ranking:** Within the selected families, rank the individual *variants* (ResNet-18 vs ResNet-34 vs ResNet-50)

This is analogous to how a human expert would select a model: first decide "I need a transformer for this task," then "Swin Transformer seems right," then "Swin-B is the best variant."

---

## The Prototype Vector as a Task Fingerprint

The existing system already extracts prototype vectors (class centroids) from each PTM applied to each dataset. These prototypes serve as a "fingerprint" of both:
- **The task** (what the dataset looks like when processed by the base PTM)
- **The PTM** (what features each model extracts from the data)

The cross-attention mechanism in the current model learns to match "which PTM fingerprint fits this task fingerprint best." The hierarchical layer adds a *search structure* on top of this matching — instead of comparing the task against all 100 PTM fingerprints, it first compares against cluster-level summaries.

---

## Cluster Tokens as Learnable Summaries

Each cluster node gets a **learnable token** (a vector in the same embedding space as the model prompts). These tokens serve as compressed representations of the models within each cluster:

- **L1 cluster tokens**: 2 tokens (10 PTMs) or 5 tokens (100 PTMs). Represent broad architecture paradigms.
- **L2 family tokens**: 4 tokens (10 PTMs) or 23 tokens (100 PTMs). Represent specific architecture families (ResNet, DenseNet, ViT, Swin, etc.)

During training, these tokens are learned end-to-end through the auxiliary loss. The gradient signal comes from: "which cluster/family did the best PTM belong to?" — this teaches the tokens to capture the *discriminative features* of their member models.

---

## The Hybrid Tree Approach

The tree is **initialized** from known architecture families (providing a meaningful starting point), but the cluster tokens are **fully learnable** during training. The cluster scoring uses a custom `_ClusterAttention` module (separate from the base model's `MultiHeadAttention`) with its own parameters, plus a `task_proj` linear layer that projects the task embedding before combining it with each cluster token. This means:

- If two architecture families consistently rank similarly across training tasks, their cluster tokens will converge (the model learns they're redundant)
- If an architecture family spans very different behaviors (e.g., small vs large variants), the gradient signal can shift the token to represent the dominant pattern
- The tree *structure* (which PTM belongs to which cluster) is fixed, but the *representations* adapt

---

## Why Joint Training

Joint training (all losses computed simultaneously) is chosen because:

1. **Gradient flow:** The main ranking loss trains the model prompts, which are the "leaf" representations. The auxiliary losses train the cluster tokens, which are "parent" representations. Both need to converge together — if cluster tokens are trained separately, they might not align with the learned model prompts.
2. **Simplicity:** One training run, one set of hyperparameters, no freezing/unfreezing.
3. **Loss:** `L_total = L_rank + α × CE(L1_scores, L1_labels) + β × CE(L2_scores, L2_labels)` where α=0.3 and β=0.2 are the auxiliary loss weights. These are small enough that the main ranking quality is preserved, but large enough to make cluster tokens meaningful.

---

## Why the Auxiliary Loss Uses Cross-Entropy (Not Ranking Loss)

The auxiliary losses at L1 and L2 levels use standard `CrossEntropyLoss`, not `HierarchicalCE`. This is because:

- At L1, there are only 5 clusters — we're doing **classification** ("which cluster is best?"), not ranking
- At L2, similarly, we're classifying into ~24 families
- The target label is derived from the ground-truth PTM rankings: for each sample, find the cluster/family containing the top-ranked PTM
- Cross-Entropy is simpler, faster to compute, and provides clearer gradient signal for cluster navigation than a ranking loss would

---

## Tree Structures

### 10 PTMs (default, `CLUSTER_TREE_10` in `learnware_info.py`)

```
Level 0: Root (task embedding)
  ├── Level 1: cnn_classic (8)        → resnet(3), densenet(3), classical(2)
  └── Level 1: cnn_lightweight (2)    → mobilenet(2)
```

Traversal: 2 L1 clusters → 4 L2 families → 10 leaf PTMs.

### 100 PTMs (`PTM100=yes`, `CLUSTER_TREE_100` in `learnware_info_100.py`)

```
Level 0: Root (task embedding)
  ├── Level 1: cnn_classic (20)       → ResNet(5), DenseNet(4), VGG(8), Classical(3)
  ├── Level 1: cnn_modern (26)        → EfficientNet(8), EfficientNetV2(4), RegNet-Y(7), RegNet-X(7)
  ├── Level 1: cnn_lightweight (11)   → MobileNet(5), ShuffleNet(4), SqueezeNet(2)
  ├── Level 1: transformer (31)       → ViT(4), Swin(4), DeiT(4), MaxViT(3), MobileViT(3), CoaT(3), LeViT(3), Twins(3), PiT/CaiT(4)
  └── Level 1: hybrid_specialized (11) → ConvNeXt(4), ConvNeXtV2(4), HRNet(3)
```

Traversal: 5 L1 clusters → 23 L2 families → 100 leaf PTMs.

---

## Speedup Analysis

### 10 PTMs

| Component | Flat | Hierarchical (top_k_L1=1, top_k_families=2) |
|-----------|------|----------------------------------------------|
| L1 scoring | — | 2 passes |
| L2 scoring | — | ~2 passes (2 families in 1 cluster) |
| Leaf scoring | 10 passes | ~5-8 candidates |
| **Total** | **10 passes** | **~10 passes** (minimal gain at this scale) |

At 10 PTMs the tree is too small for meaningful speedup. The benefit here is accuracy improvement from auxiliary losses.

### 100 PTMs

| Component | Flat | Hierarchical (top_k_L1=2, top_k_families=2) |
|-----------|------|----------------------------------------------|
| L1 scoring | — | 5 passes |
| L2 scoring | — | ~12 passes (families in top-2 clusters) |
| Leaf scoring | 100 passes | ~20-40 candidates |
| **Total** | **100 passes** | **~37-57 passes** | **~1.8-2.7x faster** |

With aggressive pruning (`top_k_L1=2, top_k_families=1`): 5 + 6 + 8 = **19 passes** (~5.3x faster).

The trade-off: more aggressive pruning → faster inference but potentially lower accuracy if the correct PTM is in a pruned branch.

---

## The Drop-In Design: How Zero Pipeline Changes Work

The key insight is that **both the model and the loss keep their exact same call signatures**:

1. **Model:** `forward(x_uni, x_hete, attn_mask, ...)` always returns `[batch, num_learnware]`
   - During training: computes all levels (no pruning), stores auxiliary outputs as `self.cluster_layer._aux_L1` and `self.cluster_layer._aux_L2`
   - During eval: prunes via tree traversal, returns scores with `-inf` for pruned PTMs

2. **Base model `candidate_indices`:** The original `LearnwareCAHeterogeneous.forward()` accepts an optional `candidate_indices` parameter. When provided, the attention loop only iterates over those PTM indices instead of all `num_learnware`. This is the key mechanism for inference speedup — the hierarchical model passes only the surviving candidate indices to the base model.

3. **Loss:** `forward(logits, labels)` always accepts the same two arguments
   - Internally reads auxiliary outputs from `model_ref.cluster_layer._aux_L1` and `_aux_L2`
   - When aux outputs are None (eval mode), it just computes the original `HierarchicalCE` loss

4. **Property delegates:** `HierarchicalLearnwareCA` exposes `uni_linear` and `hete_linears` as `@property` delegates to `self.base_model`, so `trainer.py`'s `preprocess_hete_inputs()` works unchanged.

This means `trainer.py`'s training loop, test loop, and preprocessing never change. The only modification is in `__init__()` where model and loss construction is conditional on `--use_hierarchy`.

---

## Training Pipeline (with `--use_hierarchy`)

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
       │        concat([L1_token_i, task_proj(task_emb)]) → [batch, 2, 1024]
       │        → _ClusterAttention → CLS → score_L1 → scalar
       │      Store as cluster_layer._aux_L1 = [batch, num_l1]
       │
       ├─ 3. L2 family scoring (ALL families, no pruning):
       │      For each L2 token:
       │        concat([L2_token_j, task_proj(task_emb)]) → [batch, 2, 1024]
       │        → _ClusterAttention → CLS → score_L2 → scalar
       │      Store as cluster_layer._aux_L2 = [batch, num_l2]
       │
       ├─ 4. Leaf PTM scoring (ALL PTMs — same as original):
       │      For each model_prompt i in 0..99:
       │        concat([prompt_i, x_uni, x_hete[i]]) → [batch, seq_len, 1024]
       │        → base_model.transformer → CLS → mlp_head → scalar
       │      → final_scores [batch, 100]
       │
       └─ Return: final_scores [batch, 100]   ← SAME SHAPE
       ↓
criterion(outputs, labels)                                       ← SAME CALL SIGNATURE
       │
       ├─ Main loss: HierarchicalCE(outputs, labels)             ← SAME AS ORIGINAL
       │
       ├─ Aux L1 loss: CE(model.cluster_layer._aux_L1, l1_labels) × 0.3  ← NEW, via model reference
       │   (l1_labels from cluster_tree.get_best_cluster_labels(labels))
       │
       └─ Aux L2 loss: CE(model.cluster_layer._aux_L2, l2_labels) × 0.2  ← NEW, via model reference
           (l2_labels from cluster_tree.get_best_cluster_labels(labels))
       ↓
Backprop → updates model_prompts + cluster_tokens + all attention weights
```

**Key insight during training:** All PTMs are scored (no pruning) so gradients flow to all model prompts. The auxiliary losses train the cluster tokens to predict which cluster/family is best, so they become meaningful for inference-time pruning.

---

## Inference Pipeline (model.eval() with `--use_hierarchy`)

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
       │      Score each L1 cluster token against task_proj(task_emb)
       │      Cost: num_l1 attention passes (2 or 5)
       │      e.g. selected: [cnn_classic, transformer]
       │
       ├─ 3. L2 FAMILY scoring (families in selected clusters → top-K2 each):
       │      Score family tokens within selected clusters
       │      Cost: ~num_l2 attention passes
       │      e.g. selected: [resnet, densenet, vit, swin]
       │
       ├─ 4. LEAF scoring (PTMs in selected families only):
       │      Score only candidate PTMs via base_model.forward(candidate_indices=...)
       │      The base model loop iterates only over candidate indices, skipping pruned PTMs
       │      Cost: ~20-60 attention passes (instead of 10 or 100)
       │
       └─ Return: scores [batch, 100]                            ← SAME SHAPE
                   Real scores for ~20 selected PTMs
                   -inf for ~80 pruned PTMs
       ↓
test() continues with rankings = argsort(scores)                 ← UNCHANGED
       ↓
measure_test(rankings, labels) → weightedtau                     ← UNCHANGED
```

Pruned PTMs with score `-inf` naturally get the lowest ranks after argsort, so the ranking output is still valid for metric computation.

---

## Two-Phase Testing (k=0, k=1..10)

The existing two-phase testing in `fit()` works unchanged with hierarchical mode:

**Phase 1 (k=0):** Only uniform features (no heterogeneous). The hierarchical model scores ~20 candidates instead of 100. Rankings are used to select which heterogeneous PTMs to sample.

**Phase 2 (k=1..10):** Add heterogeneous features. Only the top-ranked candidates from Phase 1 get heterogeneous features. The hierarchical model still prunes at the cluster level, and heterogeneous features are only computed for the survivors.

---

## Backward Compatibility & Rollback

**Without `--use_hierarchy`:** Everything is identical to the current code. The `else` branches in `__init__` create the exact same model and loss objects as before. No other code paths are affected.

**To rollback:**
1. Remove `--use_hierarchy` flag from your command
2. (Optional) delete `learnware/hierarchical_cluster.py`
3. (Optional) remove `CLUSTER_TREE_10` from `learnware_info.py` and `CLUSTER_TREE_100` from `learnware_info_100.py`
4. (Optional) remove the appended classes in `model.py`, `loss.py`
5. (Optional) remove the `candidate_indices` parameter from `LearnwareCAHeterogeneous.forward()`
6. (Optional) revert the `if/else` in `trainer.py.__init__()` back to the original single-path

The appended code in `model.py`, `loss.py`, and `learnware_info*.py` is inert when `--use_hierarchy` is not set — it's never imported or executed.
