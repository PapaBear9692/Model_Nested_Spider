"""
Hierarchical Clustering Layer for Model Spider.

Organizes PTMs into a coarse-to-fine tree (L1 clusters → L2 families → leaf PTMs).
At inference, prunes unpromising branches to reduce attention passes from O(N) to O(sqrt(N)).

Works with any number of PTMs — the tree structure is passed in at construction time.
"""

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np


# =============================================================================
# Cluster Tree (static data structure, no learnable params)
# =============================================================================

class ClusterTree:
    """Parses a cluster tree config dict into index structures.

    Args:
        tree_config: dict of {L1_name: {L2_name: [leaf_names]}}
        all_models: ordered list of all PTM names (e.g. BKB_SPECIFIC_RANK)
    """

    def __init__(self, tree_config: dict, all_models: list):
        self.raw_config = tree_config
        self.leaf_names = list(all_models)  # ordered list, same as BKB_SPECIFIC_RANK

        # L1 clusters
        self.level1_names = list(tree_config.keys())
        self.level1_to_idx = {n: i for i, n in enumerate(self.level1_names)}

        # L2 families (flattened across all L1 clusters)
        self.level2_names = []
        self.level2_to_l1 = []  # parallel list: which L1 each L2 belongs to
        for l1_name, families in tree_config.items():
            for l2_name in families:
                self.level2_names.append(l2_name)
                self.level2_to_l1.append(self.level1_to_idx[l1_name])
        self.level2_to_idx = {n: i for i, n in enumerate(self.level2_names)}

        # Leaf PTM → path mapping
        self.leaf_to_path = {}  # leaf_name → (l1_idx, l2_idx, leaf_idx_within_family)
        self.family_to_leaves = {}  # l2_idx → [global_leaf_idx, ...]
        self.l1_to_families = {i: [] for i in range(len(self.level1_names))}

        for l2_idx, l2_name in enumerate(self.level2_names):
            l1_idx = self.level2_to_l1[l2_idx]
            self.l1_to_families[l1_idx].append(l2_idx)
            leaf_global_indices = []
            for leaf_name in tree_config[self.level1_names[l1_idx]][l2_name]:
                global_idx = self.leaf_names.index(leaf_name)
                leaf_global_indices.append(global_idx)
                self.leaf_to_path[leaf_name] = (l1_idx, l2_idx, len(leaf_global_indices) - 1)
            self.family_to_leaves[l2_idx] = leaf_global_indices

        self.num_l1 = len(self.level1_names)
        self.num_l2 = len(self.level2_names)
        self.num_leaves = len(self.leaf_names)

        # Precompute as tensors for faster inference
        self._build_tensors()

    def _build_tensors(self):
        """Precompute index tensors for vectorized inference."""
        # family_to_leaves as padded tensor: [num_l2, max_family_size]
        max_family_size = max(len(v) for v in self.family_to_leaves.values())
        family_leaves_padded = torch.full((self.num_l2, max_family_size), -1, dtype=torch.long)
        family_leaves_mask = torch.zeros(self.num_l2, max_family_size, dtype=torch.bool)
        for l2_idx, leaves in self.family_to_leaves.items():
            family_leaves_padded[l2_idx, :len(leaves)] = torch.tensor(leaves)
            family_leaves_mask[l2_idx, :len(leaves)] = True

        # l1_to_families as padded tensor: [num_l1, max_families_per_l1]
        max_families = max(len(v) for v in self.l1_to_families.values())
        l1_families_padded = torch.full((self.num_l1, max_families), -1, dtype=torch.long)
        l1_families_mask = torch.zeros(self.num_l1, max_families, dtype=torch.bool)
        for l1_idx, families in self.l1_to_families.items():
            l1_families_padded[l1_idx, :len(families)] = torch.tensor(families)
            l1_families_mask[l1_idx, :len(families)] = True

        # leaf_to_path as tensor: [num_leaves, 2] → (l1_idx, l2_idx)
        leaf_path_tensor = torch.zeros(self.num_leaves, 2, dtype=torch.long)
        for leaf_name, (l1_idx, l2_idx, _) in self.leaf_to_path.items():
            global_idx = self.leaf_names.index(leaf_name)
            leaf_path_tensor[global_idx] = torch.tensor([l1_idx, l2_idx])

        self.register_buffer = False  # not an nn.Module, store plain
        self._family_leaves_padded = family_leaves_padded
        self._family_leaves_mask = family_leaves_mask
        self._l1_families_padded = l1_families_padded
        self._l1_families_mask = l1_families_mask
        self._leaf_path_tensor = leaf_path_tensor
        self._max_family_size = max_family_size

    def get_families(self, l1_idx):
        """Return L2 family indices for a given L1 cluster."""
        return self.l1_to_families[l1_idx]

    def get_leaves(self, l2_idx):
        """Return global PTM indices for a given L2 family."""
        return self.family_to_leaves[l2_idx]

    def get_best_cluster_labels(self, labels):
        """Derive L1 and L2 labels from ground-truth rankings.

        Args:
            labels: [batch, num_learnware] — higher value = better rank

        Returns:
            l1_labels: [batch] — index of L1 cluster containing best PTM
            l2_labels: [batch] — index of L2 family containing best PTM
        """
        best_ptm_idx = labels.argmax(dim=-1)
        paths = self._leaf_path_tensor.to(labels.device)[best_ptm_idx]  # [batch, 2]
        return paths[:, 0], paths[:, 1]


# =============================================================================
# Hierarchical Cluster Layer (learnable)
# =============================================================================

class HierarchicalClusterLayer(nn.Module):
    """Learnable cluster navigation layer.

    Contains cluster tokens and separate attention+scoring modules for L1 and L2.
    During training: scores all clusters (no pruning), stores aux outputs.
    During inference: prunes using top-K selection at each level.
    """

    def __init__(self, cluster_tree: ClusterTree, dim: int, heads: int = 1, dropout: float = 0.1):
        super().__init__()
        self.tree = cluster_tree
        self.dim = dim

        # Learnable cluster tokens
        self.cluster_tokens_L1 = nn.Parameter(torch.randn(1, cluster_tree.num_l1, dim))
        self.cluster_tokens_L2 = nn.Parameter(torch.randn(1, cluster_tree.num_l2, dim))

        # Task projection (projects task embedding before cluster scoring)
        self.task_proj = nn.Linear(dim, dim)

        # Attention + scoring for L1
        self.attn_L1 = _ClusterAttention(dim, heads, dropout)
        self.score_L1 = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))

        # Attention + scoring for L2
        self.attn_L2 = _ClusterAttention(dim, heads, dropout)
        self.score_L2 = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))

        # Stored auxiliary outputs (set during training forward)
        self._aux_L1 = None
        self._aux_L2 = None

    def _score_level(self, tokens, task_emb, attn_module, score_head):
        """Score a set of tokens against a task embedding.

        Args:
            tokens: [1, num_tokens, dim] (broadcast to batch)
            task_emb: [batch, 1, dim]
            attn_module: _ClusterAttention module
            score_head: nn.Sequential scoring head

        Returns:
            scores: [batch, num_tokens]
        """
        b = task_emb.shape[0]
        tokens_expanded = tokens.expand(b, -1, -1)  # [batch, num_tokens, dim]
        task_proj = self.task_proj(task_emb)  # [batch, 1, dim]
        # concat each token with projected task embedding
        combined = torch.stack([tokens_expanded, task_proj.expand(-1, tokens_expanded.shape[1], -1)], dim=2)
        batch_size, num_tokens = b, tokens_expanded.shape[1]
        combined = combined.view(batch_size * num_tokens, 2, self.dim)

        attended = attn_module(combined, combined, combined)  # [batch*num_tokens, 2, dim]
        cls_tokens = attended[:, 0]  # [batch*num_tokens, dim]
        scores = score_head(cls_tokens).squeeze(-1)  # [batch*num_tokens]
        scores = scores.view(batch_size, num_tokens)  # [batch, num_tokens]
        return scores

    def forward_training(self, task_emb, base_model, x_uni, x_hete, attn_mask, attn_mask_func):
        """Training forward: score all levels, no pruning.

        Returns:
            final_scores: [batch, num_learnware] — same shape as base_model.forward()
        """
        # Clear stale aux outputs
        self._aux_L1 = None
        self._aux_L2 = None

        # Score L1 clusters
        self._aux_L1 = self._score_level(self.cluster_tokens_L1, task_emb, self.attn_L1, self.score_L1)

        # Score L2 families
        self._aux_L2 = self._score_level(self.cluster_tokens_L2, task_emb, self.attn_L2, self.score_L2)

        # Score all leaf PTMs via base model (no pruning)
        final_scores = base_model.forward(x_uni, x_hete, attn_mask, attn_mask_func)
        return final_scores

    def forward_inference(self, task_emb, base_model, x_uni, x_hete, attn_mask, attn_mask_func,
                          top_k_L1, top_k_families):
        """Inference forward: prune at each level, only score candidate PTMs.

        Returns:
            final_scores: [batch, num_learnware] — real scores for selected PTMs, -inf for pruned
        """
        batch_size = task_emb.shape[0]
        device = task_emb.device
        num_learnware = self.tree.num_leaves

        # Step 1: Score L1 clusters and select top-K
        l1_scores = self._score_level(self.cluster_tokens_L1, task_emb, self.attn_L1, self.score_L1)
        top_k_L1 = min(top_k_L1, self.tree.num_l1)
        top_l1_indices = torch.topk(l1_scores, top_k_L1, dim=-1).indices  # [batch, top_k_L1]

        # Step 2: Score L2 families and select top-K per selected L1
        l2_scores = self._score_level(self.cluster_tokens_L2, task_emb, self.attn_L2, self.score_L2)

        # Build candidate leaf set per batch element
        candidate_leaves = [[] for _ in range(batch_size)]

        for b in range(batch_size):
            selected_families = []
            for l1_idx in top_l1_indices[b]:
                l1_idx_val = l1_idx.item()
                family_indices = self.tree.get_families(l1_idx_val)
                if len(family_indices) == 0:
                    continue
                family_scores = l2_scores[b, family_indices]
                top_k = min(top_k_families, len(family_indices))
                top_family_local = torch.topk(family_scores, top_k).indices
                for fl in top_family_local:
                    selected_families.append(family_indices[fl.item()])

            for l2_idx in selected_families:
                candidate_leaves[b].extend(self.tree.get_leaves(l2_idx))

        # Step 3: Find the union of candidates across the batch (for the shared base model loop)
        all_candidates = set()
        for leaves in candidate_leaves:
            all_candidates.update(leaves)
        sorted_candidates = sorted(all_candidates)

        if len(sorted_candidates) == 0:
            return torch.full((batch_size, num_learnware), float('-inf'), device=device)

        # Score ONLY the candidate PTMs (skip the rest — this is the speedup)
        candidate_scores = base_model.forward(
            x_uni, x_hete, attn_mask, attn_mask_func,
            candidate_indices=sorted_candidates
        )  # [batch, len(sorted_candidates)]

        # Map candidate scores back to full [batch, num_learnware] tensor
        final_scores = torch.full((batch_size, num_learnware), float('-inf'), device=device)
        candidate_tensor = torch.tensor(sorted_candidates, device=device)
        final_scores[:, candidate_tensor] = candidate_scores

        return final_scores


class _ClusterAttention(nn.Module):
    """Simple self-attention for cluster scoring."""

    def __init__(self, dim, heads=1, dropout=0.1):
        super().__init__()
        self.n_head = heads
        self.d_k = dim
        self.d_v = dim

        self.w_qs = nn.Linear(dim, heads * dim, bias=False)
        self.w_ks = nn.Linear(dim, heads * dim, bias=False)
        self.w_vs = nn.Linear(dim, heads * dim, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (dim + dim)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (dim + dim)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (dim + dim)))

        self.temperature = np.power(dim, 0.5)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim)
        self.fc = nn.Linear(heads * dim, dim)
        nn.init.xavier_normal_(self.fc.weight)

    def forward(self, q, k, v, attn_mask=None):
        sz_b, len_q, _ = q.size()
        residual = q

        q = self.w_qs(q).view(sz_b, len_q, self.n_head, self.d_k).permute(2, 0, 1, 3).contiguous().view(-1, len_q, self.d_k)
        k = self.w_ks(k).view(sz_b, len_q, self.n_head, self.d_k).permute(2, 0, 1, 3).contiguous().view(-1, len_q, self.d_k)
        v = self.w_vs(v).view(sz_b, len_q, self.n_head, self.d_v).permute(2, 0, 1, 3).contiguous().view(-1, len_q, self.d_v)

        attn = torch.bmm(q, k.transpose(1, 2)) / self.temperature
        if attn_mask is not None:
            attn.masked_fill_(attn_mask, -1e9)
        attn = F.softmax(attn, dim=2)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        output = output.view(self.n_head, sz_b, len_q, self.d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)
        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)
        return output
