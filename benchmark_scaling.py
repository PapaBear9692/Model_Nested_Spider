"""
Synthetic Scaling Benchmark: Normal vs Hierarchical Model Spider.

Creates dummy models at N=10,20,50,100,200 PTMs and measures inference time
to demonstrate the O(N) vs O(sqrt(N)) scaling of hierarchical clustering.
"""

import torch
import time
import math
from learnware.model import LearnwareCAHeterogeneous
from learnware.hierarchical_cluster import ClusterTree
from learnware.model import HierarchicalLearnwareCA


def generate_balanced_tree(N, num_l1=5):
    """Generate a balanced cluster tree for N PTMs."""
    l1_names = [f'L1_{i}' for i in range(num_l1)]

    # Distribute N PTMs across L1 clusters
    base_per_l1 = N // num_l1
    remainder = N % num_l1

    tree = {}
    ptm_idx = 0
    for i, l1 in enumerate(l1_names):
        n_in_l1 = base_per_l1 + (1 if i < remainder else 0)
        # Create families of ~3 PTMs each
        num_families = max(1, math.ceil(n_in_l1 / 3))
        families = {}
        fam_size = n_in_l1 // num_families
        fam_remainder = n_in_l1 % num_families
        ptm_count = 0
        for f in range(num_families):
            f_size = fam_size + (1 if f < fam_remainder else 0)
            members = [f'ptm_{ptm_idx + j}' for j in range(f_size)]
            families[f'F{i}_{f}'] = members
            ptm_count += f_size
        ptm_idx += ptm_count
        tree[l1] = families

    # Collect all model names in order
    all_models = []
    for l1 in tree:
        for fam in tree[l1]:
            all_models.extend(tree[l1][fam])

    return tree, all_models


def create_normal_model(N, dim=1024):
    """Create a normal Model Spider with N PTMs (synthetic feature dims)."""
    model = LearnwareCAHeterogeneous(
        num_learnware=N, dim=dim, hdim=dim,
        uni_hete_proto_dim=(10, 10),
        data_sub_url='swin_base_7_checkpoint',
        pool='cls', heads=1, dropout=0.1, emb_dropout=0.1,
    )
    return model


def create_hierarchical_model(N, dim=1024, top_k_L1=2, top_k_families=2):
    """Create a hierarchical Model Spider with N PTMs."""
    tree_config, all_models = generate_balanced_tree(N)
    model = HierarchicalLearnwareCA(
        num_learnware=N, dim=dim, hdim=dim,
        uni_hete_proto_dim=(10, 10),
        data_sub_url='swin_base_7_checkpoint',
        cluster_tree_config=tree_config,
        all_models=all_models,
        top_k_L1=top_k_L1,
        top_k_families=top_k_families,
        pool='cls', heads=1, dropout=0.1, emb_dropout=0.1,
    )
    return model


def run_benchmark():
    device = torch.device('cuda')
    dim = 1024
    batch_size = 4
    num_iterations = 50
    proto_len = 10

    test_sizes = [10, 20, 50, 100, 200]

    print("=" * 90)
    print("Synthetic Scaling Benchmark: Normal vs Hierarchical Model Spider")
    print(f"Device: {torch.cuda.get_device_name()}, Dim: {dim}, Batch: {batch_size}, Iterations: {num_iterations}")
    print("=" * 90)
    print()

    results = []

    for N in test_sizes:
        print(f"--- N={N} PTMs ---")

        # Normal model
        normal_model = create_normal_model(N, dim).to(device).eval()

        # Hierarchical model (top_k_L1=2, top_k_families=2)
        hier_model = create_hierarchical_model(N, dim, top_k_L1=2, top_k_families=2).to(device).eval()

        # Create dummy inputs
        x_uni = torch.randn(batch_size, proto_len, dim, device=device)
        # For heterogeneous mode, x_hete needs to be a dict of padded tensors per PTM
        x_hete = {i: torch.randn(batch_size, proto_len, dim, device=device) for i in range(N)}
        attn_mask = None

        # Warm up
        with torch.no_grad():
            for _ in range(3):
                normal_model(x_uni, x_hete, attn_mask)
                hier_model(x_uni, x_hete, attn_mask)

        torch.cuda.synchronize()

        # Time normal model
        start = time.time()
        with torch.no_grad():
            for _ in range(num_iterations):
                normal_model(x_uni, x_hete, attn_mask)
        torch.cuda.synchronize()
        normal_time = (time.time() - start) / num_iterations * 1000  # ms

        # Time hierarchical model
        start = time.time()
        with torch.no_grad():
            for _ in range(num_iterations):
                hier_model(x_uni, x_hete, attn_mask)
        torch.cuda.synchronize()
        hier_time = (time.time() - start) / num_iterations * 1000  # ms

        # Get pass counts
        normal_passes = N
        passes = hier_model.cluster_layer._inference_passes
        hier_total = passes['l1'] + passes['l2'] + passes['leaf']
        speedup = normal_time / hier_time if hier_time > 0 else float('inf')

        tree_config, _ = generate_balanced_tree(N)
        num_l1 = len(tree_config)
        num_l2 = sum(len(fams) for fams in tree_config.values())

        results.append({
            'N': N, 'normal_ms': normal_time, 'hier_ms': hier_time,
            'speedup': speedup, 'normal_passes': normal_passes,
            'hier_passes': hier_total, 'l1': passes['l1'], 'l2': passes['l2'],
            'leaf': passes['leaf'], 'num_l1': num_l1, 'num_l2': num_l2,
            'pruned': N - passes['leaf'],
        })

        print(f"  Tree: {num_l1} L1 clusters, {num_l2} L2 families")
        print(f"  Normal:   {normal_time:6.2f} ms  ({normal_passes} passes)")
        print(f"  Hier:     {hier_time:6.2f} ms  ({passes['l1']}+{passes['l2']}+{passes['leaf']}={hier_total} passes, pruned {N - passes['leaf']}/{N})")
        print(f"  Speedup:  {speedup:.2f}x")
        print()

    # Summary table
    print("=" * 90)
    print("Summary Table")
    print("=" * 90)
    print(f"{'N':>5} | {'Normal (ms)':>12} | {'Hier (ms)':>10} | {'Speedup':>8} | {'Passes N/H':>12} | {'Pruned':>8}")
    print("-" * 90)
    for r in results:
        print(f"{r['N']:>5} | {r['normal_ms']:>10.2f}   | {r['hier_ms']:>8.2f}   | {r['speedup']:>6.2f}x  | {r['normal_passes']:>3} / {r['hier_passes']:<3}      | {r['pruned']:>3}/{r['N']}")
    print()

    # Memory
    normal_mem = torch.cuda.max_memory_allocated() / (1024**2)
    print(f"Peak GPU memory: {normal_mem:.0f} MB")


if __name__ == '__main__':
    run_benchmark()
