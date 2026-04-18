"""Quick benchmark: how long to extract prototypes for 100 PTMs."""
import torch, time, torchvision.models as models, timm, torch.cuda as cuda

device = torch.device('cuda')
dummy = torch.randn(64, 3, 224, 224, device=device)

test_models = {
    'resnet50': lambda: models.resnet50(weights='DEFAULT'),
    'vgg16': lambda: models.vgg16(weights='DEFAULT'),
    'mobilenet_v2': lambda: models.mobilenet_v2(weights='DEFAULT'),
    'efficientnet_b0': lambda: models.efficientnet_b0(weights='DEFAULT'),
    'convnext_tiny': lambda: models.convnext_tiny(weights='DEFAULT'),
    'vit_b_16': lambda: models.vit_b_16(weights='DEFAULT'),
    'swin_b': lambda: models.swin_b(weights='DEFAULT'),
    'deit_tiny': lambda: timm.create_model('deit_tiny_patch16_224', pretrained=True),
    'maxvit_tiny': lambda: timm.create_model('maxvit_tiny_224', pretrained=True),
}

print(f"GPU: {cuda.get_device_name(0)}")
print(f"VRAM: {cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB\n")

times = []
for name, fn in test_models.items():
    # Load
    t0 = time.time()
    m = fn().to(device).eval()
    load_t = time.time() - t0

    # Warmup + time 10 batches
    with torch.no_grad():
        for _ in range(3): m(dummy)
        cuda.synchronize()
        t0 = time.time()
        for _ in range(10): m(dummy)
        cuda.synchronize()
        batch_t = (time.time() - t0) / 10 * 1000  # ms

    mem = cuda.max_memory_allocated() / 1024**2
    cuda.reset_peak_memory_stats()
    del m; cuda.empty_cache()
    times.append((name, load_t, batch_t, mem))
    print(f"  {name:<20} load={load_t:.1f}s  batch(64)={batch_t:.0f}ms  peak={mem:.0f}MB")

# Estimate total time
avg_load = sum(t[1] for t in times) / len(times)
avg_batch = sum(t[2] for t in times) / len(times)

print(f"\n--- Estimates for 100 PTMs ---")
print(f"Avg load time:  {avg_load:.1f}s")
print(f"Avg batch time: {avg_batch:.0f}ms (bs=64)")

# Per dataset: 100 models × (load + N_batches × batch_time)
for n_samples in [64, 1000, 5000, 10000, 40000]:
    n_batches = max(1, n_samples // 64)
    per_model = avg_load + n_batches * avg_batch / 1000
    total = 100 * per_model
    print(f"  {n_samples:>6} samples/ds: {per_model:.1f}s/model × 100 = {total/60:.0f} min")

# Full estimate: 13 source (64 samples) + 9 test (varied)
src_time = 13 * 100 * (avg_load + 1 * avg_batch / 1000) / 60
test_datasets = {'CIFAR10': 1000, 'CIFAR100': 10000, 'Caltech101': 5000, 'DTD': 3000, 'Pet': 5000, 'Aircraft': 7000, 'Cars': 10000, 'SUN397': 40000, 'dSprites': 40000}
test_time = sum(100 * (avg_load + (s/64) * avg_batch/1000) for s in test_datasets.values()) / 60

print(f"\n--- Full 100 PTM Generation Estimate ---")
print(f"13 source datasets (~64 samples each): {src_time:.0f} min")
print(f"9 test datasets (varied sizes):         {test_time:.0f} min")
print(f"Total:                                  {src_time + test_time:.0f} min ({(src_time + test_time)/60:.1f} hours)")
