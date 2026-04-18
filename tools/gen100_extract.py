"""
Feature extraction functions for 100 PTM Prototype Pipeline.

Handles model loading, hook registration, inference, and prototype computation
for all 100 models across torchvision and timm sources.
"""

import os
import sys
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from pathlib import Path
from collections import defaultdict

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from learnware.learnware_info_100 import (
    BKB_100_SPECIFIC_RANK,
    MODEL_100_2FEAT_DIM,
    TORCHVISION_MODELS_100,
    TIMM_MODELS_100,
)

# =============================================================================
# STANDARD TRANSFORMS
# =============================================================================

IMAGENET_NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)


def get_transform(model_name: str, input_size: int = 224):
    """Get appropriate transform for the model."""
    if "inception" in model_name and "v3" in model_name:
        return transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            IMAGENET_NORMALIZE,
        ])
    if "384" in model_name:
        return transforms.Compose([
            transforms.Resize(384),
            transforms.CenterCrop(384),
            transforms.ToTensor(),
            IMAGENET_NORMALIZE,
        ])
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        IMAGENET_NORMALIZE,
    ])


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model(model_name: str, device: torch.device):
    """Load a pre-trained model by name. Returns (model, is_timm)."""
    if model_name in TORCHVISION_MODELS_100:
        import torchvision.models as models
        try:
            model_fn = getattr(models, model_name)
            model = model_fn(weights="DEFAULT")
        except Exception:
            model = model_fn(pretrained=True)
        is_timm = False
    else:
        import timm
        model = timm.create_model(model_name, pretrained=True)
        is_timm = True

    model = model.to(device).eval()
    return model, is_timm


# =============================================================================
# FEATURE HOOKS
# =============================================================================

def register_feature_hook(model, model_name: str):
    """
    Register a forward hook to extract features from the penultimate layer.
    Returns (hook_handle, features_list).
    """
    features = []

    if any(k in model_name for k in ["resnet", "regnet", "shufflenet"]):
        target = model.avgpool
        def hook_fn(module, input, output):
            features.append(output.flatten(1).detach())

    elif "densenet" in model_name:
        target = model.features
        def hook_fn(module, input, output):
            features.append(F.adaptive_avg_pool2d(output, (1, 1)).flatten(1).detach())

    elif any(k in model_name for k in ["vgg", "alexnet"]):
        target = model.classifier[-1]
        def hook_fn(module, input, output):
            features.append(input[0].detach())

    elif any(k in model_name for k in ["efficientnet", "mobilenet", "mnasnet"]):
        target = model.classifier[-1] if hasattr(model, "classifier") else model.head
        def hook_fn(module, input, output):
            features.append(input[0].detach())

    elif "convnext" in model_name:
        target = model.classifier[-1] if hasattr(model, "classifier") else model.head
        def hook_fn(module, input, output):
            features.append(input[0].detach())

    elif any(k in model_name for k in ["vit_", "deit_", "swin_"]):
        target = model.head if hasattr(model, "head") else model.fc
        def hook_fn(module, input, output):
            features.append(input[0].detach())

    else:
        # Generic fallback: try head, then fc, then classifier
        if hasattr(model, "head"):
            target = model.head
        elif hasattr(model, "fc"):
            target = model.fc
        elif hasattr(model, "classifier"):
            cls = model.classifier
            target = cls[-1] if isinstance(cls, nn.Sequential) else cls
        else:
            raise ValueError(f"Cannot determine feature layer for {model_name}")

        def hook_fn(module, input, output):
            features.append(input[0].detach())

    hook = target.register_forward_hook(hook_fn)
    return hook, features


# =============================================================================
# PROTOTYPE COMPUTATION
# =============================================================================

def extract_prototypes_from_loader(
    model,
    model_name: str,
    dataloader,
    num_classes: int,
    device: torch.device,
    max_samples_per_class: int = 100,
):
    """
    Extract prototype features (class-wise means) from a model.

    Returns:
        prototypes: Tensor of shape [num_classes, feature_dim]
    """
    hook, features_list = register_feature_hook(model, model_name)

    class_features = defaultdict(list)
    feature_dim = MODEL_100_2FEAT_DIM[model_name]

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            _ = model(images)

            if features_list:
                batch_features = features_list[-1].cpu()
                features_list.clear()

                for feat, label in zip(batch_features, labels):
                    label_idx = label.item()
                    if len(class_features[label_idx]) < max_samples_per_class:
                        class_features[label_idx].append(feat)

    hook.remove()

    prototypes = torch.zeros(num_classes, feature_dim)
    for class_idx in range(num_classes):
        if class_idx in class_features and class_features[class_idx]:
            prototypes[class_idx] = torch.stack(class_features[class_idx]).mean(dim=0)

    return prototypes


# =============================================================================
# DATASET LOADING
# =============================================================================

def get_dataloader(dataset_name: str, data_path: str, model_name: str,
                    batch_size: int = 32, num_workers: int = 2,
                    max_samples_per_class: int = 100, input_size: int = 224):
    """
    Create a DataLoader for a dataset. Returns (dataloader, num_classes).

    For torchvision datasets that support auto-download, this handles it.
    For others, data_path must point to the downloaded dataset.
    """
    import torchvision.datasets as datasets
    from torch.utils.data import DataLoader, Subset

    # CIFAR images are 32x32 — need Resize for models expecting 224+
    if dataset_name in ("CIFAR10", "CIFAR100"):
        transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            IMAGENET_NORMALIZE,
        ])
    else:
        transform = get_transform(model_name, input_size)
    num_classes = None

    if dataset_name == "CIFAR10":
        dataset = datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)
        num_classes = 10
    elif dataset_name == "CIFAR100":
        dataset = datasets.CIFAR100(root=data_path, train=True, download=True, transform=transform)
        num_classes = 100
    elif dataset_name == "Caltech101":
        try:
            dataset = datasets.Caltech101(root=data_path, download=True, transform=transform)
        except Exception:
            print(f"    [WARN] Caltech101 auto-download failed. Place manually in {data_path}/caltech101/")
            raise
        num_classes = 101
    elif dataset_name == "DTD":
        dataset = datasets.DTD(root=data_path, split="train", download=True, transform=transform)
        num_classes = 47
    elif dataset_name == "Pet":
        dataset = datasets.OxfordIIITPet(root=data_path, split="trainval", download=True, transform=transform)
        num_classes = 37
    elif dataset_name == "SUN397":
        try:
            dataset = datasets.SUN397(root=data_path, download=True, transform=transform)
        except Exception:
            print(f"    [WARN] SUN397 auto-download failed. Place manually in {data_path}/sun397/")
            raise
        num_classes = 397
    elif dataset_name == "Aircraft":
        dataset = datasets.FGVCAircraft(root=data_path, annotation_level="variant",
                                         split="train", download=True, transform=transform)
        num_classes = 100
    elif dataset_name == "Cars":
        dataset = datasets.StanfordCars(root=data_path, split="train",
                                         download=True, transform=transform)
        num_classes = 196
    elif dataset_name == "dSprites":
        dataset, num_classes = _load_dsprites(data_path, transform, max_samples_per_class)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Subsample to max_samples_per_class for faster extraction
    if max_samples_per_class and len(dataset) > num_classes * max_samples_per_class * 2:
        dataset = _subsample_per_class(dataset, num_classes, max_samples_per_class)

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return dataloader, num_classes


def _subsample_per_class(dataset, num_classes, max_per_class):
    """Subsample dataset to at most max_per_class samples per class."""
    from torch.utils.data import Subset
    class_indices = defaultdict(list)

    for i in range(len(dataset)):
        try:
            _, label = dataset[i]
            if isinstance(label, torch.Tensor):
                label = label.item()
            class_indices[label].append(i)
        except Exception:
            break

    selected = []
    for label, indices in class_indices.items():
        if len(indices) > max_per_class:
            selected.extend(random.sample(indices, max_per_class))
        else:
            selected.extend(indices)

    return Subset(dataset, selected)


def _load_dsprites(data_path, transform, max_samples):
    """Load dSprites dataset (custom handling)."""
    # dSprites is a numpy file — provide a minimal dataset
    # For now, return None with a warning
    print(f"  [WARN] dSprites requires manual download. Place in {data_path}/dsprites")
    # Return empty dataset
    from torch.utils.data import TensorDataset
    dummy = TensorDataset(torch.zeros(1, 3, 64, 64), torch.zeros(1, dtype=torch.long))
    return dummy, 1


# =============================================================================
# SINGLE MODEL EXTRACTION
# =============================================================================

def extract_one_model(
    model_name: str,
    datasets_to_process: list,
    feature_dir: Path,
    device: torch.device,
    batch_size: int = 32,
    data_path: str = "./raw_data",
):
    """
    Extract features from one model across all specified datasets.

    Args:
        model_name: Name of the model (e.g. 'resnet50')
        datasets_to_process: List of (ds_name, input_size, num_classes) tuples
        feature_dir: Base directory for saving .pt feature files
        device: torch device
        batch_size: Batch size for inference
        data_path: Path to raw datasets

    Returns:
        dict of {ds_name: "ok" | "skip" | "error"}
    """
    results = {}

    # Check if ALL datasets already done for this model
    all_skip = True
    for ds_name, _, _ in datasets_to_process:
        out_file = feature_dir / ds_name / f"{model_name}.pt"
        if not out_file.exists():
            all_skip = False
            break
    if all_skip:
        for ds_name, _, _ in datasets_to_process:
            results[ds_name] = "skip"
        return results

    # Load model
    try:
        model, is_timm = load_model(model_name, device)
    except Exception as e:
        print(f"    FAILED to load {model_name}: {e}")
        for ds_name, _, _ in datasets_to_process:
            results[ds_name] = f"load_error: {e}"
        return results

    model_mem = torch.cuda.max_memory_allocated() / 1024**2
    torch.cuda.reset_peak_memory_stats()

    for ds_name, input_size, num_classes in datasets_to_process:
        out_file = feature_dir / ds_name / f"{model_name}.pt"
        if out_file.exists():
            results[ds_name] = "skip"
            continue

        try:
            dataloader, ds_nc = get_dataloader(
                ds_name, data_path, model_name,
                batch_size=batch_size, num_workers=0,
                input_size=input_size,
            )
            # Use provided num_classes if dataset doesn't report it
            nc = num_classes if ds_nc is None else ds_nc

            prototypes = extract_prototypes_from_loader(
                model, model_name, dataloader, nc, device,
            )

            out_file.parent.mkdir(parents=True, exist_ok=True)
            torch.save(prototypes, out_file)
            results[ds_name] = "ok"

        except Exception as e:
            print(f"    ERROR on {ds_name}: {e}")
            # Don't save placeholder — leave file missing so retry picks it up
            results[ds_name] = f"error: {e}"

    del model
    torch.cuda.empty_cache()
    return results
