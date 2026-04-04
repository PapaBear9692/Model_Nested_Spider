"""
Prototype Data Generator for 100 PTMs

This script generates prototype features for all 100 PTMs on a given dataset.
Prototype features are class-wise mean features extracted from the penultimate layer.

Usage:
    python tools/generate_prototype_100.py --dataset CIFAR10 --data_path /path/to/data --output_path ./prototype_data

Output:
    Creates .pkl files with structure:
    [uniform_features, heterogeneous_features_dict]
    where heterogeneous_features_dict contains features from all 100 PTMs
"""

import os
import sys
import argparse
import pickle
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from learnware.learnware_info_100 import BKB_100_SPECIFIC_RANK, MODEL_100_2FEAT_DIM


# =============================================================================
# Model Loader
# =============================================================================

TORCHVISION_MODELS = {
    'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
    'densenet121', 'densenet161', 'densenet169', 'densenet201',
    'vgg11', 'vgg13', 'vgg16', 'vgg19', 'vgg11_bn', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn',
    'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3',
    'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7',
    'regnet_y_400mf', 'regnet_y_800mf', 'regnet_y_1_6gf', 'regnet_y_3_2gf',
    'regnet_y_8gf', 'regnet_y_16gf', 'regnet_y_32gf',
    'regnet_x_400mf', 'regnet_x_800mf', 'regnet_x_1_6gf', 'regnet_x_3_2gf',
    'regnet_x_8gf', 'regnet_x_16gf', 'regnet_x_32gf',
    'mobilenet_v2', 'mobilenet_v3_small', 'mobilenet_v3_large',
    'mnasnet0_5', 'mnasnet1_0',
    'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0',
    'squeezenet1_0', 'squeezenet1_1',
    'googlenet', 'inception_v3', 'alexnet',
    'convnext_tiny', 'convnext_small', 'convnext_base', 'convnext_large',
    'vit_b_16', 'vit_b_32', 'vit_l_16', 'vit_l_32',
    'swin_t', 'swin_s', 'swin_b', 'swin_v2_t',
}


def get_model(model_name: str, device: torch.device):
    """Load a pre-trained model by name."""

    if model_name in TORCHVISION_MODELS:
        import torchvision.models as models

        # Load with default pretrained weights
        try:
            model_fn = getattr(models, model_name)
            model = model_fn(weights="DEFAULT")
        except Exception as e:
            print(f"Warning: Could not load {model_name} with DEFAULT weights: {e}")
            model = model_fn(pretrained=True)

    else:
        # Use timm for other models
        import timm
        model = timm.create_model(model_name, pretrained=True)

    model = model.to(device)
    model.eval()
    return model


def get_feature_hook(model, model_name: str):
    """
    Register a forward hook to extract features from the penultimate layer.
    Returns the hook handle and a list to store features.
    """
    features = []

    # Determine the feature extraction layer based on model architecture
    if 'resnet' in model_name or 'regnet' in model_name or 'shufflenet' in model_name:
        # ResNet family: use avgpool output (before fc)
        target_layer = model.avgpool
        def hook_fn(module, input, output):
            features.append(output.flatten(1).detach())
    elif 'densenet' in model_name:
        target_layer = model.features
        def hook_fn(module, input, output):
            features.append(F.adaptive_avg_pool2d(output, (1, 1)).flatten(1).detach())
    elif 'vgg' in model_name or 'alexnet' in model_name:
        target_layer = model.classifier[-1]  # Last FC layer
        def hook_fn(module, input, output):
            features.append(input[0].detach())
    elif 'efficientnet' in model_name or 'mobilenet' in model_name or 'mnasnet' in model_name:
        target_layer = model.classifier[-1]
        def hook_fn(module, input, output):
            features.append(input[0].detach())
    elif 'convnext' in model_name:
        target_layer = model.classifier[-1]
        def hook_fn(module, input, output):
            features.append(input[0].detach())
    elif 'vit' in model_name or 'deit' in model_name:
        # Vision Transformers: use [CLS] token
        target_layer = model.head
        def hook_fn(module, input, output):
            features.append(input[0].detach())
    elif 'swin' in model_name:
        target_layer = model.head
        def hook_fn(module, input, output):
            features.append(input[0].detach())
    else:
        # Generic: try to get the layer before the final classifier
        if hasattr(model, 'head'):
            target_layer = model.head
        elif hasattr(model, 'fc'):
            target_layer = model.fc
        elif hasattr(model, 'classifier'):
            target_layer = model.classifier[-1] if isinstance(model.classifier, nn.Sequential) else model.classifier
        else:
            raise ValueError(f"Cannot determine feature layer for {model_name}")

        def hook_fn(module, input, output):
            features.append(input[0].detach())

    hook = target_layer.register_forward_hook(hook_fn)
    return hook, features


def get_transform(model_name: str, input_size: int = 224):
    """Get appropriate transform for the model."""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    # Special handling for inception_v3
    if 'inception' in model_name:
        return transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            normalize,
        ])

    # Special handling for ViT with larger input
    if '384' in model_name:
        return transforms.Compose([
            transforms.Resize(384),
            transforms.CenterCrop(384),
            transforms.ToTensor(),
            normalize,
        ])

    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        normalize,
    ])


# =============================================================================
# Prototype Extraction
# =============================================================================

def extract_prototypes(
    model_name: str,
    dataloader: DataLoader,
    num_classes: int,
    device: torch.device,
    max_samples_per_class: int = 100,
):
    """
    Extract prototype features (class-wise means) from a model.

    Returns:
        prototype_features: Tensor of shape [num_classes, feature_dim]
    """
    model = get_model(model_name, device)
    hook, features_list = get_feature_hook(model, model_name)

    # Collect features per class
    class_features = defaultdict(list)

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc=f"Extracting {model_name}", leave=False):
            images = images.to(device)
            _ = model(images)

            # Get features from hook
            batch_features = features_list[-1].cpu()
            features_list.clear()

            # Organize by class
            for feat, label in zip(batch_features, labels):
                label_idx = label.item()
                if len(class_features[label_idx]) < max_samples_per_class:
                    class_features[label_idx].append(feat)

    hook.remove()
    del model
    torch.cuda.empty_cache()

    # Compute mean feature per class (prototype)
    feature_dim = MODEL_100_2FEAT_DIM[model_name]
    prototypes = torch.zeros(num_classes, feature_dim)

    for class_idx in range(num_classes):
        if class_idx in class_features and len(class_features[class_idx]) > 0:
            class_feats = torch.stack(class_features[class_idx])
            prototypes[class_idx] = class_feats.mean(dim=0)

    return prototypes


def generate_prototype_data(
    dataset_name: str,
    data_path: str,
    output_path: str,
    num_classes: int,
    models_to_use: list = None,
    max_samples_per_class: int = 100,
    batch_size: int = 32,
    num_workers: int = 4,
    uniform_model: str = "swin_base_patch16_window7_224",
):
    """
    Generate prototype data for a dataset using all 100 PTMs.

    Args:
        dataset_name: Name of the dataset (e.g., "CIFAR10")
        data_path: Path to the dataset
        output_path: Path to save the prototype .pkl files
        num_classes: Number of classes in the dataset
        models_to_use: List of model names to use (default: all 100)
        max_samples_per_class: Maximum samples per class for prototype computation
        batch_size: Batch size for feature extraction
        num_workers: Number of data loader workers
        uniform_model: Model for uniform features (default: Swin Transformer)
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if models_to_use is None:
        models_to_use = BKB_100_SPECIFIC_RANK

    # Create output directory
    output_dir = Path(output_path) / f"{dataset_name}_100ptm"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset (simplified - you may need to adapt this)
    # This is a placeholder - implement based on your dataset structure
    print(f"Loading dataset: {dataset_name}")

    # TODO: Implement dataset loading based on your data structure
    # For now, using torchvision datasets as example
    if dataset_name == "CIFAR10":
        import torchvision.datasets as datasets
        transform = get_transform("resnet50")  # Standard transform
        dataset = datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)
        num_classes = 10
    elif dataset_name == "CIFAR100":
        import torchvision.datasets as datasets
        transform = get_transform("resnet50")
        dataset = datasets.CIFAR100(root=data_path, train=True, download=True, transform=transform)
        num_classes = 100
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented. Please add custom loading.")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"Dataset loaded: {len(dataset)} samples, {num_classes} classes")

    # Extract uniform features (using Swin Transformer)
    print("\n" + "="*70)
    print("Extracting UNIFORM features (Swin Transformer)...")
    print("="*70)

    uniform_prototypes = extract_prototypes(
        "swin_b",  # Use swin_b from torchvision
        dataloader,
        num_classes,
        device,
        max_samples_per_class
    )

    print(f"Uniform prototypes shape: {uniform_prototypes.shape}")

    # Reload dataset for each model (to apply correct transforms)
    # For efficiency, we'll use the same transform for all models

    # Extract heterogeneous features from all 100 PTMs
    print("\n" + "="*70)
    print("Extracting HETEROGENEOUS features (100 PTMs)...")
    print("="*70)

    heterogeneous_prototypes = {}

    for i, model_name in enumerate(models_to_use):
        print(f"\n[{i+1}/{len(models_to_use)}] Processing {model_name}...")

        try:
            # Recreate dataloader with appropriate transform
            transform = get_transform(model_name)
            dataset.transform = transform
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            )

            prototypes = extract_prototypes(
                model_name,
                dataloader,
                num_classes,
                device,
                max_samples_per_class
            )

            heterogeneous_prototypes[model_name] = prototypes
            print(f"  ✓ {model_name}: shape {prototypes.shape}")

        except Exception as e:
            print(f"  ✗ Failed: {e}")
            # Add zeros as placeholder
            feature_dim = MODEL_100_2FEAT_DIM.get(model_name, 2048)
            heterogeneous_prototypes[model_name] = torch.zeros(num_classes, feature_dim)

    # Save prototype data
    print("\n" + "="*70)
    print("Saving prototype data...")
    print("="*70)

    prototype_data = [
        uniform_prototypes,  # x[0]: uniform features
        heterogeneous_prototypes  # x[1]: heterogeneous features dict
    ]

    save_file = output_dir / f"prototypes_100ptm.pkl"
    with open(save_file, 'wb') as f:
        pickle.dump(prototype_data, f)

    print(f"Saved to: {save_file}")
    print(f"File size: {os.path.getsize(save_file) / 1024 / 1024:.2f} MB")

    return prototype_data


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate prototype data for 100 PTMs")

    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., CIFAR10)")
    parser.add_argument("--data_path", type=str, required=True, help="Path to dataset")
    parser.add_argument("--output_path", type=str, default="./prototype_data_100", help="Output directory")
    parser.add_argument("--num_classes", type=int, default=10, help="Number of classes")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--max_samples_per_class", type=int, default=100, help="Max samples per class")

    args = parser.parse_args()

    generate_prototype_data(
        dataset_name=args.dataset,
        data_path=args.data_path,
        output_path=args.output_path,
        num_classes=args.num_classes,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples_per_class=args.max_samples_per_class,
    )
