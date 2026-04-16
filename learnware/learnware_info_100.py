"""
100 PTM Mode Configuration for Model-Spider

This file contains all configuration needed to run Model-Spider with 100 PTMs.
To use this configuration, set environment variable: PTM100=yes

DO NOT modify the original learnware_info.py - this file is standalone.

Usage:
    # In your code or before running:
    import os
    os.environ['PTM100'] = 'yes'

    # Then import this module:
    from learnware.learnware_info_100 import (
        BKB_SPECIFIC_RANK,
        BKB_SPECIFIC_RANK2ID,
        MODEL2FEAT_DIM,
        NUM_LEARNWARE
    )
"""

import os
import torch

# =============================================================================
# 100 PRE-TRAINED MODELS
# =============================================================================

BKB_100_SPECIFIC_RANK = [
    # === ResNet Family (5) ===
    'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',

    # === DenseNet Family (4) ===
    'densenet121', 'densenet161', 'densenet169', 'densenet201',

    # === VGG Family (8) ===
    'vgg11', 'vgg13', 'vgg16', 'vgg19',
    'vgg11_bn', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn',

    # === EfficientNet Family (8) ===
    'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3',
    'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7',

    # === RegNet-Y Family (7) ===
    'regnet_y_400mf', 'regnet_y_800mf', 'regnet_y_1_6gf', 'regnet_y_3_2gf',
    'regnet_y_8gf', 'regnet_y_16gf', 'regnet_y_32gf',

    # === RegNet-X Family (7) ===
    'regnet_x_400mf', 'regnet_x_800mf', 'regnet_x_1_6gf', 'regnet_x_3_2gf',
    'regnet_x_8gf', 'regnet_x_16gf', 'regnet_x_32gf',

    # === MobileNet Family (5) ===
    'mobilenet_v2', 'mobilenet_v3_small', 'mobilenet_v3_large',
    'mnasnet0_5', 'mnasnet1_0',

    # === ShuffleNet Family (4) ===
    'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0',

    # === SqueezeNet Family (2) ===
    'squeezenet1_0', 'squeezenet1_1',

    # === GoogleNet/Inception/AlexNet (3) ===
    'googlenet', 'inception_v3', 'alexnet',

    # === ConvNeXt Family (4) ===
    'convnext_tiny', 'convnext_small', 'convnext_base', 'convnext_large',

    # === Vision Transformer (4) ===
    'vit_b_16', 'vit_b_32', 'vit_l_16', 'vit_l_32',

    # === Swin Transformer (4) ===
    'swin_t', 'swin_s', 'swin_b', 'swin_v2_t',

    # === EfficientNetV2 (4) - from timm ===
    'efficientnetv2_s', 'efficientnetv2_m', 'efficientnetv2_l', 'efficientnetv2_xl',

    # === DeiT (4) - from timm ===
    'deit_tiny_patch16_224', 'deit_small_patch16_224',
    'deit_base_patch16_224', 'deit_base_patch16_384',

    # === MobileViT (3) - from timm ===
    'mobilevit_s', 'mobilevit_xs', 'mobilevit_xxs',

    # === MaxViT (3) - from timm ===
    'maxvit_tiny_224', 'maxvit_small_224', 'maxvit_base_224',

    # === ConvNeXtV2 (4) - from timm ===
    'convnextv2_atto', 'convnextv2_femto', 'convnextv2_nano', 'convnextv2_tiny',

    # === CoaT (3) - from timm ===
    'coat_lite_tiny', 'coat_lite_small', 'coat_lite_medium',

    # === LeViT (3) - from timm ===
    'levit_128s', 'levit_128', 'levit_192',

    # === HRNet (3) - from timm ===
    'hrnet_w18_small', 'hrnet_w18', 'hrnet_w30',

    # === Twins (3) - from timm ===
    'twins_pcpvt_small', 'twins_pcpvt_base', 'twins_pcpvt_large',

    # === TNT, PiT, CaiT (5) - from timm ===
    'tnt_s_patch16_224', 'pit_b_224', 'pit_s_224', 'pit_xs_224', 'cait_s24_224',
]

# Verify count
assert len(BKB_100_SPECIFIC_RANK) == 100, f"Expected 100 PTMs, got {len(BKB_100_SPECIFIC_RANK)}"

# Create ID mapping
BKB_100_SPECIFIC_RANK2ID = {name: i for i, name in enumerate(BKB_100_SPECIFIC_RANK)}

# Number of learnwares
NUM_LEARNWARE_100 = 100

# =============================================================================
# FEATURE DIMENSIONS
# =============================================================================

MODEL_100_2FEAT_DIM = {
    # ResNet Family
    'resnet18': 512,
    'resnet34': 512,
    'resnet50': 2048,
    'resnet101': 2048,
    'resnet152': 2048,

    # DenseNet Family
    'densenet121': 1024,
    'densenet161': 2208,
    'densenet169': 1664,
    'densenet201': 1920,

    # VGG Family
    'vgg11': 4096,
    'vgg13': 4096,
    'vgg16': 4096,
    'vgg19': 4096,
    'vgg11_bn': 4096,
    'vgg13_bn': 4096,
    'vgg16_bn': 4096,
    'vgg19_bn': 4096,

    # EfficientNet Family
    'efficientnet_b0': 1280,
    'efficientnet_b1': 1280,
    'efficientnet_b2': 1408,
    'efficientnet_b3': 1536,
    'efficientnet_b4': 1792,
    'efficientnet_b5': 2048,
    'efficientnet_b6': 2304,
    'efficientnet_b7': 2560,

    # RegNet-Y Family
    'regnet_y_400mf': 440,
    'regnet_y_800mf': 560,
    'regnet_y_1_6gf': 888,
    'regnet_y_3_2gf': 1512,
    'regnet_y_8gf': 2016,
    'regnet_y_16gf': 2280,
    'regnet_y_32gf': 2640,

    # RegNet-X Family
    'regnet_x_400mf': 400,
    'regnet_x_800mf': 672,
    'regnet_x_1_6gf': 912,
    'regnet_x_3_2gf': 1008,
    'regnet_x_8gf': 1920,
    'regnet_x_16gf': 2048,
    'regnet_x_32gf': 2320,

    # MobileNet Family
    'mobilenet_v2': 1280,
    'mobilenet_v3_small': 576,
    'mobilenet_v3_large': 960,
    'mnasnet0_5': 1280,
    'mnasnet1_0': 1280,

    # ShuffleNet Family
    'shufflenet_v2_x0_5': 1024,
    'shufflenet_v2_x1_0': 1024,
    'shufflenet_v2_x1_5': 1024,
    'shufflenet_v2_x2_0': 2048,

    # SqueezeNet Family
    'squeezenet1_0': 512,
    'squeezenet1_1': 512,

    # GoogleNet/Inception/AlexNet
    'googlenet': 1024,
    'inception_v3': 2048,
    'alexnet': 4096,

    # ConvNeXt Family
    'convnext_tiny': 768,
    'convnext_small': 768,
    'convnext_base': 1024,
    'convnext_large': 1536,

    # Vision Transformer
    'vit_b_16': 768,
    'vit_b_32': 768,
    'vit_l_16': 1024,
    'vit_l_32': 1024,

    # Swin Transformer
    'swin_t': 768,
    'swin_s': 768,
    'swin_b': 1024,
    'swin_v2_t': 768,

    # EfficientNetV2 (timm)
    'efficientnetv2_s': 1280,
    'efficientnetv2_m': 1280,
    'efficientnetv2_l': 1280,
    'efficientnetv2_xl': 1280,

    # DeiT (timm)
    'deit_tiny_patch16_224': 192,
    'deit_small_patch16_224': 384,
    'deit_base_patch16_224': 768,
    'deit_base_patch16_384': 768,

    # MobileViT (timm)
    'mobilevit_s': 640,
    'mobilevit_xs': 384,
    'mobilevit_xxs': 320,

    # MaxViT (timm)
    'maxvit_tiny_224': 512,
    'maxvit_small_224': 768,
    'maxvit_base_224': 1024,

    # ConvNeXtV2 (timm)
    'convnextv2_atto': 320,
    'convnextv2_femto': 384,
    'convnextv2_nano': 512,
    'convnextv2_tiny': 768,

    # CoaT (timm)
    'coat_lite_tiny': 192,
    'coat_lite_small': 320,
    'coat_lite_medium': 512,

    # LeViT (timm)
    'levit_128s': 384,
    'levit_128': 384,
    'levit_192': 384,

    # HRNet (timm)
    'hrnet_w18_small': 2048,
    'hrnet_w18': 2048,
    'hrnet_w30': 2048,

    # Twins (timm)
    'twins_pcpvt_small': 768,
    'twins_pcpvt_base': 768,
    'twins_pcpvt_large': 768,

    # TNT, PiT, CaiT (timm)
    'tnt_s_patch16_224': 640,
    'pit_b_224': 768,
    'pit_s_224': 576,
    'pit_xs_224': 384,
    'cait_s24_224': 768,
}

# =============================================================================
# UNIFIED FEATURE DIMENSION (for projection)
# =============================================================================

UNIFIED_FEAT_DIM_100 = 1024  # Project all features to 1024-dim

# =============================================================================
# DATASET INFORMATION (same as original)
# =============================================================================

DATASET2NUM_CLASSES = {
    'CIFAR10': 10,
    'CIFAR100': 100,
    'Caltech101': 101,
    'DTD': 47,
    'Pet': 37,
    'Aircraft': 100,
    'Cars': 196,
    'SUN397': 397,
    'dSprites': 1,  # For ranking, not classification
    'ImageNet': 1000,
    'CUB2011': 200,
    'Dogs': 120,
    'Flowers': 102,
    'Food': 101,
    'EuroSAT': 10,
    'STL10': 10,
    'SVHN': 10,
    'PACS': 7,
    'VLCS': 5,
    'OfficeHome': 65,
    'SmallNORB': 5,
    'Resisc45': 45,
    'AID': 30,
    'NABirds': 555,
}

# =============================================================================
# MODEL SOURCE TRACKING
# =============================================================================

TORCHVISION_MODELS_100 = {
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

# Remaining models are from timm
TIMM_MODELS_100 = set(BKB_100_SPECIFIC_RANK) - TORCHVISION_MODELS_100

# =============================================================================
# ALIASES FOR COMPATIBILITY
# =============================================================================

# These aliases allow the code to work with either 10 or 100 PTM mode
# When PTM100=yes is set, use these

def get_100_mode_config():
    """
    Get configuration for 100 PTM mode.

    Returns:
        dict with keys: BKB_SPECIFIC_RANK, BKB_SPECIFIC_RANK2ID, MODEL2FEAT_DIM, NUM_LEARNWARE
    """
    return {
        'BKB_SPECIFIC_RANK': BKB_100_SPECIFIC_RANK,
        'BKB_SPECIFIC_RANK2ID': BKB_100_SPECIFIC_RANK2ID,
        'MODEL2FEAT_DIM': MODEL_100_2FEAT_DIM,
        'NUM_LEARNWARE': NUM_LEARNWARE_100,
        'UNIFIED_FEAT_DIM': UNIFIED_FEAT_DIM_100,
    }


# =============================================================================
# CLUSTER TREE FOR 100 PTM HIERARCHICAL MODE
# =============================================================================

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
        'pit_cait_tnt': ['tnt_s_patch16_224', 'pit_b_224', 'pit_s_224', 'pit_xs_224', 'cait_s24_224'],
    },
    'hybrid_specialized': {
        'convnext': ['convnext_tiny', 'convnext_small', 'convnext_base', 'convnext_large'],
        'convnextv2': ['convnextv2_atto', 'convnextv2_femto', 'convnextv2_nano', 'convnextv2_tiny'],
        'hrnet': ['hrnet_w18_small', 'hrnet_w18', 'hrnet_w30'],
    },
}

# Verify cluster tree matches 100 PTM list
_all_clustered = []
for _l1 in CLUSTER_TREE_100.values():
    for _leaves in _l1.values():
        _all_clustered.extend(_leaves)
assert sorted(_all_clustered) == sorted(BKB_100_SPECIFIC_RANK), \
    f"Cluster tree mismatch: tree has {sorted(_all_clustered)}, list has {sorted(BKB_100_SPECIFIC_RANK)}"

# =============================================================================
# CHECK PTM100 ENVIRONMENT VARIABLE
# =============================================================================

def is_100_mode():
    """Check if 100 PTM mode is enabled via environment variable."""
    return os.environ.get('PTM100', '').lower() == 'yes'


# Convenience: Auto-set aliases when module is imported with PTM100=yes
if is_100_mode():
    BKB_SPECIFIC_RANK = BKB_100_SPECIFIC_RANK
    BKB_SPECIFIC_RANK2ID = BKB_100_SPECIFIC_RANK2ID
    MODEL2FEAT_DIM = MODEL_100_2FEAT_DIM
    NUM_LEARNWARE = NUM_LEARNWARE_100
    print(f"[100 PTM Mode] Loaded configuration for {NUM_LEARNWARE} pre-trained models")


if __name__ == "__main__":
    print("="*70)
    print("100 PTM Configuration Summary")
    print("="*70)
    print(f"Total PTMs: {len(BKB_100_SPECIFIC_RANK)}")
    print(f"Torchvision models: {len(TORCHVISION_MODELS_100)}")
    print(f"TIMM models: {len(TIMM_MODELS_100)}")
    print(f"Unified feature dimension: {UNIFIED_FEAT_DIM_100}")
    print()
    print("Model list:")
    for i, name in enumerate(BKB_100_SPECIFIC_RANK):
        dim = MODEL_100_2FEAT_DIM[name]
        source = "torchvision" if name in TORCHVISION_MODELS_100 else "timm"
        print(f"  {i+1:3d}. [{source:12s}] {name:<30} dim={dim}")
