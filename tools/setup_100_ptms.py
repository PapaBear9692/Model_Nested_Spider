"""
100 PTM Setup Script for Model-Spider

This script:
1. Downloads 100 pre-trained models from torchvision and timm
2. Extracts feature dimensions for each model
3. Saves model info for Model-Spider configuration

Usage:
    python tools/setup_100_ptms.py --download --info
"""

import os
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# =============================================================================
# 100 PRE-TRAINED MODELS SELECTION
# =============================================================================

PTM_100_LIST = [
    # === ResNet Family (5) ===
    ("torchvision", "resnet18", 512),
    ("torchvision", "resnet34", 512),
    ("torchvision", "resnet50", 2048),
    ("torchvision", "resnet101", 2048),
    ("torchvision", "resnet152", 2048),

    # === DenseNet Family (4) ===
    ("torchvision", "densenet121", 1024),
    ("torchvision", "densenet161", 2208),
    ("torchvision", "densenet169", 1664),
    ("torchvision", "densenet201", 1920),

    # === VGG Family (8) ===
    ("torchvision", "vgg11", 4096),
    ("torchvision", "vgg13", 4096),
    ("torchvision", "vgg16", 4096),
    ("torchvision", "vgg19", 4096),
    ("torchvision", "vgg11_bn", 4096),
    ("torchvision", "vgg13_bn", 4096),
    ("torchvision", "vgg16_bn", 4096),
    ("torchvision", "vgg19_bn", 4096),

    # === EfficientNet Family (8) ===
    ("torchvision", "efficientnet_b0", 1280),
    ("torchvision", "efficientnet_b1", 1280),
    ("torchvision", "efficientnet_b2", 1408),
    ("torchvision", "efficientnet_b3", 1536),
    ("torchvision", "efficientnet_b4", 1792),
    ("torchvision", "efficientnet_b5", 2048),
    ("torchvision", "efficientnet_b6", 2304),
    ("torchvision", "efficientnet_b7", 2560),

    # === RegNetY Family (7) ===
    ("torchvision", "regnet_y_400mf", 440),
    ("torchvision", "regnet_y_800mf", 560),
    ("torchvision", "regnet_y_1_6gf", 888),
    ("torchvision", "regnet_y_3_2gf", 1512),
    ("torchvision", "regnet_y_8gf", 2016),
    ("torchvision", "regnet_y_16gf", 2280),
    ("torchvision", "regnet_y_32gf", 2640),

    # === RegNetX Family (7) ===
    ("torchvision", "regnet_x_400mf", 400),
    ("torchvision", "regnet_x_800mf", 672),
    ("torchvision", "regnet_x_1_6gf", 912),
    ("torchvision", "regnet_x_3_2gf", 1008),
    ("torchvision", "regnet_x_8gf", 1920),
    ("torchvision", "regnet_x_16gf", 2048),
    ("torchvision", "regnet_x_32gf", 2320),

    # === MobileNet Family (3) ===
    ("torchvision", "mobilenet_v2", 1280),
    ("torchvision", "mobilenet_v3_small", 576),
    ("torchvision", "mobilenet_v3_large", 960),

    # === MNASNet Family (2) ===
    ("torchvision", "mnasnet0_5", 1280),
    ("torchvision", "mnasnet1_0", 1280),

    # === ShuffleNet Family (4) ===
    ("torchvision", "shufflenet_v2_x0_5", 1024),
    ("torchvision", "shufflenet_v2_x1_0", 1024),
    ("torchvision", "shufflenet_v2_x1_5", 1024),
    ("torchvision", "shufflenet_v2_x2_0", 2048),

    # === SqueezeNet Family (2) ===
    ("torchvision", "squeezenet1_0", 512),
    ("torchvision", "squeezenet1_1", 512),

    # === GoogleNet/Inception (3) ===
    ("torchvision", "googlenet", 1024),
    ("torchvision", "inception_v3", 2048),
    ("torchvision", "alexnet", 4096),

    # === ConvNeXt Family (4) ===
    ("torchvision", "convnext_tiny", 768),
    ("torchvision", "convnext_small", 768),
    ("torchvision", "convnext_base", 1024),
    ("torchvision", "convnext_large", 1536),

    # === Vision Transformer (4) ===
    ("torchvision", "vit_b_16", 768),
    ("torchvision", "vit_b_32", 768),
    ("torchvision", "vit_l_16", 1024),
    ("torchvision", "vit_l_32", 1024),

    # === Swin Transformer (4) ===
    ("torchvision", "swin_t", 768),
    ("torchvision", "swin_s", 768),
    ("torchvision", "swin_b", 1024),
    ("torchvision", "swin_v2_t", 768),

    # === TIMM: EfficientNetV2 (4) ===
    ("timm", "efficientnetv2_s", 1280),
    ("timm", "efficientnetv2_m", 1280),
    ("timm", "efficientnetv2_l", 1280),
    ("timm", "efficientnetv2_xl", 1280),

    # === TIMM: DeiT (4) ===
    ("timm", "deit_tiny_patch16_224", 192),
    ("timm", "deit_small_patch16_224", 384),
    ("timm", "deit_base_patch16_224", 768),
    ("timm", "deit_base_patch16_384", 768),

    # === TIMM: MobileViT (3) ===
    ("timm", "mobilevit_s", 640),
    ("timm", "mobilevit_xs", 384),
    ("timm", "mobilevit_xxs", 320),

    # === TIMM: MaxViT (3) ===
    ("timm", "maxvit_tiny_224", 512),
    ("timm", "maxvit_small_224", 768),
    ("timm", "maxvit_base_224", 1024),

    # === TIMM: ConvNeXtV2 (4) ===
    ("timm", "convnextv2_atto", 320),
    ("timm", "convnextv2_femto", 384),
    ("timm", "convnextv2_nano", 512),
    ("timm", "convnextv2_tiny", 768),

    # === TIMM: CoaT (3) ===
    ("timm", "coat_lite_tiny", 192),
    ("timm", "coat_lite_small", 320),
    ("timm", "coat_lite_medium", 512),

    # === TIMM: LeViT (3) ===
    ("timm", "levit_128s", 384),
    ("timm", "levit_128", 384),
    ("timm", "levit_192", 384),

    # === TIMM: HRNet (3) ===
    ("timm", "hrnet_w18_small", 2048),
    ("timm", "hrnet_w18", 2048),
    ("timm", "hrnet_w30", 2048),

    # === Additional models to reach 100 ===
    ("timm", "twins_pcpvt_small", 768),
    ("timm", "twins_pcpvt_base", 768),
    ("timm", "twins_pcpvt_large", 768),
    ("timm", "tnt_s_patch16_224", 640),
    ("timm", "pit_b_224", 768),
    ("timm", "pit_s_224", 576),
    ("timm", "pit_xs_224", 384),
    ("timm", "cait_s24_224", 768),
]

# Verify we have exactly 100
assert len(PTM_100_LIST) == 100, f"Expected 100 PTMs, got {len(PTM_100_LIST)}"


def download_models(save_dir: str):
    """Download all 100 pre-trained models."""
    import torch

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading 100 PTMs to {save_dir}...")

    for i, (source, model_name, feat_dim) in enumerate(PTM_100_LIST):
        print(f"\n[{i+1}/100] Downloading {model_name}...")

        try:
            if source == "torchvision":
                import torchvision.models as models
                model_fn = getattr(models, model_name)
                model = model_fn(weights="DEFAULT")
            else:  # timm
                import timm
                model = timm.create_model(model_name, pretrained=True)

            # Save model weights
            torch.save(
                model.state_dict(),
                save_path / f"{model_name}.pth"
            )
            print(f"  ✓ Saved {model_name} (feat_dim={feat_dim})")

            # Free memory
            del model
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  ✗ Failed: {e}")

    print(f"\nDownload complete! Models saved to {save_dir}")


def generate_model_info(output_file: str):
    """Generate learnware_info.py compatible configuration."""

    # Create model name to feature dimension mapping
    model2feat_dim = {}
    bkb_specific_rank = []

    for source, model_name, feat_dim in PTM_100_LIST:
        model2feat_dim[model_name] = feat_dim
        bkb_specific_rank.append(model_name)

    # Create BKB_SPECIFIC_RANK2ID mapping
    bkb_specific_rank2id = {name: i for i, name in enumerate(bkb_specific_rank)}

    # Generate Python code
    code = f'''# Auto-generated 100 PTM configuration
# Generated by tools/setup_100_ptms.py

BKB_100_SPECIFIC_RANK = {bkb_specific_rank}

BKB_100_SPECIFIC_RANK2ID = {bkb_specific_rank2id}

MODEL_100_2FEAT_DIM = {model2feat_dim}
'''

    with open(output_file, 'w') as f:
        f.write(code)

    print(f"Generated configuration saved to {output_file}")

    # Also save as JSON for reference
    json_file = output_file.replace('.py', '.json')
    with open(json_file, 'w') as f:
        json.dump({
            "models": bkb_specific_rank,
            "feature_dims": model2feat_dim
        }, f, indent=2)
    print(f"JSON reference saved to {json_file}")


def print_summary():
    """Print summary of selected models."""
    print("=" * 70)
    print("100 PTM MODEL ZOO SUMMARY")
    print("=" * 70)

    # Group by source
    torchvision_count = sum(1 for s, _, _ in PTM_100_LIST if s == "torchvision")
    timm_count = sum(1 for s, _, _ in PTM_100_LIST if s == "timm")

    print(f"\nTotal PTMs: {len(PTM_100_LIST)}")
    print(f"  - Torchvision: {torchvision_count}")
    print(f"  - TIMM: {timm_count}")

    # Group by architecture family
    families = {
        "ResNet": [], "DenseNet": [], "VGG": [], "EfficientNet": [],
        "RegNet": [], "MobileNet": [], "ShuffleNet": [], "SqueezeNet": [],
        "ConvNeXt": [], "ViT": [], "Swin": [], "DeiT": [],
        "MaxViT": [], "CoaT": [], "LeViT": [], "HRNet": [], "Other": []
    }

    for _, name, dim in PTM_100_LIST:
        name_lower = name.lower()
        categorized = False
        for family in families:
            if family.lower() in name_lower:
                families[family].append((name, dim))
                categorized = True
                break
        if not categorized:
            families["Other"].append((name, dim))

    print("\n" + "-" * 70)
    print("BY ARCHITECTURE FAMILY:")
    print("-" * 70)

    for family, models in families.items():
        if models:
            print(f"\n{family} ({len(models)} models):")
            for name, dim in models:
                print(f"  - {name:<30} dim={dim}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setup 100 PTMs for Model-Spider")
    parser.add_argument("--download", action="store_true", help="Download model weights")
    parser.add_argument("--save_dir", type=str, default="required_files/pretrained_100", help="Directory to save weights")
    parser.add_argument("--info", action="store_true", help="Generate model info configuration")
    parser.add_argument("--output", type=str, default="learnware/learnware_info_100.py", help="Output file for model info")
    parser.add_argument("--summary", action="store_true", help="Print summary of selected models")

    args = parser.parse_args()

    if args.summary or (not args.download and not args.info):
        print_summary()

    if args.download:
        download_models(args.save_dir)

    if args.info:
        generate_model_info(args.output)
