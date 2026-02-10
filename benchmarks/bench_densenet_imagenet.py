#!/usr/bin/env python3
"""
DenseNet-161 ImageNetV2 accuracy validation with NOVA Winograd F(6,3).

Runs the full ImageNetV2 matched-frequency test set (10,000 images) through
DenseNet-161 in three configurations:
  1. FP32 baseline (direct convolution)
  2. MIOpen FP16 (standard convolution)
  3. NOVA F(6,3) FP16

Validates that NOVA preserves DenseNet-161's ~77% top-1 accuracy.

Usage:
    python benchmarks/bench_densenet_imagenet.py
    python benchmarks/bench_densenet_imagenet.py --batch-size 64
"""

import argparse
import json
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from nova_winograd import NovaWinogradConv2d, replace_conv2d_with_nova

DATASET_PATH = "/mnt/data/imagenetv2-matched-frequency-format-val"


def evaluate(model, loader, alpha_to_true, device):
    correct_1 = correct_5 = total = 0
    nan_count = inf_count = 0

    # Detect model dtype (FP32 or FP16)
    model_dtype = next(model.parameters()).dtype

    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, dtype=model_dtype)
            # Remap alphabetically-sorted labels to true class indices
            true_labels = torch.tensor(
                [alpha_to_true[l.item()] for l in labels],
                device=device,
            )
            out = model(images)
            nan_count += torch.isnan(out).sum().item()
            inf_count += torch.isinf(out).sum().item()

            _, pred5 = out.topk(5, dim=1)
            total += true_labels.size(0)
            correct_1 += (pred5[:, 0] == true_labels).sum().item()
            correct_5 += (pred5 == true_labels.unsqueeze(1)).any(dim=1).sum().item()

    return {
        "top1": round(100.0 * correct_1 / total, 2),
        "top5": round(100.0 * correct_5 / total, 2),
        "total": total,
        "nan_count": nan_count,
        "inf_count": inf_count,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda")
    prop = torch.cuda.get_device_properties(0)
    print(f"Device: {prop.name}\nMemory: {prop.total_memory / 1e9:.1f} GB\n")

    # ImageNetV2 dataset
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    dataset = datasets.ImageFolder(DATASET_PATH, transform=transform)

    # GOTCHA: ImageFolder sorts class folders alphabetically.
    # Folder names are "0", "1", ..., "999" but sorted as strings:
    # "0", "1", "10", "100", ... We must remap to true numeric indices.
    alpha_to_true = {i: int(c) for i, c in enumerate(dataset.classes)}

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)
    print(f"ImageNetV2: {len(dataset)} images, {len(dataset.classes)} classes\n")

    results = {"experiment": "NOVA DenseNet-161 ImageNetV2", "device": prop.name}

    # 1. FP32 Baseline
    print("=" * 60)
    print("[1/3] FP32 Baseline (direct convolution)...")
    model_fp32 = models.densenet161(weights=models.DenseNet161_Weights.IMAGENET1K_V1)
    model_fp32 = model_fp32.to(device).eval()
    t0 = time.time()
    r = evaluate(model_fp32, loader, alpha_to_true, device)
    t_fp32 = time.time() - t0
    print(f"  Top-1: {r['top1']}%  Top-5: {r['top5']}%  NaN: {r['nan_count']}  ({t_fp32:.1f}s)")
    results["fp32_baseline"] = r
    del model_fp32
    torch.cuda.empty_cache()

    # 2. MIOpen FP16 (standard)
    print("\n[2/3] MIOpen FP16 (standard convolution)...")
    model_fp16 = models.densenet161(weights=models.DenseNet161_Weights.IMAGENET1K_V1)
    model_fp16 = model_fp16.to(device).half().eval()
    t0 = time.time()
    r = evaluate(model_fp16, loader, alpha_to_true, device)
    t_fp16 = time.time() - t0
    print(f"  Top-1: {r['top1']}%  Top-5: {r['top5']}%  NaN: {r['nan_count']}  ({t_fp16:.1f}s)")
    results["miopen_fp16"] = r

    # 3. NOVA F(6,3) FP16
    print("\n[3/3] NOVA F(6,3) FP16...")
    n_replaced = replace_conv2d_with_nova(model_fp16)
    results["nova_layers_replaced"] = n_replaced
    t0 = time.time()
    r = evaluate(model_fp16, loader, alpha_to_true, device)
    t_nova = time.time() - t0
    print(f"  Top-1: {r['top1']}%  Top-5: {r['top5']}%  NaN: {r['nan_count']}  ({t_nova:.1f}s)")
    results["nova_fp16"] = r
    del model_fp16
    torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 60)
    print("DenseNet-161 ImageNetV2 Accuracy Summary")
    print("-" * 60)
    print(f"  FP32 Baseline:   {results['fp32_baseline']['top1']}% top-1  "
          f"({results['fp32_baseline']['nan_count']} NaN)")
    print(f"  MIOpen FP16:     {results['miopen_fp16']['top1']}% top-1  "
          f"({results['miopen_fp16']['nan_count']} NaN)")
    print(f"  NOVA F(6,3) FP16: {results['nova_fp16']['top1']}% top-1  "
          f"({results['nova_fp16']['nan_count']} NaN)")
    delta = results['nova_fp16']['top1'] - results['fp32_baseline']['top1']
    print(f"  NOVA vs FP32: {delta:+.2f}%")
    print(f"  Layers replaced: {n_replaced}")

    out_path = "nova_densenet_imagenet.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
