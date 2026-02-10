#!/usr/bin/env python3
"""
DenseNet-161 benchmark with NOVA Winograd F(6,3).

DenseNet-161 has 78 eligible 3x3 stride=1 Conv2d layers — 6x more than
ResNet-50's 13 — with channel sizes ranging from 48 to 192, exercising
a different part of the configuration space.

Measures: single-image correctness, NaN/Inf count, batch=1 end-to-end latency.

Usage:
    python benchmarks/bench_densenet.py
"""

import argparse
import json

import torch
import torch.nn as nn
import torchvision.models as models

from nova_winograd import NovaWinogradConv2d, replace_conv2d_with_nova


def count_conv_layers(model):
    eligible = ineligible = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d) and not isinstance(m, NovaWinogradConv2d):
            if (m.kernel_size == (3, 3) and m.stride == (1, 1)
                    and m.dilation == (1, 1) and m.groups == 1):
                eligible += 1
            else:
                ineligible += 1
        elif isinstance(m, NovaWinogradConv2d):
            eligible += 1
    return eligible, ineligible


def benchmark_model(model, x, warmup=10, repeat=50):
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeat):
            out = model(x)
        end.record()
        torch.cuda.synchronize()
    return start.elapsed_time(end) / repeat, out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda")
    prop = torch.cuda.get_device_properties(0)
    print(f"Device: {prop.name}\nMemory: {prop.total_memory / 1e9:.1f} GB\n")

    print("Loading DenseNet-161 (pretrained)...")
    model = models.densenet161(weights=models.DenseNet161_Weights.IMAGENET1K_V1)
    model = model.cuda().half().eval()

    eligible, ineligible = count_conv_layers(model)
    print(f"DenseNet-161 Conv2d: {eligible} eligible, {ineligible} other\n")

    # Single-image input (224x224)
    torch.manual_seed(args.seed)
    x = torch.randn(1, 3, 224, 224, device=device, dtype=torch.float16)

    # Standard (MIOpen) baseline
    print("=" * 60)
    print("Benchmarking STANDARD DenseNet-161 (MIOpen)...")
    with torch.no_grad():
        ref = model(x).clone()
    ms_std, out_std = benchmark_model(model, x)
    nan_std = torch.isnan(out_std).sum().item()
    inf_std = torch.isinf(out_std).sum().item()
    print(f"  Latency: {ms_std:.2f} ms")
    print(f"  NaN: {nan_std}, Inf: {inf_std}")
    print(f"  Top-5 classes: {torch.topk(out_std, 5).indices.tolist()}\n")

    # Replace with NOVA
    print("Replacing Conv2d with NOVA Winograd F(6,3)...")
    n_replaced = replace_conv2d_with_nova(model)
    print()

    # NOVA benchmark
    print("=" * 60)
    print("Benchmarking NOVA DenseNet-161...")
    ms_nova, out_nova = benchmark_model(model, x)
    rel_err = (torch.norm(out_nova.float() - ref.float()) / torch.norm(ref.float())).item()
    nan_nova = torch.isnan(out_nova).sum().item()
    inf_nova = torch.isinf(out_nova).sum().item()
    print(f"  Latency: {ms_nova:.2f} ms  ({ms_std / ms_nova:.2f}x)")
    print(f"  Rel err vs standard: {rel_err:.6f}")
    print(f"  NaN: {nan_nova}, Inf: {inf_nova}")
    print(f"  Top-5 classes: {torch.topk(out_nova, 5).indices.tolist()}")

    # Check prediction agreement
    pred_std = torch.argmax(ref, dim=1).item()
    pred_nova = torch.argmax(out_nova, dim=1).item()
    agree = pred_std == pred_nova

    # Summary
    print("\n" + "=" * 60)
    print(f"Model: DenseNet-161 ({eligible} eligible layers, channels 48-192)")
    print(f"Layers replaced: {n_replaced}")
    print(f"Standard: {ms_std:.2f} ms  |  NOVA: {ms_nova:.2f} ms  ({ms_std / ms_nova:.2f}x)")
    print(f"Rel err: {rel_err:.6f}  |  NaN/Inf: {nan_nova}/{inf_nova}")
    print(f"Prediction agreement: {'YES' if agree else 'NO'} (std={pred_std}, nova={pred_nova})")

    results = {
        "experiment": "NOVA DenseNet-161",
        "device": prop.name,
        "eligible_layers": eligible,
        "ineligible_layers": ineligible,
        "nova_layers_replaced": n_replaced,
        "channel_range": "48-192",
        "standard_ms": round(ms_std, 2),
        "nova_ms": round(ms_nova, 2),
        "speedup": round(ms_std / ms_nova, 3),
        "rel_err_vs_standard": round(rel_err, 6),
        "prediction_agreement": agree,
        "nova_nan": nan_nova,
        "nova_inf": inf_nova,
    }
    out_path = "nova_densenet_benchmark.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
