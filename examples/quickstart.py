#!/usr/bin/env python3
"""
NOVA Winograd F(6,3) — Quick Start Example

Demonstrates three ways to use NOVA:
1. Single-layer replacement
2. Whole-model surgery
3. Functional API
"""

import torch
import torch.nn as nn

from nova_winograd import (
    NovaWinogradConv2d,
    nova_forward,
    replace_conv2d_with_nova,
)


def example_single_layer():
    """Replace a single Conv2d layer."""
    print("=== Example 1: Single Layer ===")

    # Standard PyTorch conv
    std_conv = nn.Conv2d(64, 128, 3, padding=1, bias=False).cuda()

    # Create NOVA equivalent (copies weights automatically)
    nova_conv = NovaWinogradConv2d.from_conv2d(std_conv)

    # Run inference
    x = torch.randn(1, 64, 56, 56, device="cuda", dtype=torch.float16)
    with torch.no_grad():
        output = nova_conv(x)

    print(f"  Input:  {x.shape}")
    print(f"  Output: {output.shape}")
    print(f"  NaN: {torch.isnan(output).sum().item()}")
    print()


def example_model_surgery():
    """Replace all eligible Conv2d layers in a model."""
    print("=== Example 2: Model Surgery ===")

    # Any model with 3x3 stride=1 convolutions
    model = nn.Sequential(
        nn.Conv2d(3, 64, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
    ).cuda().half()

    # One-line replacement
    n = replace_conv2d_with_nova(model)

    x = torch.randn(1, 3, 32, 32, device="cuda", dtype=torch.float16)
    with torch.no_grad():
        output = model(x)

    print(f"  Replaced {n} layers")
    print(f"  Output:  {output.shape}")
    print()


def example_functional():
    """Use the functional API directly."""
    print("=== Example 3: Functional API ===")

    x = torch.randn(1, 64, 28, 28, device="cuda", dtype=torch.float16)
    w = torch.randn(128, 64, 3, 3, device="cuda", dtype=torch.float32) * 0.1

    output = nova_forward(x, w, padding=1)

    print(f"  Input:  {x.shape} (fp16)")
    print(f"  Weight: {w.shape} (fp32)")
    print(f"  Output: {output.shape} (fp16)")
    print(f"  NaN: {torch.isnan(output).sum().item()}")
    print()


if __name__ == "__main__":
    example_single_layer()
    example_model_surgery()
    example_functional()
    print("All examples completed successfully.")
