#!/usr/bin/env python3
"""
NOVA Winograd F(6,3) — 60-Second Live Demo

Demonstrates drop-in Winograd convolution replacement on two models:
  1. ResNet-50: ImageNet classification (224×224)
  2. SDXL: Image generation (1024×1024)

Usage:
    python demo.py              # Quick demo (ResNet-50 only)
    python demo.py --full       # Full demo (ResNet-50 + SDXL generation)
"""

import argparse
import time

import torch
import torch.nn as nn
import torchvision.models as models

from nova_winograd import NovaWinogradConv2d, replace_conv2d_with_nova


def count_eligible(model):
    n = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d) and not isinstance(m, NovaWinogradConv2d):
            if (m.kernel_size == (3, 3) and m.stride == (1, 1)
                    and m.dilation == (1, 1) and m.groups == 1):
                n += 1
        elif isinstance(m, NovaWinogradConv2d):
            n += 1
    return n


def demo_resnet():
    print("=" * 60)
    print("PART 1: ResNet-50 — ImageNet Classification")
    print("=" * 60)

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model = model.cuda().half().eval()
    x = torch.randn(1, 3, 224, 224, device="cuda", dtype=torch.float16)

    # Standard inference
    with torch.no_grad():
        ref = model(x)
    pred_std = torch.argmax(ref).item()

    # Replace with NOVA
    n = replace_conv2d_with_nova(model)

    # NOVA inference
    with torch.no_grad():
        out = model(x)
    pred_nova = torch.argmax(out).item()
    rel_err = (torch.norm(out.float() - ref.float()) / torch.norm(ref.float())).item()
    nan_count = torch.isnan(out).sum().item()

    print(f"\n  Layers replaced: {n}")
    print(f"  Prediction: std={pred_std}, nova={pred_nova} ({'MATCH' if pred_std == pred_nova else 'DIFFER'})")
    print(f"  Rel error: {rel_err:.4f}")
    print(f"  NaN/Inf: {nan_count} / {torch.isinf(out).sum().item()}")

    # Quick latency check
    with torch.no_grad():
        for _ in range(10):
            model(x)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(50):
            model(x)
        end.record()
        torch.cuda.synchronize()
    ms = start.elapsed_time(end) / 50
    print(f"  NOVA latency: {ms:.2f} ms/image")

    del model
    torch.cuda.empty_cache()
    return n, rel_err, nan_count, ms


def demo_sdxl():
    print("\n" + "=" * 60)
    print("PART 2: SDXL — 1024×1024 Image Generation")
    print("=" * 60)

    from diffusers import StableDiffusionXLPipeline

    print("\nLoading SDXL pipeline...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16, variant="fp16", use_safetensors=True,
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    eligible = count_eligible(pipe.unet)
    n = replace_conv2d_with_nova(pipe.unet)

    print(f"\nGenerating 1024×1024 image...")
    gen = torch.Generator("cuda").manual_seed(42)
    t0 = time.time()
    with torch.no_grad():
        result = pipe("a beautiful sunset over mountains, photorealistic",
                     num_inference_steps=20, generator=gen, output_type="pt")
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    img = result.images[0]
    nan_count = torch.isnan(img).sum().item()

    pil = pipe.image_processor.postprocess(img.unsqueeze(0), output_type="pil")[0]
    pil.save("nova_sdxl_demo.png")

    print(f"  SDXL layers replaced: {n}/{eligible}")
    print(f"  Generation time: {elapsed:.1f}s (20 steps)")
    print(f"  NaN/Inf: {nan_count} / {torch.isinf(img).sum().item()}")
    print(f"  Saved: nova_sdxl_demo.png")

    del pipe
    torch.cuda.empty_cache()
    return n, elapsed, nan_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Include SDXL generation")
    args = parser.parse_args()

    prop = torch.cuda.get_device_properties(0)
    print(f"NOVA Winograd F(6,3) — Live Demo")
    print(f"Device: {prop.name} ({prop.total_memory / 1e9:.0f} GB)\n")

    rn_layers, rn_err, rn_nan, rn_ms = demo_resnet()

    sdxl_layers = sdxl_time = sdxl_nan = None
    if args.full:
        sdxl_layers, sdxl_time, sdxl_nan = demo_sdxl()

    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Model':<20} {'Layers':>8} {'Error':>10} {'NaN':>6} {'Note'}")
    print("-" * 60)
    print(f"{'ResNet-50':<20} {rn_layers:>8} {rn_err:>10.4f} {rn_nan:>6} {rn_ms:.1f} ms/img")
    if sdxl_layers is not None:
        print(f"{'SDXL Base':<20} {sdxl_layers:>8} {'N/A':>10} {sdxl_nan:>6} {sdxl_time:.1f}s total")
    print()
    print("Key: All NaN counts should be 0. Rel error should be <0.1.")
    print("NOVA F(6,3): 5.6x fewer arithmetic ops than MIOpen's F(2,3).")


if __name__ == "__main__":
    main()
