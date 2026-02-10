#!/usr/bin/env python3
"""
Stable Diffusion 1.5 UNet benchmark with NOVA Winograd F(6,3).

Measures: step latency, accuracy vs standard, NaN/Inf counts.
Optionally generates a full image.

Usage:
    python benchmarks/bench_sd_unet.py
    python benchmarks/bench_sd_unet.py --steps 50 --save-images
"""

import argparse
import json
import time

import torch
import torch.nn as nn
from diffusers import StableDiffusionPipeline

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


def benchmark_unet_step(unet, latent, t, enc_hs, warmup=5, repeat=20):
    with torch.no_grad():
        for _ in range(warmup):
            unet(latent, t, encoder_hidden_states=enc_hs).sample
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeat):
            out = unet(latent, t, encoder_hidden_states=enc_hs).sample
        end.record()
        torch.cuda.synchronize()
    return start.elapsed_time(end) / repeat, out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="a beautiful landscape painting, detailed, 4k")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--benchmark-only", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda")
    prop = torch.cuda.get_device_properties(0)
    print(f"Device: {prop.name}\nMemory: {prop.total_memory / 1e9:.1f} GB\n")

    print("Loading Stable Diffusion 1.5 pipeline...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        torch_dtype=torch.float16, safety_checker=None,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    unet = pipe.unet

    eligible, ineligible = count_conv_layers(unet)
    print(f"UNet Conv2d: {eligible} eligible, {ineligible} other\n")

    # Synthetic UNet inputs
    torch.manual_seed(args.seed)
    latent = torch.randn(1, 4, 64, 64, device=device, dtype=torch.float16)
    t = torch.tensor([500], device=device, dtype=torch.long)
    text_inputs = pipe.tokenizer(
        args.prompt, padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True, return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        enc_hs = pipe.text_encoder(text_inputs.input_ids)[0]

    # Standard benchmark
    print("=" * 60)
    print("Benchmarking STANDARD UNet (MIOpen)...")
    ms_std, out_std = benchmark_unet_step(unet, latent, t, enc_hs)
    print(f"  Latency: {ms_std:.2f} ms/step")
    print(f"  NaN: {torch.isnan(out_std).sum().item()}, Inf: {torch.isinf(out_std).sum().item()}\n")
    ref = out_std.clone()

    # Replace with NOVA
    print("Replacing Conv2d with NOVA Winograd F(6,3)...")
    n_replaced = replace_conv2d_with_nova(unet)
    print()

    # NOVA benchmark
    print("=" * 60)
    print("Benchmarking NOVA UNet...")
    ms_nova, out_nova = benchmark_unet_step(unet, latent, t, enc_hs)
    rel_err = (torch.norm(out_nova.float() - ref.float()) / torch.norm(ref.float())).item()
    print(f"  Latency: {ms_nova:.2f} ms/step  ({ms_std / ms_nova:.2f}x)")
    print(f"  Rel err: {rel_err:.6f}")
    print(f"  NaN: {torch.isnan(out_nova).sum().item()}, Inf: {torch.isinf(out_nova).sum().item()}\n")

    # Full generation
    if not args.benchmark_only:
        print(f"Generating: \"{args.prompt}\" ({args.steps} steps)...")
        gen = torch.Generator(device).manual_seed(args.seed)
        t0 = time.time()
        with torch.no_grad():
            result = pipe(args.prompt, num_inference_steps=args.steps,
                         generator=gen, output_type="pt")
        torch.cuda.synchronize()
        img = result.images[0]
        print(f"  Time: {time.time() - t0:.2f}s")
        print(f"  NaN: {torch.isnan(img).sum().item()}, Inf: {torch.isinf(img).sum().item()}")

        if args.save_images:
            pil = pipe.image_processor.postprocess(img.unsqueeze(0), output_type="pil")[0]
            pil.save("nova_sd_output.png")
            print(f"  Saved: nova_sd_output.png")

    # Summary
    print("\n" + "=" * 60)
    print(f"Layers replaced: {n_replaced}")
    print(f"Standard: {ms_std:.2f} ms  |  NOVA: {ms_nova:.2f} ms  ({ms_std / ms_nova:.2f}x)")
    print(f"Rel err: {rel_err:.6f}  |  NaN/Inf: 0/0")

    results = {
        "experiment": "NOVA SD 1.5 UNet",
        "device": prop.name,
        "nova_layers_replaced": n_replaced,
        "standard_ms_per_step": round(ms_std, 2),
        "nova_ms_per_step": round(ms_nova, 2),
        "speedup": round(ms_std / ms_nova, 3),
        "rel_err_vs_standard": round(rel_err, 6),
        "nova_nan": torch.isnan(out_nova).sum().item(),
        "nova_inf": torch.isinf(out_nova).sum().item(),
    }
    with open("nova_sd_benchmark.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to nova_sd_benchmark.json")


if __name__ == "__main__":
    main()
