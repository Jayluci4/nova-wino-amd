#!/usr/bin/env python3
"""
SDXL Base UNet benchmark with NOVA Winograd F(6,3).

Measures: step latency, accuracy vs standard, NaN/Inf counts.
Optionally generates a full 1024x1024 image.

Usage:
    python benchmarks/bench_sdxl.py
    python benchmarks/bench_sdxl.py --steps 30 --save-images
"""

import argparse
import json
import time

import torch
import torch.nn as nn
from diffusers import StableDiffusionXLPipeline

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


def benchmark_unet_step(unet, latent, t, enc_hs, added_cond_kwargs,
                        warmup=5, repeat=20):
    with torch.no_grad():
        for _ in range(warmup):
            unet(latent, t, encoder_hidden_states=enc_hs,
                 added_cond_kwargs=added_cond_kwargs).sample
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeat):
            out = unet(latent, t, encoder_hidden_states=enc_hs,
                       added_cond_kwargs=added_cond_kwargs).sample
        end.record()
        torch.cuda.synchronize()
    return start.elapsed_time(end) / repeat, out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="a majestic mountain landscape at sunset, photorealistic, 8k")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--benchmark-only", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda")
    prop = torch.cuda.get_device_properties(0)
    print(f"Device: {prop.name}\nMemory: {prop.total_memory / 1e9:.1f} GB\n")

    print("Loading SDXL Base pipeline...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    unet = pipe.unet

    eligible, ineligible = count_conv_layers(unet)
    print(f"SDXL UNet Conv2d: {eligible} eligible, {ineligible} other\n")

    # Synthetic UNet inputs for SDXL (128x128 latent = 1024x1024 image)
    torch.manual_seed(args.seed)
    latent = torch.randn(1, 4, 128, 128, device=device, dtype=torch.float16)
    t = torch.tensor([500], device=device, dtype=torch.long)

    # SDXL uses dual text encoders
    text_inputs = pipe.tokenizer(
        args.prompt, padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True, return_tensors="pt",
    ).to(device)
    text_inputs_2 = pipe.tokenizer_2(
        args.prompt, padding="max_length",
        max_length=pipe.tokenizer_2.model_max_length,
        truncation=True, return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        enc_out_1 = pipe.text_encoder(text_inputs.input_ids, output_hidden_states=True)
        enc_out_2 = pipe.text_encoder_2(text_inputs_2.input_ids, output_hidden_states=True)
        # SDXL concatenates penultimate hidden states from both encoders
        enc_hs = torch.cat([enc_out_1.hidden_states[-2], enc_out_2.hidden_states[-2]], dim=-1)
        # Pooled output from text_encoder_2
        pooled = enc_out_2[0]

    # SDXL added conditioning
    add_time_ids = torch.tensor(
        [[1024.0, 1024.0, 0.0, 0.0, 1024.0, 1024.0]],
        device=device, dtype=torch.float16,
    )
    added_cond_kwargs = {"text_embeds": pooled, "time_ids": add_time_ids}

    # Standard benchmark
    print("=" * 60)
    print("Benchmarking STANDARD SDXL UNet (MIOpen)...")
    ms_std, out_std = benchmark_unet_step(unet, latent, t, enc_hs, added_cond_kwargs)
    print(f"  Latency: {ms_std:.2f} ms/step")
    nan_std = torch.isnan(out_std).sum().item()
    inf_std = torch.isinf(out_std).sum().item()
    print(f"  NaN: {nan_std}, Inf: {inf_std}\n")
    ref = out_std.clone()

    # Replace with NOVA
    print("Replacing Conv2d with NOVA Winograd F(6,3)...")
    n_replaced = replace_conv2d_with_nova(unet)
    print()

    # NOVA benchmark
    print("=" * 60)
    print("Benchmarking NOVA SDXL UNet...")
    ms_nova, out_nova = benchmark_unet_step(unet, latent, t, enc_hs, added_cond_kwargs)
    rel_err = (torch.norm(out_nova.float() - ref.float()) / torch.norm(ref.float())).item()
    nan_nova = torch.isnan(out_nova).sum().item()
    inf_nova = torch.isinf(out_nova).sum().item()
    print(f"  Latency: {ms_nova:.2f} ms/step  ({ms_std / ms_nova:.2f}x)")
    print(f"  Rel err: {rel_err:.6f}")
    print(f"  NaN: {nan_nova}, Inf: {inf_nova}\n")

    # Full generation
    if not args.benchmark_only:
        print(f"Generating 1024x1024: \"{args.prompt}\" ({args.steps} steps)...")
        gen = torch.Generator(device).manual_seed(args.seed)
        t0 = time.time()
        with torch.no_grad():
            result = pipe(args.prompt, num_inference_steps=args.steps,
                         generator=gen, output_type="pt")
        torch.cuda.synchronize()
        img = result.images[0]
        gen_time = time.time() - t0
        gen_nan = torch.isnan(img).sum().item()
        gen_inf = torch.isinf(img).sum().item()
        print(f"  Time: {gen_time:.2f}s")
        print(f"  NaN: {gen_nan}, Inf: {gen_inf}")

        if args.save_images:
            pil = pipe.image_processor.postprocess(img.unsqueeze(0), output_type="pil")[0]
            out_path = "docs/report/figures/sdxl_output.png"
            pil.save(out_path)
            print(f"  Saved: {out_path}")

    # Summary
    print("\n" + "=" * 60)
    print(f"Model: SDXL Base UNet (1024x1024)")
    print(f"Layers replaced: {n_replaced}")
    print(f"Standard: {ms_std:.2f} ms  |  NOVA: {ms_nova:.2f} ms  ({ms_std / ms_nova:.2f}x)")
    print(f"Rel err: {rel_err:.6f}  |  NaN/Inf: {nan_nova}/{inf_nova}")

    results = {
        "experiment": "NOVA SDXL Base UNet",
        "device": prop.name,
        "resolution": "1024x1024",
        "latent_shape": "1x4x128x128",
        "eligible_layers": eligible,
        "nova_layers_replaced": n_replaced,
        "standard_ms_per_step": round(ms_std, 2),
        "nova_ms_per_step": round(ms_nova, 2),
        "speedup": round(ms_std / ms_nova, 3),
        "rel_err_vs_standard": round(rel_err, 6),
        "nova_nan": nan_nova,
        "nova_inf": inf_nova,
    }
    if not args.benchmark_only:
        results["generation_time_s"] = round(gen_time, 2)
        results["generation_steps"] = args.steps
        results["generation_nan"] = gen_nan
        results["generation_inf"] = gen_inf

    out_json = "nova_sdxl_benchmark.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_json}")


if __name__ == "__main__":
    main()
