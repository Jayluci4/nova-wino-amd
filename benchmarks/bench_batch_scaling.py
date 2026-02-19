#!/usr/bin/env python3
"""
Batch-scaling benchmark: NOVA vs MIOpen across batch sizes 1-64.

Identifies the exact crossover point where MIOpen's fused F(2,3) overtakes
NOVA's multi-pass F(6,3), and measures how the gap scales.

Usage:
    python benchmarks/bench_batch_scaling.py
"""

import json
import torch
import torch.nn as nn

from nova_winograd import NovaWinogradConv2d
from nova_winograd.ops import compute_tiling


def benchmark_fn(fn, warmup=20, repeat=100):
    """Time a function using CUDA events (ms)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeat


def main():
    torch.cuda.init()
    prop = torch.cuda.get_device_properties(0)
    print(f"Device: {prop.name}")
    print(f"CUs: {prop.multi_processor_count}, VRAM: {prop.total_memory / 1e9:.1f} GB\n")

    layers = [
        ("conv2_x", 64,  64,  56, 56),
        ("conv3_x", 128, 128, 28, 28),
        ("conv4_x", 256, 256, 14, 14),
        ("conv5_x", 512, 512,  7,  7),
    ]
    batch_sizes = [1, 2, 4, 8, 16, 32]

    all_results = []

    for layer_name, C, K, H, W in layers:
        print(f"\n{'='*80}")
        print(f"  {layer_name}: C={C}, K={K}, {H}x{H}")
        print(f"{'='*80}")

        hdr = f"{'Batch':>6} {'MIOpen':>10} {'NOVA':>10} {'Ratio':>8} {'NOVA BW':>10} {'Traffic':>10} {'ArithInt':>10}"
        print(hdr)
        print("-" * len(hdr))

        for batch in batch_sizes:
            pad = 1
            nh, nw, H_out, W_out = compute_tiling(H, W, pad)
            P = nh * nw
            BP = batch * P
            flops = 2.0 * batch * K * C * H_out * W_out * 9

            # Memory traffic (multi-pass)
            V_bytes = 64 * C * BP * 2
            M_bytes = 64 * K * BP * 2
            U_bytes = 64 * K * C * 2
            input_bytes = batch * C * H * W * 2
            output_bytes = batch * K * H_out * W_out * 2
            total_traffic = input_bytes + V_bytes + (U_bytes + V_bytes + M_bytes) + M_bytes + output_bytes
            arith_intensity = flops / total_traffic

            x = torch.randn(batch, C, H, W, device="cuda", dtype=torch.float16)
            w_fp32 = torch.randn(K, C, 3, 3, device="cuda", dtype=torch.float32) * 0.1

            # MIOpen
            conv_mio = nn.Conv2d(C, K, 3, padding=pad, bias=False).cuda().half()
            conv_mio.weight.data.copy_(w_fp32.half())
            with torch.no_grad():
                ms_mio = benchmark_fn(lambda: conv_mio(x))

            # NOVA
            nova = NovaWinogradConv2d(C, K, kernel_size=3, padding=pad).cuda()
            nova.weight.data.copy_(w_fp32)
            _ = nova(x)
            with torch.no_grad():
                ms_nova = benchmark_fn(lambda: nova(x))

            ratio = ms_nova / ms_mio
            bw = total_traffic / (ms_nova * 1e6)  # GB/s

            print(f"{batch:>6} {ms_mio:>8.3f}ms {ms_nova:>8.3f}ms {ratio:>7.2f}x "
                  f"{bw:>8.0f}GB/s {total_traffic/1e6:>8.1f}MB {arith_intensity:>9.1f}")

            all_results.append({
                "layer": layer_name, "batch": batch,
                "C": C, "K": K, "H": H, "W": W,
                "tiles": P, "BP": BP,
                "ms_miopen": round(ms_mio, 4),
                "ms_nova": round(ms_nova, 4),
                "ratio": round(ratio, 3),
                "traffic_MB": round(total_traffic / 1e6, 2),
                "bandwidth_GBs": round(bw, 1),
                "arith_intensity": round(arith_intensity, 2),
                "flops": flops,
            })

            del x, conv_mio, nova
            torch.cuda.empty_cache()

    # Find crossover points
    print("\n\n=== CROSSOVER ANALYSIS ===\n")
    for layer_name, C, K, H, W in layers:
        layer_results = [r for r in all_results if r["layer"] == layer_name]
        crossover = None
        for i, r in enumerate(layer_results):
            if r["ratio"] > 1.0 and (i == 0 or layer_results[i-1]["ratio"] <= 1.0):
                crossover = r["batch"]
                break
        if crossover:
            print(f"{layer_name}: NOVA loses starting at batch={crossover}")
        else:
            nova_always_wins = all(r["ratio"] <= 1.0 for r in layer_results)
            if nova_always_wins:
                print(f"{layer_name}: NOVA wins at all tested batch sizes")
            else:
                print(f"{layer_name}: NOVA loses at batch=1 already")

    output = {
        "experiment": "NOVA vs MIOpen Batch Scaling",
        "device": prop.name,
        "results": all_results,
    }
    with open("nova_batch_scaling.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to nova_batch_scaling.json")


if __name__ == "__main__":
    main()
