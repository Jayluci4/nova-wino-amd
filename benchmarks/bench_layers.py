#!/usr/bin/env python3
"""
Performance benchmark: NOVA HIP kernel vs MIOpen native vs Python Winograd.

Covers all ResNet-50 3x3 stride=1 layers at batch sizes 1, 8, 32.

Usage:
    python benchmarks/bench_layers.py
"""

import json
import torch
import torch.nn as nn

from nova_winograd import NovaWinogradConv2d


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
    prop = torch.cuda.get_device_properties(0)
    print(f"Device: {prop.name}")
    print(f"CUs: {prop.multi_processor_count}, Memory: {prop.total_memory / 1e9:.1f} GB\n")

    configs = [
        ("conv2_x [B=1]",   1,   64,   64, 56, 56),
        ("conv3_x [B=1]",   1,  128,  128, 28, 28),
        ("conv4_x [B=1]",   1,  256,  256, 14, 14),
        ("conv5_x [B=1]",   1,  512,  512,  7,  7),
        ("conv2_x [B=8]",   8,   64,   64, 56, 56),
        ("conv3_x [B=8]",   8,  128,  128, 28, 28),
        ("conv4_x [B=8]",   8,  256,  256, 14, 14),
        ("conv5_x [B=8]",   8,  512,  512,  7,  7),
        ("conv2_x [B=32]", 32,   64,   64, 56, 56),
        ("conv3_x [B=32]", 32,  128,  128, 28, 28),
    ]

    results = []
    header = f"{'Config':<22} {'MIOpen':>10} {'NOVA HIP':>10} {'HIP/MIO':>8} {'Equiv GFLOPS':>12}"
    print(header)
    print("=" * len(header))

    for name, batch, C, K, H, W in configs:
        pad = 1
        flops = 2.0 * batch * K * C * H * W * 9
        x = torch.randn(batch, C, H, W, device="cuda", dtype=torch.float16)
        w_fp32 = torch.randn(K, C, 3, 3, device="cuda", dtype=torch.float32) * 0.1

        # MIOpen native
        conv_mio = nn.Conv2d(C, K, 3, padding=pad, bias=False).cuda().half()
        conv_mio.weight.data.copy_(w_fp32.half())
        with torch.no_grad():
            ms_mio = benchmark_fn(lambda: conv_mio(x))

        # NOVA HIP
        nova = NovaWinogradConv2d(C, K, kernel_size=3, padding=pad).cuda()
        nova.weight.data.copy_(w_fp32)
        _ = nova(x)  # warmup / cache weights
        with torch.no_grad():
            ms_nova = benchmark_fn(lambda: nova(x))

        ratio = ms_nova / ms_mio
        gflops = flops / (ms_nova * 1e6)

        print(f"{name:<22} {ms_mio:>8.3f}ms {ms_nova:>8.3f}ms {ratio:>7.2f}x {gflops:>10.0f}")

        results.append({
            "config": name, "batch": batch, "C": C, "K": K, "H": H, "W": W,
            "ms_miopen": round(ms_mio, 4),
            "ms_nova_hip": round(ms_nova, 4),
            "ratio_vs_miopen": round(ratio, 3),
            "equiv_gflops": round(gflops, 1),
            "flops": flops,
        })

    total_nova = sum(r["ms_nova_hip"] for r in results)
    total_mio = sum(r["ms_miopen"] for r in results)
    print(f"\nAggregate: NOVA {total_nova:.2f}ms vs MIOpen {total_mio:.2f}ms "
          f"({total_nova / total_mio:.1f}x)")

    output = {
        "experiment": "NOVA Winograd F(6,3) HIP Kernel Benchmark",
        "device": prop.name,
        "results": results,
        "aggregate": {
            "total_nova_ms": round(total_nova, 3),
            "total_miopen_ms": round(total_mio, 3),
            "nova_vs_miopen_ratio": round(total_nova / total_mio, 2),
        },
    }
    out_path = "nova_kernel_benchmark.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
