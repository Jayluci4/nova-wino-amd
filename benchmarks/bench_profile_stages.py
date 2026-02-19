#!/usr/bin/env python3
"""
Per-stage profiling of the NOVA 3-pass Winograd pipeline.

Measures individual timings for:
  1. Input transform kernel
  2. rocBLAS batched GEMM
  3. Output transform kernel
  4. Kernel launch overhead (empty kernel baseline)
  5. Total end-to-end

Also computes analytical memory traffic per stage.

Usage:
    python benchmarks/bench_profile_stages.py
"""

import ctypes
import json
import torch
import torch.nn as nn

from nova_winograd.ops import get_lib, ptr, compute_tiling, _ensure_lib


def time_event_pair(fn, warmup=20, repeat=200):
    """Time a function using CUDA events, return ms."""
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


def profile_config(name, batch, C, K, H, W, pad=1):
    """Profile each stage independently for a given config."""
    lib = get_lib()

    nh, nw, H_out, W_out = compute_tiling(H, W, pad)
    P = nh * nw
    BP = batch * P

    # Allocate tensors
    x = torch.randn(batch, C, H, W, device="cuda", dtype=torch.float16)
    w = torch.randn(K, C, 3, 3, device="cuda", dtype=torch.float32) * 0.1
    output = torch.empty(batch, K, H_out, W_out, device="cuda", dtype=torch.float16)

    V_gemm = torch.empty(64, C, BP, device="cuda", dtype=torch.float16)
    M_gemm = torch.empty(64, K, BP, device="cuda", dtype=torch.float16)
    U_gemm = torch.empty(64, K, C, device="cuda", dtype=torch.float16)

    # Filter transform (one-time cost, not timed in pipeline)
    lib.nova_filter_transform(ptr(w), ptr(U_gemm), K, C)
    torch.cuda.synchronize()

    # Create handle and set weights
    handle = lib.nova_create()
    lib.nova_set_weights(handle, ptr(w), K, C)
    torch.cuda.synchronize()

    # --- Stage 1: Input Transform ---
    def run_input_transform():
        lib.nova_input_transform(
            ptr(x), ptr(V_gemm),
            batch, C, H, W, pad, nh, nw, BP
        )

    ms_input = time_event_pair(run_input_transform)

    # --- Stage 2: rocBLAS GEMM (via full forward minus transforms) ---
    # We time the full forward_workspace and subtract transforms
    def run_full():
        lib.nova_forward_workspace(
            handle, ptr(x), ptr(output),
            ptr(V_gemm), ptr(M_gemm),
            batch, H, W, pad, nh, nw
        )

    ms_full = time_event_pair(run_full)

    # --- Stage 3: Output Transform ---
    def run_output_transform():
        lib.nova_output_transform(
            ptr(M_gemm), ptr(output),
            batch, K, H_out, W_out, nh, nw, BP
        )

    ms_output = time_event_pair(run_output_transform)

    # GEMM is approximately: full - input - output
    ms_gemm_approx = max(0, ms_full - ms_input - ms_output)

    # --- MIOpen baseline ---
    conv_mio = nn.Conv2d(C, K, 3, padding=pad, bias=False).cuda().half()
    conv_mio.weight.data.copy_(w.half())
    with torch.no_grad():
        ms_miopen = time_event_pair(lambda: conv_mio(x))

    # --- Memory traffic analysis (bytes) ---
    input_bytes = batch * C * H * W * 2
    output_bytes = batch * K * H_out * W_out * 2
    V_bytes = 64 * C * BP * 2
    M_bytes = 64 * K * BP * 2
    U_bytes = 64 * K * C * 2

    # Input transform: read input + write V_gemm
    traffic_input = input_bytes + V_bytes
    # GEMM: read U_gemm + read V_gemm + write M_gemm
    traffic_gemm = U_bytes + V_bytes + M_bytes
    # Output transform: read M_gemm + write output
    traffic_output = M_bytes + output_bytes
    # Total multi-pass traffic
    traffic_total = traffic_input + traffic_gemm + traffic_output
    # Fused baseline traffic (input + filter + output only)
    traffic_fused = input_bytes + U_bytes + output_bytes

    # Equivalent FLOPs for the convolution
    flops = 2.0 * batch * K * C * H_out * W_out * 9

    lib.nova_destroy(handle)

    result = {
        "config": name,
        "batch": batch, "C": C, "K": K, "H": H, "W": W,
        "tiles": P, "BP": BP, "nh": nh, "nw": nw,
        "ms_input_transform": round(ms_input, 4),
        "ms_gemm_approx": round(ms_gemm_approx, 4),
        "ms_output_transform": round(ms_output, 4),
        "ms_full_nova": round(ms_full, 4),
        "ms_miopen": round(ms_miopen, 4),
        "nova_vs_miopen": round(ms_full / ms_miopen, 3),
        "pct_input_transform": round(100 * ms_input / ms_full, 1),
        "pct_gemm": round(100 * ms_gemm_approx / ms_full, 1),
        "pct_output_transform": round(100 * ms_output / ms_full, 1),
        "traffic_input_MB": round(traffic_input / 1e6, 3),
        "traffic_gemm_MB": round(traffic_gemm / 1e6, 3),
        "traffic_output_MB": round(traffic_output / 1e6, 3),
        "traffic_total_MB": round(traffic_total / 1e6, 3),
        "traffic_fused_MB": round(traffic_fused / 1e6, 3),
        "traffic_overhead_x": round(traffic_total / traffic_fused, 2),
        "equiv_gflops": round(flops / (ms_full * 1e6), 1),
        "peak_bw_GBs": round(traffic_total / (ms_full * 1e6), 1),
    }
    return result


def main():
    torch.cuda.init()
    prop = torch.cuda.get_device_properties(0)
    print(f"Device: {prop.name}")
    print(f"CUs: {prop.multi_processor_count}, VRAM: {prop.total_memory / 1e9:.1f} GB\n")

    configs = [
        # (name, batch, C, K, H, W)
        ("conv2_x [B=1]",    1,   64,   64, 56, 56),
        ("conv3_x [B=1]",    1,  128,  128, 28, 28),
        ("conv4_x [B=1]",    1,  256,  256, 14, 14),
        ("conv5_x [B=1]",    1,  512,  512,  7,  7),
        ("conv2_x [B=8]",    8,   64,   64, 56, 56),
        ("conv3_x [B=8]",    8,  128,  128, 28, 28),
        ("conv4_x [B=8]",    8,  256,  256, 14, 14),
        ("conv5_x [B=8]",    8,  512,  512,  7,  7),
        ("conv2_x [B=32]",  32,   64,   64, 56, 56),
        ("conv3_x [B=32]",  32,  128,  128, 28, 28),
        ("conv4_x [B=32]",  32,  256,  256, 14, 14),
        ("conv5_x [B=32]",  32,  512,  512,  7,  7),
    ]

    results = []

    # Header
    hdr = (f"{'Config':<22} {'InputT':>8} {'GEMM':>8} {'OutT':>8} "
           f"{'Total':>8} {'MIOpen':>8} {'Ratio':>7} "
           f"{'%InT':>5} {'%GEMM':>6} {'%OutT':>6} "
           f"{'TrafTot':>8} {'TrafFus':>8} {'Ovrhd':>6}")
    print(hdr)
    print("=" * len(hdr))

    for name, batch, C, K, H, W in configs:
        r = profile_config(name, batch, C, K, H, W)
        results.append(r)

        print(f"{r['config']:<22} "
              f"{r['ms_input_transform']:>7.3f}ms "
              f"{r['ms_gemm_approx']:>7.3f}ms "
              f"{r['ms_output_transform']:>7.3f}ms "
              f"{r['ms_full_nova']:>7.3f}ms "
              f"{r['ms_miopen']:>7.3f}ms "
              f"{r['nova_vs_miopen']:>6.2f}x "
              f"{r['pct_input_transform']:>4.0f}% "
              f"{r['pct_gemm']:>5.0f}% "
              f"{r['pct_output_transform']:>5.0f}% "
              f"{r['traffic_total_MB']:>7.2f}MB "
              f"{r['traffic_fused_MB']:>7.2f}MB "
              f"{r['traffic_overhead_x']:>5.1f}x")

    # Summary
    print("\n\n=== BOTTLENECK ANALYSIS ===\n")
    for r in results:
        bottleneck = "GEMM" if r["pct_gemm"] > 50 else (
            "Input Transform" if r["pct_input_transform"] > r["pct_output_transform"]
            else "Output Transform"
        )
        print(f"{r['config']:<22} Bottleneck: {bottleneck:<18} "
              f"Traffic overhead: {r['traffic_overhead_x']:.1f}x vs fused  "
              f"BW utilization: {r['peak_bw_GBs']:.0f} GB/s")

    # Save
    output = {
        "experiment": "NOVA Winograd Per-Stage Profiling",
        "device": prop.name,
        "results": results,
    }
    out_path = "nova_stage_profile.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
