#!/usr/bin/env python3
"""Main experiment: compare naive, per-channel, and rotated FP8 quantization
on real Llama-3.1-70B layer 0 activations.

Usage:
    # With pre-saved activations:
    python -m experiments.rotated_fp8.run_experiment --activations-dir /mnt/data/activations/llama70b_layer0

    # Fresh extraction (downloads model shard if needed):
    python -m experiments.rotated_fp8.run_experiment --extract

    # With synthetic data (no model needed):
    python -m experiments.rotated_fp8.run_experiment --synthetic
"""

import argparse
import json
import os
import sys
import time
from collections import OrderedDict

import torch
from scipy import stats as scipy_stats

from .hadamard import generate_random_signs, random_hadamard_transform
from .fp8_quantize import (
    quantize_fp8_naive,
    quantize_fp8_per_channel,
    quantize_fp8_rotated,
    dequantize_fp8,
    dequantize_fp8_rotated,
    quantize_int8_naive,
    quantize_int8_rotated,
    dequantize_int8,
    dequantize_int8_rotated,
    compute_error_metrics,
)


DEVICE = 'cuda'


def analyze_outlier_statistics(activations: dict) -> dict:
    """Analyze outlier characteristics of activation tensors.

    Reports per-channel max/mean ratio, kurtosis, and dynamic range.
    """
    stats = {}
    for name, x in activations.items():
        x = x.float()
        # Flatten all dims except last (channel dim) for per-channel stats
        x_flat = x.reshape(-1, x.shape[-1])
        channel_max = x_flat.abs().amax(dim=0)  # (hidden,)
        channel_mean = x_flat.abs().mean(dim=0)

        # Per-channel max/mean ratio: measures outlier severity
        ratio = channel_max / (channel_mean + 1e-10)

        flat = x.flatten()
        kurtosis = scipy_stats.kurtosis(flat.cpu().numpy(), fisher=True)  # excess kurtosis

        stats[name] = {
            'shape': list(x.shape),
            'global_max': x.abs().max().item(),
            'global_mean': x.abs().mean().item(),
            'max_mean_ratio': ratio.max().item(),
            'mean_max_mean_ratio': ratio.mean().item(),
            'kurtosis': float(kurtosis),
            'dynamic_range_db': float(20 * torch.log10(x.abs().max() / (x.abs().min() + 1e-10)).item()),
        }
    return stats


def run_quantization_comparison(x: torch.Tensor, name: str, signs: torch.Tensor) -> dict:
    """Run all quantization schemes on a single activation tensor.

    Compares FP8 (floating-point) and INT8 (fixed-point) quantization, with and
    without Hadamard rotation. FP8 benefits from rotation via max error reduction;
    INT8 benefits via dramatic MSE reduction.

    Args:
        x: Activation tensor (float32, any device).
        name: Layer name for logging.
        signs: Random sign vector for Hadamard rotation.

    Returns:
        Dict with metrics for each scheme.
    """
    x_gpu = x.to(DEVICE, dtype=torch.float32)
    signs_gpu = signs.to(DEVICE).float()
    results = {}

    # --- FP8 schemes ---
    # 1. Naive FP8 (per-tensor)
    x_fp8, scale = quantize_fp8_naive(x_gpu)
    recon = dequantize_fp8(x_fp8, scale)
    results['fp8_naive'] = compute_error_metrics(x_gpu, recon)

    # 2. Per-channel FP8
    x_fp8_ch, scales_ch = quantize_fp8_per_channel(x_gpu)
    recon_ch = dequantize_fp8(x_fp8_ch, scales_ch)
    results['fp8_per_channel'] = compute_error_metrics(x_gpu, recon_ch)

    # 3. Rotated FP8
    x_fp8_rot, scale_rot, _ = quantize_fp8_rotated(x_gpu, signs_gpu)
    recon_rot = dequantize_fp8_rotated(x_fp8_rot, scale_rot, signs_gpu)
    results['fp8_rotated'] = compute_error_metrics(x_gpu, recon_rot)

    # --- INT8 schemes ---
    # 4. Naive INT8 (per-tensor)
    x_int8, scale_i = quantize_int8_naive(x_gpu)
    recon_i = dequantize_int8(x_int8, scale_i)
    results['int8_naive'] = compute_error_metrics(x_gpu, recon_i)

    # 5. Rotated INT8
    x_int8_r, scale_ir, _ = quantize_int8_rotated(x_gpu, signs_gpu)
    recon_ir = dequantize_int8_rotated(x_int8_r, scale_ir, signs_gpu)
    results['int8_rotated'] = compute_error_metrics(x_gpu, recon_ir)

    return results


def generate_synthetic_activations(seed: int = 42) -> dict:
    """Generate synthetic activations mimicking LLM outlier patterns.

    Simulates the characteristic outlier distribution seen in transformer
    linear layer inputs: most channels have small activations, but ~1-2%
    of channels consistently produce values 50-200x larger.
    """
    torch.manual_seed(seed)
    activations = {}

    layer_names = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj']
    seq_len = 512
    hidden = 8192

    for name in layer_names:
        # Base: small activations
        x = torch.randn(1, seq_len, hidden) * 0.1

        # Outlier channels (~1.5% of channels)
        n_outliers = int(hidden * 0.015)
        outlier_idx = torch.randperm(hidden)[:n_outliers]
        outlier_magnitude = 20.0 + torch.rand(n_outliers) * 80.0  # 20x-100x
        x[:, :, outlier_idx] *= outlier_magnitude.unsqueeze(0)

        activations[name] = x
    return activations


def print_results_table(all_results: dict, outlier_stats: dict):
    """Print a formatted comparison table."""
    print("\n" + "=" * 100)
    print("ROTATED FP8 QUANTIZATION — EXPERIMENT RESULTS")
    print("=" * 100)

    # Outlier statistics
    print("\n--- Outlier Statistics ---")
    print(f"{'Layer':<12} {'Shape':<20} {'Max/Mean':>10} {'Kurtosis':>10} {'Dynamic Range':>14}")
    print("-" * 70)
    for name, s in outlier_stats.items():
        short = name.split('.')[-1] if '.' in name else name
        print(f"{short:<12} {str(s['shape']):<20} {s['max_mean_ratio']:>10.1f} "
              f"{s['kurtosis']:>10.1f} {s['dynamic_range_db']:>11.1f} dB")

    # Quantization comparison
    print("\n--- Quantization Error Comparison ---")
    print(f"{'Layer':<12} {'Scheme':<16} {'MSE':>12} {'Max Error':>12} {'SNR (dB)':>10} {'Rel Error':>10}")
    print("-" * 80)

    schemes = ['fp8_naive', 'fp8_per_channel', 'fp8_rotated', 'int8_naive', 'int8_rotated']

    fp8_max_err_improvements = []
    int8_mse_improvements = []

    for name, results in all_results.items():
        short = name.split('.')[-1] if '.' in name else name
        for scheme in schemes:
            m = results[scheme]
            print(f"{short:<12} {scheme:<16} {m['mse']:>12.6f} {m['max_error']:>12.4f} "
                  f"{m['snr_db']:>10.1f} {m['relative_error']:>10.6f}")
        print()

        # Track FP8 max error improvement
        if results['fp8_naive']['max_error'] > 0 and results['fp8_rotated']['max_error'] > 0:
            fp8_max_err_improvements.append(
                results['fp8_naive']['max_error'] / results['fp8_rotated']['max_error'])
        # Track INT8 MSE improvement
        if results['int8_naive']['mse'] > 0 and results['int8_rotated']['mse'] > 0:
            int8_mse_improvements.append(
                results['int8_naive']['mse'] / results['int8_rotated']['mse'])

    # Summary
    print("-" * 80)
    print("SUMMARY OF IMPROVEMENTS (Rotated vs Naive):")
    if fp8_max_err_improvements:
        avg = sum(fp8_max_err_improvements) / len(fp8_max_err_improvements)
        print(f"  FP8 max error reduction:  {avg:.1f}x (rotation compresses dynamic range)")
    if int8_mse_improvements:
        avg = sum(int8_mse_improvements) / len(int8_mse_improvements)
        print(f"  INT8 MSE reduction:       {avg:.1f}x (rotation eliminates grid waste)")
    print()
    print("Note: FP8 is a floating-point format with ~6.25% relative precision.")
    print("Rotation's MSE benefit is most dramatic for INT8's uniform quantization grid,")
    print("where outliers waste the entire dynamic range of non-outlier channels.")
    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(description="Rotated FP8 Quantization Experiment")
    parser.add_argument('--activations-dir', type=str, default=None,
                        help="Directory with pre-saved .pt activation files")
    parser.add_argument('--extract', action='store_true',
                        help="Extract activations from model (downloads shard if needed)")
    parser.add_argument('--synthetic', action='store_true',
                        help="Use synthetic data (no model needed)")
    parser.add_argument('--output', type=str, default=None,
                        help="Save results JSON to this path")
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")

    # Load activations
    if args.synthetic:
        print("\nUsing synthetic activations (LLM-like outlier patterns)...")
        activations = generate_synthetic_activations(args.seed)
    elif args.activations_dir and os.path.isdir(args.activations_dir):
        print(f"\nLoading cached activations from {args.activations_dir}...")
        from .load_activations import load_cached_activations
        activations = load_cached_activations(args.activations_dir)
    elif args.extract:
        print("\nExtracting activations from model...")
        from .load_activations import load_activations
        save_dir = "/mnt/data/activations/llama70b_layer0"
        activations = load_activations(save_dir=save_dir)
        # Convert to short names
        activations = {k.split('.')[-1]: v for k, v in activations.items()}
    else:
        print("No activation source specified. Use --synthetic, --extract, or --activations-dir.")
        print("Defaulting to --synthetic.")
        activations = generate_synthetic_activations(args.seed)

    # Analyze outlier statistics
    print("\nAnalyzing outlier statistics...")
    outlier_stats = analyze_outlier_statistics(activations)

    # Generate random signs for Hadamard rotation
    first_key = next(iter(activations))
    dim = activations[first_key].shape[-1]
    signs = generate_random_signs(dim, device=DEVICE, dtype=torch.float32, seed=args.seed)
    print(f"Hadamard rotation dimension: {dim}")

    # Run quantization comparison for each layer
    print("\nRunning quantization comparison...")
    all_results = OrderedDict()
    t0 = time.time()
    for name, x in activations.items():
        short = name.split('.')[-1] if '.' in name else name
        results = run_quantization_comparison(x, short, signs)
        all_results[name] = results
    elapsed = time.time() - t0
    print(f"Completed in {elapsed:.2f}s")

    # Print results
    print_results_table(all_results, outlier_stats)

    # Save results
    output_path = args.output or os.path.join(
        os.path.dirname(__file__), "results.json"
    )
    save_data = {
        'outlier_stats': outlier_stats,
        'quantization_results': {k: v for k, v in all_results.items()},
        'config': {
            'seed': args.seed,
            'device': torch.cuda.get_device_name(0),
            'pytorch_version': torch.__version__,
            'fp8_format': 'float8_e4m3fnuz',
            'fp8_max': 240.0,
            'hadamard_dim': dim,
            'synthetic': args.synthetic or (not args.extract and not args.activations_dir),
        }
    }
    with open(output_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
