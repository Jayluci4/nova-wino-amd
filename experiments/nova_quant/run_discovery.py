#!/usr/bin/env python3
"""NOVA-Quant: Discover MI300X-optimal rotation matrices via Evolution Strategy.

This is the "NOVA twist" — instead of using random Hadamard rotations (QuIP#)
or SGD-learned dense matrices (SpinQuant), we use Evolution Strategies to
discover structured rotations that are both high-precision AND hardware-native.

Usage:
    # With pre-saved activations (recommended):
    python -m experiments.nova_quant.run_discovery --activations-dir /mnt/data/activations/llama70b_layer0

    # With synthetic data (no model needed):
    python -m experiments.nova_quant.run_discovery --synthetic

    # Full search (more generations, more restarts):
    python -m experiments.nova_quant.run_discovery --activations-dir /mnt/data/activations/llama70b_layer0 \
        --generations 500 --restarts 10 --pop-size 64
"""

import argparse
import json
import os
import sys
import time
from collections import OrderedDict

import torch

from .rotation import NOVARotation, make_random_hadamard_rotation, make_identity_rotation
from .cost_function import (
    evaluate_rotation_quality,
    evaluate_w4a4_quality,
    measure_rotation_latency,
    nova_cost,
    RotationMetrics,
)
from .es_search import NOVAQuantES, StructureAwareES, SearchConfig, SearchResult
from .triton_rotate import benchmark_rotation_methods


DEVICE = 'cuda'


def generate_synthetic_weights(activations: dict, seed: int = 42) -> dict:
    """Generate synthetic weight matrices matching activation proj names.

    Creates random weight matrices with realistic shapes for W4A4 evaluation.
    Each weight matrix W has shape (out_features, in_features) where in_features
    matches the activation's last dim. Shapes are derived from the activation
    dimensions (works for any Llama model size).

    Supports both plain names ('q_proj') and layer-prefixed names ('L0_q_proj').
    """
    torch.manual_seed(seed + 1000)  # Different seed from activations
    weights = {}
    first_key = next(iter(activations))
    hidden_size = activations[first_key].shape[-1]
    # Derive realistic weight shapes from hidden_size
    # Llama uses GQA with 8 KV heads regardless of model size
    kv_dim = hidden_size // 4  # 8B: 1024, 70B: 2048 (approximate)
    intermediate_size = int(hidden_size * 3.5)  # 8B: 14336, 70B: 28672
    weight_shapes = {
        'q_proj': (hidden_size, hidden_size),
        'k_proj': (kv_dim, hidden_size),
        'v_proj': (kv_dim, hidden_size),
        'o_proj': (hidden_size, hidden_size),
        'gate_proj': (intermediate_size, hidden_size),
        'up_proj': (intermediate_size, hidden_size),
    }
    for name in activations:
        in_features = activations[name].shape[-1]
        # Strip layer prefix (e.g., 'L0_q_proj' -> 'q_proj') for shape lookup
        base_name = name.split('_', 1)[1] if name.startswith('L') and '_' in name else name
        if base_name in weight_shapes:
            out_features, _ = weight_shapes[base_name]
        else:
            out_features = in_features
        # Use Kaiming-style init for realistic magnitude
        w = torch.randn(out_features, in_features) * (2.0 / in_features) ** 0.5
        weights[name] = w
    return weights


def load_activations(args) -> tuple:
    """Load activation tensors (and optionally weights) based on CLI args.

    Returns:
        (activations, weights) tuple. weights is None unless target='w4a4'.
    """
    need_weights = getattr(args, 'target', 'int8') == 'w4a4'

    # Multi-layer mode: load from full model or cached multi-layer dir
    if getattr(args, 'multilayer', False):
        save_dir = getattr(args, 'activations_dir', None)
        if save_dir and os.path.isdir(save_dir):
            # Check if it's a multi-layer cache (has L<N>_ prefixed files)
            from ..rotated_fp8.load_activations import load_cached_activations
            activations, weights = load_cached_activations(save_dir, load_weights=True)
            if not weights:
                weights = generate_synthetic_weights(activations, args.seed)
            return activations, weights
        # Load from full model
        from ..rotated_fp8.load_activations import load_multilayer_activations
        model_name = getattr(args, 'model', 'NousResearch/Meta-Llama-3.1-8B')
        layer_indices = getattr(args, 'layer_indices', None)
        save_dir = save_dir or f"/mnt/data/activations/{model_name.split('/')[-1].lower()}_multilayer"
        activations, weights = load_multilayer_activations(
            model_name=model_name, save_dir=save_dir,
            layer_indices=layer_indices)
        return activations, weights

    if args.synthetic:
        print("Using synthetic activations...")
        from ..rotated_fp8.run_experiment import generate_synthetic_activations
        activations = generate_synthetic_activations(args.seed)
        weights = generate_synthetic_weights(activations, args.seed) if need_weights else None
        return activations, weights

    if args.activations_dir and os.path.isdir(args.activations_dir):
        print(f"Loading cached activations from {args.activations_dir}...")
        from ..rotated_fp8.load_activations import load_cached_activations
        if need_weights:
            activations, weights = load_cached_activations(
                args.activations_dir, load_weights=True)
            if not weights:
                print("  No cached weights found, generating synthetic weights...")
                weights = generate_synthetic_weights(activations, args.seed)
            return activations, weights
        return load_cached_activations(args.activations_dir), None

    print("No activation source specified. Use --synthetic or --activations-dir.")
    print("Defaulting to --synthetic.")
    from ..rotated_fp8.run_experiment import generate_synthetic_activations
    activations = generate_synthetic_activations(args.seed)
    weights = generate_synthetic_weights(activations, args.seed) if need_weights else None
    return activations, weights


def evaluate_and_print(name: str, rotation: NOVARotation, activations: dict,
                       x_sample: torch.Tensor, weights: dict = None,
                       target: str = 'int8',
                       weight_group_size: int = 128) -> RotationMetrics:
    """Evaluate a rotation and print results."""
    if target == 'w4a4' and weights is not None:
        metrics = evaluate_w4a4_quality(
            rotation, activations, weights, DEVICE,
            weight_group_size=weight_group_size)
    else:
        metrics = evaluate_rotation_quality(rotation, activations, DEVICE)
    latency = measure_rotation_latency(rotation, x_sample)
    metrics.latency_ms = latency

    print(f"\n  {name}:")
    print(f"    Structure:     {rotation}")
    if target == 'w4a4':
        print(f"    W4A4 GEMM MSE: {metrics.w4a4_gemm_mse:.8f}")
        print(f"    W4A4 GEMM SNR: {metrics.w4a4_gemm_snr_db:.1f} dB")
        print(f"    W4A4 Act MSE:  {metrics.w4a4_act_mse:.8f}")
        print(f"    W4A4 Wgt MSE:  {metrics.w4a4_weight_mse:.8f}")
    else:
        print(f"    FP8  MSE:      {metrics.fp8_mse:.8f}")
        print(f"    FP8  Max Err:  {metrics.fp8_max_error:.6f}")
        print(f"    FP8  SNR:      {metrics.fp8_snr_db:.1f} dB")
        print(f"    INT8 MSE:      {metrics.int8_mse:.8f}")
        print(f"    INT8 Max Err:  {metrics.int8_max_error:.6f}")
        print(f"    INT8 SNR:      {metrics.int8_snr_db:.1f} dB")
    print(f"    Latency:       {latency:.3f} ms")
    return metrics


def print_comparison(baselines: dict, nova_metrics: RotationMetrics, target: str = 'int8'):
    """Print comparison table between baselines and NOVA-discovered rotation."""
    print("\n" + "=" * 90)
    print("NOVA-QUANT DISCOVERY RESULTS")
    print("=" * 90)

    all_methods = {**baselines, 'NOVA-Discovered': nova_metrics}

    if target == 'w4a4':
        print(f"\n{'Method':<25} {'W4A4 GEMM MSE':>14} {'W4A4 SNR':>10} "
              f"{'Act MSE':>12} {'Wgt MSE':>12} {'Latency':>10}")
        print("-" * 90)
        for name, m in all_methods.items():
            marker = " <-- NOVA" if name == 'NOVA-Discovered' else ""
            print(f"{name:<25} {m.w4a4_gemm_mse:>14.8f} {m.w4a4_gemm_snr_db:>9.1f}dB "
                  f"{m.w4a4_act_mse:>12.8f} {m.w4a4_weight_mse:>12.8f} "
                  f"{m.latency_ms:>9.3f}ms{marker}")
    else:
        print(f"\n{'Method':<25} {'INT8 MSE':>12} {'INT8 SNR':>10} {'FP8 MaxErr':>12} "
              f"{'FP8 SNR':>10} {'Latency':>10}")
        print("-" * 90)
        for name, m in all_methods.items():
            marker = " <-- NOVA" if name == 'NOVA-Discovered' else ""
            print(f"{name:<25} {m.int8_mse:>12.8f} {m.int8_snr_db:>9.1f}dB "
                  f"{m.fp8_max_error:>12.6f} {m.fp8_snr_db:>9.1f}dB "
                  f"{m.latency_ms:>9.3f}ms{marker}")

    # Improvement summary
    rh = baselines.get('Random Hadamard')
    if rh:
        print("\n--- Improvement over Random Hadamard ---")
        if target == 'w4a4':
            if nova_metrics.w4a4_gemm_mse > 0 and rh.w4a4_gemm_mse > 0:
                print(f"  W4A4 GEMM MSE reduction: {rh.w4a4_gemm_mse / nova_metrics.w4a4_gemm_mse:.1f}x")
            print(f"  W4A4 SNR improvement:    +{nova_metrics.w4a4_gemm_snr_db - rh.w4a4_gemm_snr_db:.1f} dB")
        else:
            if nova_metrics.int8_mse > 0 and rh.int8_mse > 0:
                print(f"  INT8 MSE reduction:     {rh.int8_mse / nova_metrics.int8_mse:.1f}x")
            if nova_metrics.fp8_max_error > 0 and rh.fp8_max_error > 0:
                print(f"  FP8 max error reduction: {rh.fp8_max_error / nova_metrics.fp8_max_error:.1f}x")
            print(f"  INT8 SNR improvement:   +{nova_metrics.int8_snr_db - rh.int8_snr_db:.1f} dB")
        if rh.latency_ms > 0:
            print(f"  Latency ratio:          {nova_metrics.latency_ms / rh.latency_ms:.2f}x")
    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(description="NOVA-Quant Discovery Sprint")
    parser.add_argument('--activations-dir', type=str, default=None)
    parser.add_argument('--synthetic', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--save-rotation', type=str, default=None,
                        help="Save best rotation to this JSON path")
    parser.add_argument('--multilayer', action='store_true',
                        help="Load activations from multiple layers (joint objective)")
    parser.add_argument('--model', type=str, default="NousResearch/Meta-Llama-3.1-8B",
                        help="Model name for --multilayer loading")
    parser.add_argument('--layer-indices', type=int, nargs='+', default=None,
                        help="Layers to capture for --multilayer (default: 8 evenly spaced)")

    # ES parameters
    parser.add_argument('--pop-size', type=int, default=32)
    parser.add_argument('--generations', type=int, default=200)
    parser.add_argument('--restarts', type=int, default=5)
    parser.add_argument('--two-phase', action='store_true',
                        help="Use two-phase search (structure then signs)")
    parser.add_argument('--target', choices=['int8', 'fp8', 'w4a4'], default='int8',
                        help="Optimization target format")
    parser.add_argument('--weight-group-size', type=int, default=128,
                        help="Group size for W4A4 per-group weight quantization")
    parser.add_argument('--alpha', type=float, default=1.0,
                        help="Weight for quantization error")
    parser.add_argument('--beta', type=float, default=0.1,
                        help="Weight for latency (ms)")

    # Benchmark mode
    parser.add_argument('--benchmark-triton', action='store_true',
                        help="Run Triton kernel benchmarks and exit")

    args = parser.parse_args()
    torch.manual_seed(args.seed)

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Target format: {args.target}")

    # Optional Triton benchmark
    if args.benchmark_triton:
        print("\n--- Triton Kernel Benchmarks ---")
        for bs in [16, 32, 64, 128]:
            r = benchmark_rotation_methods(dim=8192, batch=200, block_size=bs)
            print(f"  block_size={bs:>4}: PyTorch={r['pytorch_ms']:.3f}ms, "
                  f"Triton={r['triton_ms']:.3f}ms, "
                  f"Speedup={r['speedup']:.2f}x")
        return

    # Load activations (and weights for W4A4)
    activations, weights = load_activations(args)
    first_key = next(iter(activations))
    dim = activations[first_key].shape[-1]
    print(f"\nActivation dim: {dim}")
    print(f"Layers: {list(activations.keys())}")
    if weights:
        print(f"Weight matrices: {list(weights.keys())}")

    # Create sample tensor for latency measurement
    x_sample = activations[first_key].to(DEVICE, dtype=torch.float32)

    eval_kwargs = dict(target=args.target, weight_group_size=args.weight_group_size)
    if args.target == 'w4a4':
        eval_kwargs['weights'] = weights

    # --- Evaluate baselines ---
    print("\n" + "=" * 60)
    print("BASELINE EVALUATION")
    print("=" * 60)

    baselines = OrderedDict()

    # 1. No rotation (identity)
    identity = make_identity_rotation(dim)
    baselines['No Rotation'] = evaluate_and_print(
        'No Rotation (identity)', identity, activations, x_sample, **eval_kwargs)

    # 2. Random Hadamard (QuIP# / QuaRot baseline)
    random_had = make_random_hadamard_rotation(dim, seed=args.seed)
    baselines['Random Hadamard'] = evaluate_and_print(
        'Random Hadamard (QuaRot baseline)', random_had, activations, x_sample,
        **eval_kwargs)

    # 3. Small-block Hadamard (fast but less effective)
    from .rotation import RotationStage
    small_block_signs = (torch.randint(0, 2, (dim,)) * 2 - 1).float()
    small_block = NOVARotation(dim=dim, stages=[
        RotationStage(block_size=128, signs=small_block_signs)
    ])
    baselines['Block-128 Hadamard'] = evaluate_and_print(
        'Block-128 Hadamard (fast baseline)', small_block, activations, x_sample,
        **eval_kwargs)

    # --- NOVA-Quant Discovery ---
    print("\n" + "=" * 60)
    print("NOVA-QUANT EVOLUTION STRATEGY SEARCH")
    print("=" * 60)

    search_config = SearchConfig(
        dim=dim,
        pop_size=args.pop_size,
        n_generations=args.generations,
        n_restarts=args.restarts,
        seed=args.seed,
    )

    def cost_fn(rotation):
        return nova_cost(
            rotation, activations, x_sample,
            alpha=args.alpha, beta=args.beta,
            target=args.target, device=DEVICE,
            weights=weights,
            weight_group_size=args.weight_group_size,
        )

    def progress_callback(restart, gen, best_cost, best_rot):
        print(f"  [R{restart+1} G{gen:>3d}] cost={best_cost:.6f} "
              f"structure={best_rot}")

    t0 = time.time()

    if args.two_phase:
        es = StructureAwareES(search_config)
        result = es.run_two_phase(cost_fn, callback=progress_callback)
    else:
        es = NOVAQuantES(search_config)
        result = es.run(cost_fn, callback=progress_callback)

    elapsed = time.time() - t0

    # --- Evaluate best found rotation ---
    print(f"\nSearch completed in {elapsed:.1f}s ({result.total_evaluations} evaluations)")
    nova_metrics = evaluate_and_print(
        'NOVA-Discovered', result.best_rotation, activations, x_sample,
        **eval_kwargs)

    # --- Comparison ---
    print_comparison(baselines, nova_metrics, target=args.target)

    # --- Save results ---
    output_path = args.output or os.path.join(os.path.dirname(__file__), "discovery_results.json")
    save_data = {
        'config': {
            'seed': args.seed,
            'pop_size': args.pop_size,
            'generations': args.generations,
            'restarts': args.restarts,
            'target': args.target,
            'alpha': args.alpha,
            'beta': args.beta,
            'two_phase': args.two_phase,
            'weight_group_size': args.weight_group_size,
            'device': torch.cuda.get_device_name(0),
            'pytorch': torch.__version__,
        },
        'baselines': {
            name: {
                'int8_mse': m.int8_mse,
                'int8_snr_db': m.int8_snr_db,
                'fp8_mse': m.fp8_mse,
                'fp8_max_error': m.fp8_max_error,
                'fp8_snr_db': m.fp8_snr_db,
                'w4a4_gemm_mse': m.w4a4_gemm_mse,
                'w4a4_gemm_snr_db': m.w4a4_gemm_snr_db,
                'w4a4_act_mse': m.w4a4_act_mse,
                'w4a4_weight_mse': m.w4a4_weight_mse,
                'latency_ms': m.latency_ms,
            }
            for name, m in baselines.items()
        },
        'nova_result': {
            'int8_mse': nova_metrics.int8_mse,
            'int8_snr_db': nova_metrics.int8_snr_db,
            'fp8_mse': nova_metrics.fp8_mse,
            'fp8_max_error': nova_metrics.fp8_max_error,
            'fp8_snr_db': nova_metrics.fp8_snr_db,
            'w4a4_gemm_mse': nova_metrics.w4a4_gemm_mse,
            'w4a4_gemm_snr_db': nova_metrics.w4a4_gemm_snr_db,
            'w4a4_act_mse': nova_metrics.w4a4_act_mse,
            'w4a4_weight_mse': nova_metrics.w4a4_weight_mse,
            'latency_ms': nova_metrics.latency_ms,
            'rotation': result.best_rotation.to_dict(),
            'total_evaluations': result.total_evaluations,
            'search_time_s': result.total_time_s,
        },
        'search_history': result.history,
    }

    with open(output_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Save rotation if requested
    rotation_path = args.save_rotation or os.path.join(
        os.path.dirname(__file__), "best_rotation.json")
    result.best_rotation.save(rotation_path)
    print(f"Best rotation saved to {rotation_path}")


if __name__ == "__main__":
    main()
