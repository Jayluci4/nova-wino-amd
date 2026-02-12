"""Per-layer W4A4 analysis: universality and transferability of rotation block size.

Questions answered:
1. UNIVERSALITY: Is block-256 optimal for all layers, or do later (more Gaussian) layers
   prefer larger blocks?
2. TRANSFERABILITY: Does the layer-0 NOVA rotation (optimized signs) help on layer 40?
   Or are random signs equally good when transferred across layers?

Approach:
- Load full Llama-3.1-70B, run a forward pass with hooks to capture activations
  at representative layers [0, 10, 20, 40, 60, 79].
- For each captured layer, evaluate W4A4 GEMM error across a spectrum of block sizes
  (identity, 128, 256, 512, 1024, 4096, 8192) with random signs.
- Also evaluate the layer-0 NOVA rotation (block-256, optimized signs) on every layer.
- Compare NOVA-layer-0 vs random-block-256 at each layer → transferability of signs.
- Compare optimal block size across layers → universality of structure.

Usage:
    python -m experiments.nova_quant.perlayer_analysis \
        --model NousResearch/Meta-Llama-3.1-70B \
        --rotation /mnt/data/activations/llama70b_layer0/w4a4_best_rotation.json
"""

import torch
import torch.nn as nn
import json
import time
import argparse
import gc
import math
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from .rotation import (
    NOVARotation, RotationStage, make_random_hadamard_rotation,
    make_identity_rotation, block_hadamard_transform,
)
from ..rotated_fp8.fp8_quantize import (
    quantize_int4_naive, dequantize_int4,
    quantize_int4_per_group, dequantize_int4_per_group,
)


# Projections to analyze (all have in_features = hidden_dim = 8192)
PROJ_NAMES = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj']

# Block sizes to sweep
BLOCK_SIZES = [128, 256, 512, 1024, 4096, 8192]

# Layers to sample
DEFAULT_LAYERS = [0, 10, 20, 40, 60, 79]


def make_block_rotation(dim: int, block_size: int, seed: int = 42) -> NOVARotation:
    """Create a single-stage block-diagonal Hadamard with random signs."""
    gen = torch.Generator(device='cpu').manual_seed(seed)
    signs = torch.randint(0, 2, (dim,), generator=gen, device='cpu') * 2 - 1
    stage = RotationStage(block_size=block_size, signs=signs.float())
    return NOVARotation(dim=dim, stages=[stage])


def compute_w4a4_gemm_error(
    x: torch.Tensor,       # (tokens, in_features) on GPU
    w: torch.Tensor,       # (out_features, in_features) on GPU
    rotation: NOVARotation,
    weight_group_size: int = 128,
) -> Dict[str, float]:
    """Compute W4A4 GEMM output error for a single (activation, weight) pair."""
    with torch.no_grad():
        # Exact GEMM
        y_exact = x @ w.t()
        signal_power = (y_exact ** 2).mean().item()

        # Rotate
        has_stages = rotation.n_stages > 0
        x_rot = rotation.forward(x) if has_stages else x
        w_rot = rotation.forward(w) if has_stages else w

        # Quantize activations (per-tensor INT4)
        x_q, x_s = quantize_int4_naive(x_rot)
        x_deq = dequantize_int4(x_q, x_s)

        # Quantize weights (per-group INT4)
        in_f = w_rot.shape[1]
        if in_f % weight_group_size == 0:
            w_q, w_s = quantize_int4_per_group(w_rot, weight_group_size)
            w_deq = dequantize_int4_per_group(w_q, w_s, weight_group_size)
        else:
            w_q, w_s = quantize_int4_naive(w_rot)
            w_deq = dequantize_int4(w_q, w_s)

        # Quantized GEMM
        y_q = x_deq @ w_deq.t()
        diff = y_exact - y_q
        gemm_mse = (diff ** 2).mean().item()

        # Component errors
        x_recon = rotation.inverse(x_deq) if has_stages else x_deq
        act_mse = ((x - x_recon) ** 2).mean().item()

        w_recon = rotation.inverse(w_deq) if has_stages else w_deq
        wgt_mse = ((w - w_recon) ** 2).mean().item()

        snr = 10 * math.log10(signal_power / gemm_mse) if gemm_mse > 0 else float('inf')

    return {
        'gemm_mse': gemm_mse,
        'gemm_snr_db': snr,
        'act_mse': act_mse,
        'wgt_mse': wgt_mse,
    }


def capture_activations(
    model, tokenizer, text: str, layer_indices: List[int],
    max_tokens: int = 2048, device: str = 'cuda',
) -> Dict[int, Dict[str, Tuple[torch.Tensor, torch.Tensor]]]:
    """Run one forward pass and capture (activation, weight) pairs at target layers.

    Returns: {layer_idx: {proj_name: (activation, weight)}}
    Activations are the INPUT to each projection (residual stream or post-attn).
    """
    captured = defaultdict(dict)
    hooks = []

    for layer_idx in layer_indices:
        layer = model.model.layers[layer_idx]

        for proj_name in PROJ_NAMES:
            # Navigate to the projection module
            if proj_name in ('q_proj', 'k_proj', 'v_proj', 'o_proj'):
                proj = getattr(layer.self_attn, proj_name)
            else:
                proj = getattr(layer.mlp, proj_name)

            # Capture input activation via forward pre-hook
            li, pn = layer_idx, proj_name  # closure capture

            def _make_hook(li, pn):
                def hook(mod, args):
                    x = args[0].detach().float().cpu()
                    # Flatten to (tokens, features)
                    x = x.view(-1, x.shape[-1])
                    captured[li][pn] = x
                return hook

            h = proj.register_forward_pre_hook(_make_hook(li, pn))
            hooks.append(h)

    # Run forward pass
    enc = tokenizer(text[:50000], return_tensors='pt', truncation=True,
                    max_length=max_tokens)
    input_ids = enc.input_ids.to(device)

    with torch.no_grad():
        model(input_ids)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Pair activations with weights
    result = {}
    for layer_idx in layer_indices:
        result[layer_idx] = {}
        layer = model.model.layers[layer_idx]
        for proj_name in PROJ_NAMES:
            if proj_name not in captured[layer_idx]:
                continue
            act = captured[layer_idx][proj_name]
            if proj_name in ('q_proj', 'k_proj', 'v_proj', 'o_proj'):
                wgt = getattr(layer.self_attn, proj_name).weight.data.float().cpu()
            else:
                wgt = getattr(layer.mlp, proj_name).weight.data.float().cpu()
            result[layer_idx][proj_name] = (act, wgt)

    return result


def analyze_layer(
    layer_data: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    nova_rotation: NOVARotation,
    dim: int,
    weight_group_size: int = 128,
    device: str = 'cuda',
) -> Dict:
    """Analyze a single layer: sweep block sizes + test NOVA transferability."""
    results = {}

    # --- Identity (no rotation) ---
    identity = make_identity_rotation(dim)
    errs = []
    for pn, (act, wgt) in layer_data.items():
        x = act.to(device, dtype=torch.float32)
        w = wgt.to(device, dtype=torch.float32)
        e = compute_w4a4_gemm_error(x, w, identity, weight_group_size)
        errs.append(e)
    results['identity'] = {
        'gemm_mse': sum(e['gemm_mse'] for e in errs) / len(errs),
        'gemm_snr_db': sum(e['gemm_snr_db'] for e in errs) / len(errs),
        'act_mse': sum(e['act_mse'] for e in errs) / len(errs),
        'wgt_mse': sum(e['wgt_mse'] for e in errs) / len(errs),
    }

    # --- Block size sweep (random signs) ---
    for bs in BLOCK_SIZES:
        rot = make_block_rotation(dim, bs, seed=42)
        errs = []
        for pn, (act, wgt) in layer_data.items():
            x = act.to(device, dtype=torch.float32)
            w = wgt.to(device, dtype=torch.float32)
            e = compute_w4a4_gemm_error(x, w, rot, weight_group_size)
            errs.append(e)
        results[f'block_{bs}_random'] = {
            'gemm_mse': sum(e['gemm_mse'] for e in errs) / len(errs),
            'gemm_snr_db': sum(e['gemm_snr_db'] for e in errs) / len(errs),
            'act_mse': sum(e['act_mse'] for e in errs) / len(errs),
            'wgt_mse': sum(e['wgt_mse'] for e in errs) / len(errs),
        }

    # --- NOVA rotation (layer-0 optimized signs, block-256) ---
    errs = []
    for pn, (act, wgt) in layer_data.items():
        x = act.to(device, dtype=torch.float32)
        w = wgt.to(device, dtype=torch.float32)
        e = compute_w4a4_gemm_error(x, w, nova_rotation, weight_group_size)
        errs.append(e)
    results['nova_layer0'] = {
        'gemm_mse': sum(e['gemm_mse'] for e in errs) / len(errs),
        'gemm_snr_db': sum(e['gemm_snr_db'] for e in errs) / len(errs),
        'act_mse': sum(e['act_mse'] for e in errs) / len(errs),
        'wgt_mse': sum(e['wgt_mse'] for e in errs) / len(errs),
    }

    # --- Activation statistics (kurtosis, max/mean) ---
    stats = {}
    for pn, (act, _) in layer_data.items():
        x = act.float()
        mean = x.mean().item()
        std = x.std().item()
        kurt = ((x - mean) ** 4).mean().item() / (std ** 4) if std > 0 else 0
        max_mean = (x.abs().max().item() / x.abs().mean().item()) if x.abs().mean().item() > 0 else 0
        stats[pn] = {'kurtosis': kurt, 'max_over_mean': max_mean}
    results['activation_stats'] = stats

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='NousResearch/Meta-Llama-3.1-70B')
    parser.add_argument('--cache-dir', default='/mnt/data/huggingface')
    parser.add_argument('--rotation',
                        default='/mnt/data/activations/llama70b_layer0/w4a4_best_rotation.json')
    parser.add_argument('--layers', type=int, nargs='+', default=DEFAULT_LAYERS)
    parser.add_argument('--max-tokens', type=int, default=2048)
    parser.add_argument('--weight-group-size', type=int, default=128)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    # Load NOVA rotation
    print(f"Loading NOVA rotation: {args.rotation}")
    nova_rot = NOVARotation.load(args.rotation)
    dim = nova_rot.dim
    print(f"  {nova_rot}")

    # Load model
    print(f"\nLoading {args.model} ...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, cache_dir=args.cache_dir,
        dtype=torch.float16, device_map='auto',
    )
    print(f"  Loaded in {time.time()-t0:.0f}s")

    # Load dataset text
    print("\nLoading WikiText-2 ...")
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    text = "\n\n".join(item['text'] for item in ds if item['text'].strip())

    # Capture activations
    print(f"\nCapturing activations at layers {args.layers} ...")
    t0 = time.time()
    layer_data = capture_activations(
        model, tokenizer, text, args.layers,
        max_tokens=args.max_tokens,
    )
    print(f"  Captured in {time.time()-t0:.0f}s")

    for li in args.layers:
        for pn in PROJ_NAMES:
            if pn in layer_data[li]:
                act, wgt = layer_data[li][pn]
                print(f"  Layer {li:2d} {pn:10s}: act {tuple(act.shape)}, wgt {tuple(wgt.shape)}")

    # Free model to reclaim GPU memory for analysis
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"\nModel freed. GPU mem: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    # Per-layer analysis
    all_results = {}
    for li in args.layers:
        print(f"\n{'='*60}")
        print(f"  LAYER {li}")
        print(f"{'='*60}")
        t0 = time.time()
        all_results[li] = analyze_layer(
            layer_data[li], nova_rot, dim,
            weight_group_size=args.weight_group_size,
        )
        dt = time.time() - t0
        print(f"  Analyzed in {dt:.0f}s")

        # Print compact results
        r = all_results[li]
        print(f"\n  {'Method':<25} {'GEMM MSE':>12} {'SNR(dB)':>10} {'Act MSE':>12} {'Wgt MSE':>12}")
        print(f"  {'-'*71}")
        print(f"  {'Identity':<25} {r['identity']['gemm_mse']:>12.6f} {r['identity']['gemm_snr_db']:>10.1f} {r['identity']['act_mse']:>12.2e} {r['identity']['wgt_mse']:>12.2e}")
        for bs in BLOCK_SIZES:
            key = f'block_{bs}_random'
            print(f"  {f'Block-{bs} (random)':<25} {r[key]['gemm_mse']:>12.6f} {r[key]['gemm_snr_db']:>10.1f} {r[key]['act_mse']:>12.2e} {r[key]['wgt_mse']:>12.2e}")
        print(f"  {'NOVA layer-0 (block-256)':<25} {r['nova_layer0']['gemm_mse']:>12.6f} {r['nova_layer0']['gemm_snr_db']:>10.1f} {r['nova_layer0']['act_mse']:>12.2e} {r['nova_layer0']['wgt_mse']:>12.2e}")

        # Activation stats
        print(f"\n  Activation statistics:")
        stats = r['activation_stats']
        for pn in PROJ_NAMES:
            if pn in stats:
                s = stats[pn]
                print(f"    {pn:10s}: kurtosis={s['kurtosis']:.0f}, max/mean={s['max_over_mean']:.1f}")

    # --- Summary tables ---
    print(f"\n\n{'='*80}")
    print("UNIVERSALITY: Optimal block size per layer")
    print(f"{'='*80}")
    print(f"{'Layer':>6} {'Best Block':>12} {'Best MSE':>12} {'Identity MSE':>14} {'Improvement':>12}")
    print(f"{'-'*56}")
    for li in args.layers:
        r = all_results[li]
        id_mse = r['identity']['gemm_mse']
        best_key, best_mse = 'identity', id_mse
        for bs in BLOCK_SIZES:
            key = f'block_{bs}_random'
            if r[key]['gemm_mse'] < best_mse:
                best_mse = r[key]['gemm_mse']
                best_key = f'{bs}'
        if r['nova_layer0']['gemm_mse'] < best_mse:
            best_mse = r['nova_layer0']['gemm_mse']
            best_key = 'NOVA-256'
        improve = id_mse / best_mse if best_mse > 0 else float('inf')
        print(f"{li:>6} {best_key:>12} {best_mse:>12.6f} {id_mse:>14.6f} {improve:>11.1f}x")

    print(f"\n{'='*80}")
    print("TRANSFERABILITY: NOVA layer-0 signs vs random signs (both block-256)")
    print(f"{'='*80}")
    print(f"{'Layer':>6} {'Random-256 MSE':>16} {'NOVA-256 MSE':>16} {'Sign Benefit':>14}")
    print(f"{'-'*52}")
    for li in args.layers:
        r = all_results[li]
        rand_mse = r['block_256_random']['gemm_mse']
        nova_mse = r['nova_layer0']['gemm_mse']
        benefit = rand_mse / nova_mse if nova_mse > 0 else float('inf')
        print(f"{li:>6} {rand_mse:>16.6f} {nova_mse:>16.6f} {benefit:>13.2f}x")

    print(f"\n{'='*80}")
    print("ACTIVATION DISTRIBUTION: How kurtosis changes across depth")
    print(f"{'='*80}")
    print(f"{'Layer':>6} {'Mean Kurtosis':>16} {'Mean Max/Mean':>16}")
    print(f"{'-'*38}")
    for li in args.layers:
        stats = all_results[li]['activation_stats']
        avg_kurt = sum(s['kurtosis'] for s in stats.values()) / len(stats)
        avg_mm = sum(s['max_over_mean'] for s in stats.values()) / len(stats)
        print(f"{li:>6} {avg_kurt:>16.0f} {avg_mm:>16.1f}")

    # Save
    if args.output:
        # Convert int keys to strings for JSON
        out = {str(k): v for k, v in all_results.items()}
        out['_config'] = {
            'model': args.model,
            'layers': args.layers,
            'block_sizes': BLOCK_SIZES,
            'max_tokens': args.max_tokens,
            'weight_group_size': args.weight_group_size,
            'rotation_file': args.rotation,
        }
        with open(args.output, 'w') as f:
            json.dump(out, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == '__main__':
    main()
