"""Diagnostic: isolate whether the group-size anomaly is in quantization or the pipeline.

Tests two hypotheses:
1. BUG hypothesis: quantize_int4_per_group has a bug for certain group sizes
2. REAL hypothesis: Hadamard-rotated weights genuinely quantize worse with small groups

Measures weight-only MSE and GEMM output error for a single layer, controlling all variables.
"""

import torch
import time
import sys

sys.path.insert(0, '/root/nova-wino-amd')

from experiments.nova_quant.rotation import NOVARotation, RotationStage
from experiments.rotated_fp8.fp8_quantize import (
    quantize_int4_per_group, dequantize_int4_per_group,
    quantize_int4_naive, dequantize_int4,
)


def measure_quant_error(w: torch.Tensor, group_size: int) -> dict:
    """Measure INT4 per-group quantization error for a weight matrix."""
    w_q, w_s = quantize_int4_per_group(w, group_size)
    w_deq = dequantize_int4_per_group(w_q, w_s, group_size)

    diff = (w.float() - w_deq.float())
    mse = (diff ** 2).mean().item()
    max_err = diff.abs().max().item()
    # Check values are actually in INT4 range
    assert w_q.min() >= -8 and w_q.max() <= 7, f"INT4 range violation: [{w_q.min()}, {w_q.max()}]"
    # Check roundtrip shape
    assert w_deq.shape == w.shape, f"Shape mismatch: {w_deq.shape} vs {w.shape}"

    return {'mse': mse, 'max_err': max_err, 'n_scales': w_s.numel()}


def measure_gemm_error(x: torch.Tensor, w: torch.Tensor, group_size: int) -> dict:
    """Measure error in Y = X @ W^T where W is INT4-quantized."""
    y_exact = x.float() @ w.float().T

    w_q, w_s = quantize_int4_per_group(w, group_size)
    w_deq = dequantize_int4_per_group(w_q, w_s, group_size)
    y_quant = x.float() @ w_deq.float().T

    diff = y_exact - y_quant
    mse = (diff ** 2).mean().item()
    snr = 10 * torch.log10((y_exact ** 2).mean() / (diff ** 2).mean()).item()

    return {'gemm_mse': mse, 'gemm_snr_db': snr}


def main():
    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dim = 8192

    print("=" * 70)
    print("DIAGNOSTIC: Weight group-size vs quantization error")
    print("=" * 70)

    # --- Test 1: Random Gaussian weights (no structure, no rotation) ---
    print("\n--- Test 1: Random Gaussian weights (iid, no structure) ---")
    w_gauss = torch.randn(dim, dim, device=device) * 0.01
    x_gauss = torch.randn(64, dim, device=device) * 0.1

    print(f"  {'g':>6}  {'W-MSE':>12}  {'Max-Err':>10}  {'GEMM-MSE':>12}  {'GEMM-SNR':>10}")
    for g in [32, 64, 128, 256]:
        qe = measure_quant_error(w_gauss, g)
        ge = measure_gemm_error(x_gauss, w_gauss, g)
        print(f"  {g:>6}  {qe['mse']:>12.6e}  {qe['max_err']:>10.6f}  {ge['gemm_mse']:>12.6e}  {ge['gemm_snr_db']:>10.2f}")

    # --- Test 2: Weights with outlier channels (simulating real LLM weights) ---
    print("\n--- Test 2: Outlier-heavy weights (LLM-like, no rotation) ---")
    w_outlier = torch.randn(dim, dim, device=device) * 0.01
    # Add 16 massive outlier channels
    outlier_idx = torch.randperm(dim)[:16]
    w_outlier[:, outlier_idx] *= 100.0
    x_outlier = torch.randn(64, dim, device=device) * 0.1
    x_outlier[:, outlier_idx] *= 50.0

    print(f"  {'g':>6}  {'W-MSE':>12}  {'Max-Err':>10}  {'GEMM-MSE':>12}  {'GEMM-SNR':>10}")
    for g in [32, 64, 128, 256]:
        qe = measure_quant_error(w_outlier, g)
        ge = measure_gemm_error(x_outlier, w_outlier, g)
        print(f"  {g:>6}  {qe['mse']:>12.6e}  {qe['max_err']:>10.6f}  {ge['gemm_mse']:>12.6e}  {ge['gemm_snr_db']:>10.2f}")

    # --- Test 3: Rotated outlier weights (Hadamard block-4096) ---
    print("\n--- Test 3: Rotated outlier-heavy weights (block-4096 Hadamard) ---")
    gen = torch.Generator(device='cpu').manual_seed(42)
    signs = (torch.randint(0, 2, (dim,), generator=gen, device='cpu') * 2 - 1).float().to(device)
    rot = NOVARotation(dim=dim, stages=[
        RotationStage(block_size=4096, signs=signs)])

    w_rot = rot.forward(w_outlier.float())
    x_rot = rot.forward(x_outlier.float())

    print(f"  {'g':>6}  {'W-MSE':>12}  {'Max-Err':>10}  {'GEMM-MSE':>12}  {'GEMM-SNR':>10}")
    for g in [32, 64, 128, 256]:
        qe = measure_quant_error(w_rot, g)
        ge = measure_gemm_error(x_rot, w_rot, g)
        print(f"  {g:>6}  {qe['mse']:>12.6e}  {qe['max_err']:>10.6f}  {ge['gemm_mse']:>12.6e}  {ge['gemm_snr_db']:>10.2f}")

    # --- Test 4: Real model weights (if available) ---
    print("\n--- Test 4: Real Llama-70B q_proj weights (layer 1) ---")
    try:
        from transformers import AutoModelForCausalLM
        print("  Loading model (this may take a minute)...")
        model = AutoModelForCausalLM.from_pretrained(
            'NousResearch/Meta-Llama-3.1-70B',
            cache_dir='/mnt/data/huggingface',
            dtype=torch.float16,
            device_map='auto',
        )
        # Get layer 1 q_proj (layer 0 is the outlier, layer 1 is "normal")
        w_real = model.model.layers[1].self_attn.q_proj.weight.data.float()
        # Also get layer 0 for comparison
        w_real_l0 = model.model.layers[0].self_attn.q_proj.weight.data.float()

        print(f"  Layer 1 q_proj shape: {w_real.shape}, std: {w_real.std():.6f}")
        print(f"  Layer 0 q_proj shape: {w_real_l0.shape}, std: {w_real_l0.std():.6f}")

        # Rotated layer 1
        w_real_rot = rot.forward(w_real)
        # Unrotated layer 0
        w_real_l0_unrot = w_real_l0

        print(f"\n  Layer 1 (rotated, block-4096):")
        print(f"  {'g':>6}  {'W-MSE':>12}  {'Max-Err':>10}  {'Scales':>8}")
        for g in [32, 64, 128, 256]:
            qe = measure_quant_error(w_real_rot, g)
            print(f"  {g:>6}  {qe['mse']:>12.6e}  {qe['max_err']:>10.6f}  {qe['n_scales']:>8}")

        print(f"\n  Layer 0 (unrotated — skip_layers=[0]):")
        print(f"  {'g':>6}  {'W-MSE':>12}  {'Max-Err':>10}  {'Scales':>8}")
        for g in [32, 64, 128, 256]:
            qe = measure_quant_error(w_real_l0_unrot, g)
            print(f"  {g:>6}  {qe['mse']:>12.6e}  {qe['max_err']:>10.6f}  {qe['n_scales']:>8}")

        # Check: do ALL 80 layers show the same pattern?
        print(f"\n  All layers — MSE ratio g=32/g=128 (should be <1 if g=32 is better):")
        for layer_idx in [0, 1, 10, 40, 79]:
            w = model.model.layers[layer_idx].self_attn.q_proj.weight.data.float()
            if layer_idx == 0:
                w_test = w  # unrotated
            else:
                w_test = rot.forward(w)  # rotated
            e32 = measure_quant_error(w_test, 32)['mse']
            e128 = measure_quant_error(w_test, 128)['mse']
            ratio = e32 / e128 if e128 > 0 else float('inf')
            rot_label = "unrotated" if layer_idx == 0 else "rotated"
            print(f"    Layer {layer_idx:>2} ({rot_label}): g32 MSE={e32:.6e}, g128 MSE={e128:.6e}, ratio={ratio:.4f}")

        del model
    except Exception as e:
        print(f"  Skipped (model not available): {e}")

    # --- Test 5: Verify quantize/dequant roundtrip is exact ---
    print("\n--- Test 5: Quantize/Dequant roundtrip sanity ---")
    w_test = torch.randn(256, 256, device=device)
    for g in [32, 64, 128]:
        w_q, w_s = quantize_int4_per_group(w_test, g)
        w_d1 = dequantize_int4_per_group(w_q, w_s, g)
        # Dequant twice should give same result (deterministic)
        w_d2 = dequantize_int4_per_group(w_q, w_s, g)
        diff = (w_d1 - w_d2).abs().max().item()
        print(f"  g={g}: double-dequant max diff = {diff:.1e} (should be 0)")
        # Check that quantized values → dequant → re-quantize gives same quantized values
        w_q2, w_s2 = quantize_int4_per_group(w_d1, g)
        w_d3 = dequantize_int4_per_group(w_q2, w_s2, g)
        diff2 = (w_d1 - w_d3).abs().max().item()
        print(f"  g={g}: re-quant max diff = {diff2:.1e} (should be 0 — idempotent)")

    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)
    print("\nIf g=32 has LOWER MSE than g=128 in Tests 1-4, the bug is in the")
    print("perplexity pipeline. If g=32 has HIGHER MSE, there's a genuine")
    print("phenomenon to understand.")


if __name__ == '__main__':
    main()
