"""Debug W4-only perplexity pipeline step by step.

Traces through the exact same code path as perplexity_eval.py to find
where the massive PPL degradation (10.27 → 1810) comes from.
"""

import torch
import torch.nn as nn
import sys
import gc

from transformers import AutoModelForCausalLM, AutoTokenizer

from .rotation import NOVARotation, RotationStage, make_identity_rotation
from ..rotated_fp8.fp8_quantize import (
    quantize_int4_per_group, dequantize_int4_per_group,
    quantize_int4_naive, dequantize_int4,
)

ELIGIBLE_PROJ = {'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj'}

def main():
    model_id = 'NousResearch/Meta-Llama-3.1-70B'
    cache_dir = '/mnt/data/huggingface'
    hidden_dim = 8192
    weight_group_size = 128

    # Step 1: Verify rotation is orthogonal
    print("=" * 60)
    print("STEP 1: Verify rotation orthogonality")
    print("=" * 60)
    gen = torch.Generator(device='cpu').manual_seed(42)
    signs = torch.randint(0, 2, (hidden_dim,), generator=gen, device='cpu') * 2 - 1
    rot_large = NOVARotation(dim=hidden_dim, stages=[
        RotationStage(block_size=4096, signs=signs.float())])

    x_test = torch.randn(3, hidden_dim)
    x_rot = rot_large.forward(x_test)
    x_inv = rot_large.inverse(x_rot)
    roundtrip_err = (x_test - x_inv).abs().max().item()
    print(f"  Roundtrip error (forward→inverse): {roundtrip_err:.2e}")

    # Check orthogonality: ||R(x)|| should equal ||x||
    norm_orig = x_test.norm(dim=-1)
    norm_rot = x_rot.norm(dim=-1)
    norm_ratio = (norm_rot / norm_orig)
    print(f"  Norm ratios: {norm_ratio.tolist()}")
    print(f"  Max norm deviation: {(norm_ratio - 1).abs().max().item():.2e}")

    # Check that R(X) @ R(W)^T ≈ X @ W^T
    W_test = torch.randn(64, hidden_dim)
    X_test = torch.randn(4, hidden_dim)
    Y_orig = X_test @ W_test.T
    Y_rot = rot_large.forward(X_test) @ rot_large.forward(W_test).T
    rot_cancel_err = (Y_orig - Y_rot).abs().max().item()
    rot_cancel_rel = (Y_orig - Y_rot).norm() / Y_orig.norm()
    print(f"  R(X)@R(W)^T vs X@W^T — max err: {rot_cancel_err:.2e}, rel err: {rot_cancel_rel:.2e}")

    # Step 2: Check quantization error on random data
    print("\n" + "=" * 60)
    print("STEP 2: Quantization error on rotated weights")
    print("=" * 60)
    W_rot = rot_large.forward(W_test)
    w_q, w_s = quantize_int4_per_group(W_rot, weight_group_size)
    w_deq = dequantize_int4_per_group(w_q, w_s, weight_group_size)
    quant_mse = ((W_rot - w_deq) ** 2).mean().item()
    quant_snr = 10 * torch.log10(torch.tensor((W_rot**2).mean() / quant_mse)).item()
    print(f"  Quant MSE: {quant_mse:.6e}")
    print(f"  Quant SNR: {quant_snr:.1f} dB")

    # End-to-end: X → R(X) → matmul with R(W_q) → should ≈ X @ W^T
    X_rot = rot_large.forward(X_test)
    Y_quant = X_rot @ w_deq.T
    Y_ref = X_test @ W_test.T
    e2e_err = (Y_quant - Y_ref).abs().max().item()
    e2e_rel = (Y_quant - Y_ref).norm() / Y_ref.norm()
    print(f"  End-to-end (R(X)@R(W_q)^T vs X@W^T) — max err: {e2e_err:.2e}, rel: {e2e_rel:.2e}")

    # Step 3: Load ONE layer of the real model and test
    print("\n" + "=" * 60)
    print("STEP 3: Real model — single layer test")
    print("=" * 60)
    print(f"Loading {model_id} ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, cache_dir=cache_dir,
        dtype=torch.float16, device_map='auto',
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)

    # Check model dtype and device distribution
    n_params = sum(p.numel() for p in model.parameters())
    devices = set()
    dtypes = set()
    for n, p in model.named_parameters():
        if p.device.type != 'meta':
            devices.add(str(p.device))
            dtypes.add(str(p.dtype))
    print(f"  {n_params/1e9:.1f}B params, devices: {devices}, dtypes: {dtypes}")

    # Find first eligible layer at layer index 1 (not skipped)
    target_name = None
    target_module = None
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if module.in_features != hidden_dim:
            continue
        parts = name.split('.')
        if not any(p in ELIGIBLE_PROJ for p in parts):
            continue
        # Get layer index
        layer_idx = None
        for i, p in enumerate(parts):
            if p == 'layers' and i + 1 < len(parts):
                try:
                    layer_idx = int(parts[i + 1])
                except ValueError:
                    pass
        if layer_idx == 1:  # Use layer 1 (not skipped)
            target_name = name
            target_module = module
            break

    if target_module is None:
        print("ERROR: No eligible layer found at layer 1!")
        return

    print(f"  Testing layer: {target_name}")
    print(f"  Weight shape: {target_module.weight.shape}")
    print(f"  Weight device: {target_module.weight.device}")
    print(f"  Weight dtype: {target_module.weight.dtype}")

    # Get the weight
    w_orig = target_module.weight.data.float().cpu()
    print(f"  Weight stats: mean={w_orig.mean():.4e}, std={w_orig.std():.4e}, "
          f"min={w_orig.min():.4e}, max={w_orig.max():.4e}")

    # Apply rotation
    w_rot = rot_large.forward(w_orig)
    print(f"  Rotated stats: mean={w_rot.mean():.4e}, std={w_rot.std():.4e}, "
          f"min={w_rot.min():.4e}, max={w_rot.max():.4e}")

    # Quantize
    w_q, w_s = quantize_int4_per_group(w_rot, weight_group_size)
    w_deq = dequantize_int4_per_group(w_q, w_s, weight_group_size)

    q_mse = ((w_rot - w_deq)**2).mean().item()
    q_snr = 10 * torch.log10(torch.tensor((w_rot**2).mean() / max(q_mse, 1e-30))).item()
    print(f"  Quant error — MSE: {q_mse:.6e}, SNR: {q_snr:.1f} dB")

    # End-to-end test with a random input
    x_in = torch.randn(1, 8, hidden_dim)
    y_ref = x_in @ w_orig.T  # Reference: no rotation, no quantization

    x_rot = rot_large.forward(x_in)
    y_quant = x_rot @ w_deq.T  # Quantized path: rotated input @ rotated quantized weight

    e2e_maxerr = (y_quant - y_ref).abs().max().item()
    e2e_rel = (y_quant - y_ref).norm() / y_ref.norm()
    print(f"  End-to-end error — max: {e2e_maxerr:.4e}, relative: {e2e_rel:.4e}")

    # Step 4: Full pipeline sanity — apply to ALL layers, run one forward pass
    print("\n" + "=" * 60)
    print("STEP 4: Full pipeline — one forward pass comparison")
    print("=" * 60)

    # First, get FP16 baseline output
    test_input = tokenizer("The capital of France is", return_tensors='pt')
    input_ids = test_input.input_ids.to('cuda')

    with torch.no_grad():
        out_fp16 = model(input_ids)
        logits_fp16 = out_fp16.logits.float().cpu()
        fp16_loss = nn.CrossEntropyLoss()(
            logits_fp16[:, :-1].reshape(-1, logits_fp16.shape[-1]),
            input_ids[:, 1:].reshape(-1).cpu()
        ).item()
    print(f"  FP16 baseline — loss: {fp16_loss:.4f}, logits range: "
          f"[{logits_fp16.min():.2f}, {logits_fp16.max():.2f}]")

    # Now apply W4-only quantization and re-test
    # Reload model fresh
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("  Reloading model for W4-only test...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, cache_dir=cache_dir,
        dtype=torch.float16, device_map='auto',
    )

    # Apply W4-only (same logic as apply_w4a4_adaptive with quant_parts='weights_only')
    skip_layers = [0]
    identity = make_identity_rotation(hidden_dim)
    hooks = []
    count = 0
    n_skipped = 0
    n_no_group = 0

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if module.in_features != hidden_dim:
            continue
        parts = name.split('.')
        if not any(p in ELIGIBLE_PROJ for p in parts):
            continue

        layer_idx = None
        for i, p in enumerate(parts):
            if p == 'layers' and i + 1 < len(parts):
                try:
                    layer_idx = int(parts[i + 1])
                except ValueError:
                    pass

        use_rotation = layer_idx not in skip_layers if layer_idx is not None else True
        rotation = rot_large if use_rotation else identity
        has_stages = rotation.n_stages > 0

        with torch.no_grad():
            w = module.weight.data.float()
            if w.device.type == 'meta':
                print(f"  WARNING: {name} is on meta device! Skipping.")
                continue

            w_rot = rotation.forward(w.cpu()) if has_stages else w.cpu()

            if w_rot.shape[1] % weight_group_size == 0:
                w_q, w_s = quantize_int4_per_group(w_rot, weight_group_size)
                w_deq = dequantize_int4_per_group(w_q, w_s, weight_group_size)
            else:
                w_q, w_s = quantize_int4_naive(w_rot)
                w_deq = dequantize_int4(w_q, w_s)
                n_no_group += 1

            module.weight.data = w_deq.to(device=module.weight.device, dtype=module.weight.dtype)
            del w, w_rot, w_q, w_s, w_deq

        def _make_hook(rot, has_s):
            def hook(mod, args):
                x = args[0]
                with torch.no_grad():
                    xf = x.float()
                    xr = rot.forward(xf) if has_s else xf
                    xd = xr.to(x.dtype)
                return (xd,) + args[1:]
            return hook

        h = module.register_forward_pre_hook(_make_hook(rotation, has_stages))
        hooks.append(h)
        count += 1

        if not use_rotation:
            n_skipped += 1

    print(f"  Applied W4-only to {count} layers ({n_skipped} skipped rotation, {n_no_group} fell back to per-tensor)")

    # Run forward pass with quantized model
    with torch.no_grad():
        out_w4 = model(input_ids)
        logits_w4 = out_w4.logits.float().cpu()
        w4_loss = nn.CrossEntropyLoss()(
            logits_w4[:, :-1].reshape(-1, logits_w4.shape[-1]),
            input_ids[:, 1:].reshape(-1).cpu()
        ).item()

    print(f"  W4-only — loss: {w4_loss:.4f}, logits range: "
          f"[{logits_w4.min():.2f}, {logits_w4.max():.2f}]")

    # Compare
    logit_diff = (logits_fp16 - logits_w4).abs()
    print(f"  Logit diff — mean: {logit_diff.mean():.4f}, max: {logit_diff.max():.4f}")

    has_nan = torch.isnan(logits_w4).any().item()
    has_inf = torch.isinf(logits_w4).any().item()
    print(f"  NaN: {has_nan}, Inf: {has_inf}")

    # Cleanup
    for h in hooks:
        h.remove()
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Step 5: Compare with apply_w4a4_adaptive directly
    print("\n" + "=" * 60)
    print("STEP 5: Compare with apply_w4a4_adaptive() directly")
    print("=" * 60)
    print("  Reloading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, cache_dir=cache_dir,
        dtype=torch.float16, device_map='auto',
    )

    from .perplexity_eval import apply_w4a4_adaptive
    gen2 = torch.Generator(device='cpu').manual_seed(42)
    signs2 = torch.randint(0, 2, (hidden_dim,), generator=gen2, device='cpu') * 2 - 1
    rot2 = NOVARotation(dim=hidden_dim, stages=[
        RotationStage(block_size=4096, signs=signs2.float())])

    n, hooks2 = apply_w4a4_adaptive(
        model, rot2,
        weight_group_size=weight_group_size,
        hidden_dim=hidden_dim, skip_layers=[0],
        act_quant='per_token',
        quant_parts='weights_only')
    print(f"  apply_w4a4_adaptive applied to {n} layers")

    with torch.no_grad():
        out_w4b = model(input_ids)
        logits_w4b = out_w4b.logits.float().cpu()
        w4b_loss = nn.CrossEntropyLoss()(
            logits_w4b[:, :-1].reshape(-1, logits_w4b.shape[-1]),
            input_ids[:, 1:].reshape(-1).cpu()
        ).item()

    print(f"  apply_w4a4_adaptive — loss: {w4b_loss:.4f}, logits range: "
          f"[{logits_w4b.min():.2f}, {logits_w4b.max():.2f}]")

    diff_ab = (logits_w4 - logits_w4b).abs()
    print(f"  Diff vs manual implementation — mean: {diff_ab.mean():.4f}, max: {diff_ab.max():.4f}")

    for h in hooks2:
        h.remove()
    del model
    gc.collect()
    torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  FP16 loss:            {fp16_loss:.4f}")
    print(f"  W4-only (manual):     {w4_loss:.4f}")
    print(f"  W4-only (adaptive):   {w4b_loss:.4f}")
    print(f"  If W4 losses are >> FP16, quantization pipeline is broken")
    print(f"  If manual != adaptive, bug is in apply_w4a4_adaptive")


if __name__ == '__main__':
    main()
