"""Debug W4-only perplexity pipeline — focused version.

Traces through the exact same code path as perplexity_eval.py.
"""

import torch
import torch.nn as nn
import sys
import gc
import os
import logging

# Suppress HF progress bars
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
logging.disable(logging.WARNING)

from transformers import AutoModelForCausalLM, AutoTokenizer

from .rotation import NOVARotation, RotationStage, make_identity_rotation
from ..rotated_fp8.fp8_quantize import (
    quantize_int4_per_group, dequantize_int4_per_group,
    quantize_int4_naive, dequantize_int4,
)
from .perplexity_eval import apply_w4a4_adaptive, apply_w4a4, is_eligible, get_layer_index

ELIGIBLE_PROJ = {'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj'}

def main():
    model_id = 'NousResearch/Meta-Llama-3.1-70B'
    cache_dir = '/mnt/data/huggingface'
    hidden_dim = 8192
    weight_group_size = 128

    # Step 1: Verify rotation
    print("=" * 60)
    print("STEP 1: Verify rotation")
    print("=" * 60)
    gen = torch.Generator(device='cpu').manual_seed(42)
    signs = torch.randint(0, 2, (hidden_dim,), generator=gen, device='cpu') * 2 - 1
    rot_large = NOVARotation(dim=hidden_dim, stages=[
        RotationStage(block_size=4096, signs=signs.float())])

    x_test = torch.randn(3, hidden_dim)
    x_rot = rot_large.forward(x_test)
    x_inv = rot_large.inverse(x_rot)
    print(f"  Roundtrip error: {(x_test - x_inv).abs().max().item():.2e}")

    W_test = torch.randn(64, hidden_dim)
    X_test = torch.randn(4, hidden_dim)
    Y_orig = X_test @ W_test.T
    Y_rot = rot_large.forward(X_test) @ rot_large.forward(W_test).T
    print(f"  R(X)@R(W)^T vs X@W^T: rel err = {(Y_orig - Y_rot).norm() / Y_orig.norm():.2e}")

    # Step 2: Load model and test FP16 baseline
    print("\n" + "=" * 60)
    print("STEP 2: FP16 baseline")
    print("=" * 60)
    print("Loading model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, cache_dir=cache_dir,
        dtype=torch.float16, device_map='auto',
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)

    # Check model dtype
    sample_weight = None
    for n, p in model.named_parameters():
        if 'layers.1.self_attn.q_proj' in n:
            sample_weight = p
            break
    if sample_weight is not None:
        print(f"  Sample weight (layer 1 q_proj): device={sample_weight.device}, dtype={sample_weight.dtype}")
        print(f"  Weight stats: mean={sample_weight.data.float().mean():.4e}, std={sample_weight.data.float().std():.4e}")

    # Count eligible layers
    n_eligible = 0
    for name, module in model.named_modules():
        if is_eligible(name, module, hidden_dim):
            n_eligible += 1
    print(f"  Eligible layers: {n_eligible}")

    test_input = tokenizer("The capital of France is", return_tensors='pt')
    input_ids = test_input.input_ids.to('cuda')

    with torch.no_grad():
        out_fp16 = model(input_ids)
        logits_fp16 = out_fp16.logits.float().cpu()
        fp16_loss = nn.CrossEntropyLoss()(
            logits_fp16[:, :-1].reshape(-1, logits_fp16.shape[-1]),
            input_ids[:, 1:].reshape(-1).cpu()
        ).item()
    print(f"  FP16 loss: {fp16_loss:.4f}")
    print(f"  Logits range: [{logits_fp16.min():.2f}, {logits_fp16.max():.2f}]")

    # Step 3: Apply W4-only quantization (manual, matching apply_w4a4_adaptive logic)
    print("\n" + "=" * 60)
    print("STEP 3: Apply W4-only (manual, same logic as apply_w4a4_adaptive)")
    print("=" * 60)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("  Reloading model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, cache_dir=cache_dir,
        dtype=torch.float16, device_map='auto',
    )

    skip_layers = [0]
    identity = make_identity_rotation(hidden_dim)
    hooks = []
    count = 0
    n_skipped_rot = 0
    n_meta = 0
    layer_errors = []  # Track per-layer quantization error

    for name, module in model.named_modules():
        if not is_eligible(name, module, hidden_dim):
            continue

        layer_idx = get_layer_index(name)
        use_rotation = layer_idx not in skip_layers if layer_idx is not None else True
        rotation = rot_large if use_rotation else identity
        has_stages = rotation.n_stages > 0

        with torch.no_grad():
            w = module.weight.data
            if w.device.type == 'meta':
                n_meta += 1
                continue

            w_float = w.float()

            # Check if weight needs to be on CPU for rotation
            w_cpu = w_float.cpu()
            w_rot = rotation.forward(w_cpu) if has_stages else w_cpu

            if w_rot.shape[1] % weight_group_size == 0:
                w_q, w_s = quantize_int4_per_group(w_rot, weight_group_size)
                w_deq = dequantize_int4_per_group(w_q, w_s, weight_group_size)
            else:
                w_q, w_s = quantize_int4_naive(w_rot)
                w_deq = dequantize_int4(w_q, w_s)

            # Track quantization error for first few layers
            if count < 12:
                q_mse = ((w_rot - w_deq)**2).mean().item()
                w_mse = (w_rot**2).mean().item()
                snr = 10 * torch.log10(torch.tensor(w_mse / max(q_mse, 1e-30))).item()
                layer_errors.append((name, layer_idx, use_rotation, q_mse, snr))

            module.weight.data = w_deq.to(device=w.device, dtype=w.dtype)
            del w_float, w_cpu, w_rot, w_q, w_s, w_deq

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
            n_skipped_rot += 1

    print(f"  Applied to {count} layers (skipped rotation: {n_skipped_rot}, meta: {n_meta})")
    print(f"\n  Per-layer quantization errors (first 12):")
    for name, li, use_rot, mse, snr in layer_errors:
        rot_str = "rotated" if use_rot else "no-rot"
        print(f"    L{li:2d} {name.split('.')[-1]:>10s} ({rot_str}): MSE={mse:.4e}, SNR={snr:.1f}dB")

    # Run forward
    with torch.no_grad():
        out_w4 = model(input_ids)
        logits_w4 = out_w4.logits.float().cpu()
        w4_loss = nn.CrossEntropyLoss()(
            logits_w4[:, :-1].reshape(-1, logits_w4.shape[-1]),
            input_ids[:, 1:].reshape(-1).cpu()
        ).item()

    print(f"\n  W4-only loss: {w4_loss:.4f}")
    print(f"  Logits range: [{logits_w4.min():.2f}, {logits_w4.max():.2f}]")
    print(f"  NaN: {torch.isnan(logits_w4).any()}, Inf: {torch.isinf(logits_w4).any()}")

    logit_diff = (logits_fp16 - logits_w4).abs()
    print(f"  Logit diff vs FP16: mean={logit_diff.mean():.2f}, max={logit_diff.max():.2f}")

    # Step 4: Test what happens WITHOUT rotation (just quantize weights, no hooks)
    print("\n" + "=" * 60)
    print("STEP 4: W4-only WITHOUT rotation (quantize-only)")
    print("=" * 60)

    for h in hooks:
        h.remove()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("  Reloading model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, cache_dir=cache_dir,
        dtype=torch.float16, device_map='auto',
    )

    count2 = 0
    for name, module in model.named_modules():
        if not is_eligible(name, module, hidden_dim):
            continue
        with torch.no_grad():
            w = module.weight.data
            if w.device.type == 'meta':
                continue
            w_float = w.float().cpu()
            # NO rotation — just quantize directly
            if w_float.shape[1] % weight_group_size == 0:
                w_q, w_s = quantize_int4_per_group(w_float, weight_group_size)
                w_deq = dequantize_int4_per_group(w_q, w_s, weight_group_size)
            else:
                w_q, w_s = quantize_int4_naive(w_float)
                w_deq = dequantize_int4(w_q, w_s)
            module.weight.data = w_deq.to(device=w.device, dtype=w.dtype)
            del w_float, w_q, w_s, w_deq
        count2 += 1

    print(f"  Applied quantize-only to {count2} layers (NO rotation, NO hooks)")

    with torch.no_grad():
        out_norot = model(input_ids)
        logits_norot = out_norot.logits.float().cpu()
        norot_loss = nn.CrossEntropyLoss()(
            logits_norot[:, :-1].reshape(-1, logits_norot.shape[-1]),
            input_ids[:, 1:].reshape(-1).cpu()
        ).item()

    print(f"  No-rotation loss: {norot_loss:.4f}")
    print(f"  Logits range: [{logits_norot.min():.2f}, {logits_norot.max():.2f}]")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  FP16 baseline:          loss = {fp16_loss:.4f}")
    print(f"  W4-only (rotated):      loss = {w4_loss:.4f}")
    print(f"  W4-only (no rotation):  loss = {norot_loss:.4f}")
    print(f"")
    print(f"  Expected: rotated << no-rotation << ∞ (both close to FP16)")
    print(f"  If rotated >> FP16: rotation is breaking the signal")
    print(f"  If no-rotation is OK: the rotation path has a bug")


if __name__ == '__main__':
    main()
