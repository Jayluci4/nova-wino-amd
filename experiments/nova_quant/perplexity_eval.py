"""Multi-layer perplexity evaluation for W4A4 rotation strategies.

Loads full Llama-3.1-70B, applies W4A4 simulated quantization with different
rotation strategies, and measures WikiText-2 perplexity.

Configurations evaluated:
1. FP16 baseline (no quantization)
2. W4A4 no rotation (identity)
3. W4A4 QuaRot (full-dim random Hadamard)
4. W4A4 NOVA block-256 (discovered optimal)

Usage:
    python -m experiments.nova_quant.perplexity_eval \
        --model NousResearch/Meta-Llama-3.1-70B \
        --rotation /mnt/data/activations/llama70b_layer0/w4a4_best_rotation.json
"""

import torch
import torch.nn as nn
import json
import time
import argparse
import gc
import sys
from typing import List, Tuple, Optional

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from .rotation import (
    NOVARotation, RotationStage, make_random_hadamard_rotation, make_identity_rotation,
)
from ..rotated_fp8.fp8_quantize import (
    quantize_int4_naive, dequantize_int4,
    quantize_int4_per_group, dequantize_int4_per_group,
    quantize_int4_per_token, dequantize_int4_per_token,
)


# Projection layers eligible for W4A4 (all have in_features = hidden_dim)
ELIGIBLE_PROJ = {'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj'}


def is_eligible(name: str, module: nn.Module, hidden_dim: int) -> bool:
    """Check if a linear layer should receive W4A4 quantization."""
    if not isinstance(module, nn.Linear):
        return False
    if module.in_features != hidden_dim:
        return False
    parts = name.split('.')
    return any(p in ELIGIBLE_PROJ for p in parts)


def apply_w4a4(model, rotation: NOVARotation, weight_group_size: int = 128,
               hidden_dim: int = 8192, act_quant: str = 'per_tensor') -> Tuple[int, List]:
    """Apply W4A4 simulated quantization to all eligible linear layers.

    Modifies model in-place:
    - Weights: rotated then fake-quantized (INT4 per-group, stored as FP16)
    - Activations: forward pre-hook rotates + fake-quantizes on each pass

    Layers on meta device (CPU/disk offloaded by device_map='auto') are skipped
    entirely — both weight modification and hook registration — to prevent silent
    corruption where activations get rotated but weights stay original.

    Args:
        act_quant: 'per_tensor' (one scale for entire activation) or
                   'per_token' (one scale per token/row). Per-token is standard
                   in real W4A4 systems (QuaRot, SpinQuant).

    Returns (n_layers_modified, hook_handles).
    """
    hooks = []
    count = 0
    n_meta = 0
    has_stages = rotation.n_stages > 0

    for name, module in model.named_modules():
        if not is_eligible(name, module, hidden_dim):
            continue

        # Skip meta-device layers (offloaded by accelerate/device_map='auto').
        # Registering rotation hooks on these would corrupt outputs: the hook
        # rotates activations but meta weights stay unrotated, so R(X)@W != X@W.
        if module.weight.device.type == 'meta':
            n_meta += 1
            continue

        # --- Weight: rotate + quantize + dequantize (once) ---
        with torch.no_grad():
            w = module.weight.data.float()
            w_rot = rotation.forward(w.cpu()) if has_stages else w.cpu()
            if w_rot.shape[1] % weight_group_size == 0:
                w_q, w_s = quantize_int4_per_group(w_rot, weight_group_size)
                w_deq = dequantize_int4_per_group(w_q, w_s, weight_group_size)
            else:
                w_q, w_s = quantize_int4_naive(w_rot)
                w_deq = dequantize_int4(w_q, w_s)
            module.weight.data = w_deq.to(device=module.weight.device, dtype=module.weight.dtype)
            del w, w_rot, w_q, w_s, w_deq

        # --- Activation hook: rotate + quantize each forward pass ---
        def _make_hook(rot, has_s, aq):
            def hook(mod, args):
                x = args[0]
                with torch.no_grad():
                    xf = x.float()
                    xr = rot.forward(xf) if has_s else xf
                    if aq == 'per_token':
                        xq, xs = quantize_int4_per_token(xr)
                        xd = dequantize_int4_per_token(xq, xs).to(x.dtype)
                    else:
                        xq, xs = quantize_int4_naive(xr)
                        xd = dequantize_int4(xq, xs).to(x.dtype)
                return (xd,) + args[1:]
            return hook

        h = module.register_forward_pre_hook(_make_hook(rotation, has_stages, act_quant))
        hooks.append(h)
        count += 1

    if n_meta > 0:
        print(f"  WARNING: {n_meta} eligible layers on meta device (offloaded) — skipped.")
        print(f"  Results reflect PARTIAL quantization ({count}/{count + n_meta} layers).")
        print(f"  For valid results, ensure enough GPU memory for the full model.")

    return count, hooks


def get_layer_index(name: str) -> Optional[int]:
    """Extract transformer layer index from module name like 'model.layers.42.self_attn.q_proj'."""
    parts = name.split('.')
    for i, p in enumerate(parts):
        if p == 'layers' and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                pass
    return None


def apply_w4a4_adaptive(
    model, rotation_large: NOVARotation, weight_group_size: int = 128,
    hidden_dim: int = 8192, skip_layers: Optional[List[int]] = None,
    act_quant: str = 'per_tensor',
    quant_parts: str = 'both',
) -> Tuple[int, List]:
    """Apply W4A4 with per-layer adaptive rotation.

    Layers in skip_layers get quantization-only (no rotation).
    All other layers get the large-block rotation.

    Layers on meta device (CPU/disk offloaded by device_map='auto') are skipped
    entirely — both weight modification and hook registration — to prevent silent
    corruption where activations get rotated but weights stay original.

    Args:
        quant_parts: 'both' (W4A4), 'weights_only' (W4 + FP16 activations),
                     or 'activations_only' (FP16 weights + A4). Rotation is
                     always applied to both sides (lossless in FP16) so we
                     isolate only the quantization error.
    """
    if skip_layers is None:
        skip_layers = [0]

    hooks = []
    count = 0
    n_meta = 0
    identity = make_identity_rotation(hidden_dim)

    for name, module in model.named_modules():
        if not is_eligible(name, module, hidden_dim):
            continue

        # Skip meta-device layers (offloaded by accelerate/device_map='auto').
        # Registering rotation hooks on these would corrupt outputs: the hook
        # rotates activations but meta weights stay unrotated, so R(X)@W != X@W.
        if module.weight.device.type == 'meta':
            n_meta += 1
            continue

        layer_idx = get_layer_index(name)
        use_rotation = layer_idx not in skip_layers if layer_idx is not None else True
        rotation = rotation_large if use_rotation else identity
        has_stages = rotation.n_stages > 0

        with torch.no_grad():
            w = module.weight.data.float()
            w_cpu = w.cpu()
            w_rot = rotation.forward(w_cpu) if has_stages else w_cpu
            if quant_parts in ('both', 'weights_only'):
                if w_rot.shape[1] % weight_group_size == 0:
                    w_q, w_s = quantize_int4_per_group(w_rot, weight_group_size)
                    w_deq = dequantize_int4_per_group(w_q, w_s, weight_group_size)
                else:
                    w_q, w_s = quantize_int4_naive(w_rot)
                    w_deq = dequantize_int4(w_q, w_s)
                module.weight.data = w_deq.to(device=w.device, dtype=module.weight.dtype)
                del w, w_cpu, w_rot, w_q, w_s, w_deq
            else:
                # A4-only: rotate weights without quantization (lossless in FP16)
                module.weight.data = w_rot.to(device=w.device, dtype=module.weight.dtype)
                del w, w_cpu, w_rot

        def _make_hook(rot, has_s, aq, qp):
            def hook(mod, args):
                x = args[0]
                with torch.no_grad():
                    xf = x.float()
                    xr = rot.forward(xf) if has_s else xf
                    if qp in ('both', 'activations_only'):
                        if aq == 'per_token':
                            xq, xs = quantize_int4_per_token(xr)
                            xd = dequantize_int4_per_token(xq, xs).to(x.dtype)
                        else:
                            xq, xs = quantize_int4_naive(xr)
                            xd = dequantize_int4(xq, xs).to(x.dtype)
                    else:
                        # W4-only: rotate activations without quantization
                        xd = xr.to(x.dtype)
                return (xd,) + args[1:]
            return hook

        h = module.register_forward_pre_hook(
            _make_hook(rotation, has_stages, act_quant, quant_parts))
        hooks.append(h)
        count += 1

    if n_meta > 0:
        print(f"  WARNING: {n_meta} eligible layers on meta device (offloaded) — skipped.")
        print(f"  Results reflect PARTIAL quantization ({count}/{count + n_meta} layers).")
        print(f"  For valid results, ensure enough GPU memory for the full model.")

    return count, hooks


@torch.no_grad()
def evaluate_perplexity(model, tokenizer, text: str,
                        max_length: int = 2048, stride: int = 2048,
                        max_tokens: Optional[int] = None,
                        device: str = 'cuda') -> Tuple[float, int]:
    """Evaluate perplexity on text using sliding-window NLL.

    Returns (perplexity, total_tokens_evaluated).
    """
    model.eval()
    encodings = tokenizer(text, return_tensors='pt')
    input_ids = encodings.input_ids[0]
    seq_len = min(input_ids.size(0), max_tokens) if max_tokens else input_ids.size(0)
    input_ids = input_ids[:seq_len]

    print(f"  Tokens: {seq_len:,}, stride={stride}, ctx={max_length}")

    nlls = []
    total_tgt = 0
    prev_end = 0
    n_chunks = 0
    t0 = time.time()

    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        trg_len = end - prev_end

        chunk = input_ids[begin:end].unsqueeze(0).to(device)
        target = chunk.clone()
        target[:, :-trg_len] = -100

        loss = model(chunk, labels=target).loss.float()
        nlls.append(loss.item() * trg_len)
        total_tgt += trg_len

        prev_end = end
        n_chunks += 1

        if n_chunks % 5 == 0:
            dt = time.time() - t0
            rppl = torch.exp(torch.tensor(sum(nlls) / total_tgt)).item()
            print(f"  [{n_chunks} chunks, {total_tgt:,} tok, {dt:.0f}s] PPL ≈ {rppl:.2f}")

        if end == seq_len:
            break

    ppl = torch.exp(torch.tensor(sum(nlls) / total_tgt)).item()
    dt = time.time() - t0
    print(f"  Done: {n_chunks} chunks, {total_tgt:,} tokens, {dt:.0f}s → PPL = {ppl:.2f}")
    return ppl, total_tgt


def load_model(model_id: str, cache_dir: str):
    """Load model + tokenizer in FP16."""
    print(f"Loading {model_id} ...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, cache_dir=cache_dir,
        torch_dtype=torch.float16, device_map='auto',
    )
    dt = time.time() - t0
    n = sum(p.numel() for p in model.parameters())
    n_meta = sum(1 for p in model.parameters() if p.device.type == 'meta')
    print(f"  Loaded in {dt:.0f}s — {n/1e9:.1f}B params")
    if n_meta > 0:
        print(f"  WARNING: {n_meta} parameter tensors on meta device (offloaded).")
        print(f"  Quantization results will be INVALID — not all layers can be modified.")
        print(f"  Free GPU memory or reduce model size for valid perplexity measurements.")
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='NousResearch/Meta-Llama-3.1-70B')
    parser.add_argument('--cache-dir', default='/mnt/data/huggingface')
    parser.add_argument('--rotation',
                        default='/mnt/data/activations/llama70b_layer0/w4a4_best_rotation.json')
    parser.add_argument('--max-length', type=int, default=2048)
    parser.add_argument('--stride', type=int, default=2048)
    parser.add_argument('--max-tokens', type=int, default=None)
    parser.add_argument('--weight-group-size', type=int, default=128)
    parser.add_argument('--configs', nargs='+',
                        default=['fp16', 'w4a4_identity', 'w4a4_quarot', 'w4a4_nova'],
                        choices=['fp16', 'w4a4_identity', 'w4a4_quarot', 'w4a4_nova',
                                 'w4a4_adaptive', 'w4_adaptive', 'a4_adaptive'])
    parser.add_argument('--act-quant', choices=['per_tensor', 'per_token'],
                        default='per_tensor',
                        help='Activation quantization granularity. per_token is standard '
                             'in real W4A4 systems (QuaRot, SpinQuant).')
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    # --- Dataset ---
    print("Loading WikiText-2 test split ...")
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    text = "\n\n".join(item['text'] for item in ds if item['text'].strip())
    print(f"  {len(text):,} chars")

    # --- Rotations ---
    print(f"Loading NOVA rotation: {args.rotation}")
    nova_rot = NOVARotation.load(args.rotation)
    dim = nova_rot.dim
    print(f"  {nova_rot}")

    rotations = {
        'fp16': None,
        'w4a4_identity': make_identity_rotation(dim),
        'w4a4_quarot': make_random_hadamard_rotation(dim, seed=42),
        'w4a4_nova': nova_rot,
    }

    results = {}

    for cfg in args.configs:
        print(f"\n{'='*60}")
        print(f"  CONFIG: {cfg}")
        print(f"{'='*60}")

        model, tokenizer = load_model(args.model, args.cache_dir)
        hooks = []

        act_quant = args.act_quant
        print(f"  Activation quantization: {act_quant}")

        if cfg in ('w4a4_adaptive', 'w4_adaptive', 'a4_adaptive'):
            # Use the rotation loaded from --rotation file
            rot_large = nova_rot
            if cfg == 'w4_adaptive':
                qp = 'weights_only'
            elif cfg == 'a4_adaptive':
                qp = 'activations_only'
            else:
                qp = 'both'
            n, hooks = apply_w4a4_adaptive(
                model, rot_large,
                weight_group_size=args.weight_group_size,
                hidden_dim=dim, skip_layers=[0],
                act_quant=act_quant,
                quant_parts=qp)
            labels = {'w4a4_adaptive': 'W4A4', 'w4_adaptive': 'W4-only (FP16 activations)',
                      'a4_adaptive': 'A4-only (FP16 weights)'}
            print(f"  {labels[cfg]} adaptive applied to {n} layers (layer 0: no rotation, rest: block-4096)")
        elif cfg != 'fp16':
            rot = rotations[cfg]
            n, hooks = apply_w4a4(model, rot,
                                  weight_group_size=args.weight_group_size,
                                  hidden_dim=dim,
                                  act_quant=act_quant)
            print(f"  W4A4 applied to {n} linear layers")

        ppl, n_tok = evaluate_perplexity(
            model, tokenizer, text,
            max_length=args.max_length,
            stride=args.stride,
            max_tokens=args.max_tokens,
        )
        results[cfg] = {'perplexity': ppl, 'tokens': n_tok}
        print(f"\n  >>> {cfg}: PPL = {ppl:.2f}")

        for h in hooks:
            h.remove()
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    # --- Summary ---
    print(f"\n{'='*60}")
    print("PERPLEXITY RESULTS — WikiText-2")
    print(f"{'='*60}")
    print(f"{'Config':<20} {'PPL':>10}")
    print(f"{'-'*30}")
    for cfg in args.configs:
        print(f"{cfg:<20} {results[cfg]['perplexity']:>10.2f}")

    if 'fp16' in results:
        fp16 = results['fp16']['perplexity']
        print(f"\nDegradation vs FP16 (PPL={fp16:.2f}):")
        for cfg in args.configs:
            if cfg == 'fp16':
                continue
            p = results[cfg]['perplexity']
            print(f"  {cfg:<20} Δ={p - fp16:+.2f}  ({(p - fp16) / fp16 * 100:+.1f}%)")

    if args.output:
        out = {
            'model': args.model,
            'dataset': 'wikitext-2-raw-v1',
            'max_length': args.max_length,
            'stride': args.stride,
            'max_tokens': args.max_tokens,
            'weight_group_size': args.weight_group_size,
            'act_quant': args.act_quant,
            'rotation_file': args.rotation,
            'results': results,
        }
        with open(args.output, 'w') as f:
            json.dump(out, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == '__main__':
    main()
