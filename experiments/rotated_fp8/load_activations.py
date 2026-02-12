"""Load real activations from a Llama model's layer 0 for quantization experiments.

Loads only shard 1 of the model, which contains the embedding layer and all
layer 0 weights. Manually constructs a single transformer block and runs a
forward pass with hooks to capture activations.

Captures hidden_size-dim activations (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj).
Skips down_proj (intermediate_size-dim, typically not power-of-2).
"""

import os
import math
import torch
import json
from pathlib import Path
from typing import Dict, Tuple, Union
from safetensors.torch import load_file

SAMPLE_PROMPT = (
    "The Walsh-Hadamard transform is a mathematical operation that decomposes a signal "
    "into a set of orthogonal, square-wave-like basis functions. In the context of neural "
    "network quantization, it serves as a rotation that spreads activation outliers across "
    "all channels, dramatically reducing the dynamic range that must be captured by "
    "low-precision formats like FP8. This is particularly important for large language "
    "models where specific channels consistently produce values orders of magnitude larger "
    "than their neighbors, making naive quantization catastrophically lossy. The key insight "
    "is that the Walsh-Hadamard transform can be computed in O(n log n) time using the "
    "butterfly algorithm, making it practical for online application during inference. "
    "When combined with random sign flips, the transform provably gaussianizes any input "
    "distribution, ensuring that the quantization grid is efficiently utilized. This "
    "technique, known as rotated quantization, achieves near-lossless compression of "
    "model activations from 16-bit to 8-bit formats, effectively halving memory bandwidth "
    "requirements while preserving model accuracy."
)


def _apply_rotary_emb(x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary positional embeddings (RoPE) to query/key tensors."""
    # x: (batch, n_heads, seq_len, head_dim)
    x_r = x.float().reshape(*x.shape[:-1], -1, 2)
    x0, x1 = x_r[..., 0], x_r[..., 1]
    cos = freqs_cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim/2)
    sin = freqs_sin.unsqueeze(0).unsqueeze(0)
    out0 = x0 * cos - x1 * sin
    out1 = x0 * sin + x1 * cos
    out = torch.stack([out0, out1], dim=-1).reshape(x.shape)
    return out.to(x.dtype)


def _build_rope_cache(seq_len: int, head_dim: int, base: float = 500000.0,
                      device: torch.device = torch.device('cpu')) -> tuple:
    """Build RoPE frequency cache for Llama-3.1."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)
    return freqs.cos(), freqs.sin()


def load_activations(
    model_name: str = "NousResearch/Meta-Llama-3.1-70B",
    cache_dir: str = None,
    save_dir: str = None,
    device: str = "cuda",
    return_weights: bool = False,
) -> Union[Dict[str, torch.Tensor], Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]:
    """Extract layer 0 activations (and optionally weights) from a Llama model.

    Manually constructs the embedding + layer 0 from shard weights and runs
    a forward pass. This avoids loading the full model.

    Args:
        model_name: HuggingFace model ID.
        cache_dir: Where model files are cached. Defaults to HF_HOME.
        save_dir: If provided, save activations (and weights) as .pt files here.
        device: Target device.
        return_weights: If True, also return weight matrices for W4A4 evaluation.

    Returns:
        If return_weights is False: Dict mapping layer names to activation tensors.
        If return_weights is True: (activations, weights) tuple where weights maps
            proj names to weight matrices (float32, on CPU). Only includes projs
            where input dim == hidden_size (power of 2).
    """
    from transformers import AutoTokenizer

    if cache_dir is None:
        cache_dir = os.environ.get('HF_HOME', '/mnt/data/huggingface')

    # Locate shard file
    model_dir = os.path.join(cache_dir, f"models--{model_name.replace('/', '--')}")
    snapshot_dirs = list(Path(model_dir, "snapshots").iterdir())
    snapshot_dir = snapshot_dirs[0]
    # Find the first shard (contains embedding + layer 0 for all Llama sizes)
    shard_files = sorted(snapshot_dir.glob("model-00001-of-*.safetensors"))
    if not shard_files:
        raise FileNotFoundError(f"No safetensor shards found in {snapshot_dir}")
    shard_path = shard_files[0]

    print(f"Loading shard from {shard_path}...")
    weights = load_file(str(shard_path), device='cpu')

    print(f"Loaded {len(weights)} tensors from shard 1:")
    for k in sorted(weights.keys()):
        print(f"  {k}: {list(weights[k].shape)}")

    # Auto-detect model config from HuggingFace
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
    hidden_size = config.hidden_size
    intermediate_size = config.intermediate_size
    num_attention_heads = config.num_attention_heads
    num_key_value_heads = config.num_key_value_heads
    head_dim = hidden_size // num_attention_heads
    rms_norm_eps = config.rms_norm_eps
    rope_theta = getattr(config, 'rope_theta', 500000.0)
    print(f"\nModel config: hidden={hidden_size}, intermediate={intermediate_size}, "
          f"heads={num_attention_heads}, kv_heads={num_key_value_heads}, head_dim={head_dim}")

    # Load tokenizer and tokenize
    print(f"\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    tokens = tokenizer(SAMPLE_PROMPT, return_tensors="pt")
    input_ids = tokens["input_ids"]  # (1, seq_len)
    seq_len = input_ids.shape[1]
    print(f"Tokenized prompt: {seq_len} tokens")

    # Move weights to device
    embed_weight = weights['model.embed_tokens.weight'].to(device=device, dtype=torch.float16)
    ln_weight = weights['model.layers.0.input_layernorm.weight'].to(device=device, dtype=torch.float16)
    post_ln_weight = weights['model.layers.0.post_attention_layernorm.weight'].to(device=device, dtype=torch.float16)

    proj_weights = {}
    for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']:
        if proj_name in ['gate_proj', 'up_proj', 'down_proj']:
            key = f'model.layers.0.mlp.{proj_name}.weight'
        else:
            key = f'model.layers.0.self_attn.{proj_name}.weight'
        proj_weights[proj_name] = weights[key].to(device=device, dtype=torch.float16)

    # Free CPU weights
    del weights

    # --- Manual forward pass through layer 0 ---
    activations = {}

    with torch.no_grad():
        # Embedding
        x = embed_weight[input_ids.to(device)]  # (1, seq_len, 8192)
        print(f"Embedding output: shape={list(x.shape)}, dtype={x.dtype}")

        # RMSNorm
        def rms_norm(x, weight, eps=rms_norm_eps):
            variance = x.float().pow(2).mean(-1, keepdim=True)
            x_normed = x.float() * torch.rsqrt(variance + eps)
            return (weight.float() * x_normed).to(x.dtype)

        residual = x
        x_normed = rms_norm(x, ln_weight)

        # Capture pre-projection activations (input to each linear layer)
        # These are what we want — the hidden states BEFORE the linear projection
        activations['q_proj'] = x_normed.detach().cpu().float()
        activations['k_proj'] = x_normed.detach().cpu().float()
        activations['v_proj'] = x_normed.detach().cpu().float()

        # Self-attention projections
        q = torch.nn.functional.linear(x_normed, proj_weights['q_proj'])
        k = torch.nn.functional.linear(x_normed, proj_weights['k_proj'])
        v = torch.nn.functional.linear(x_normed, proj_weights['v_proj'])

        # Reshape for multi-head attention
        batch = 1
        q = q.view(batch, seq_len, num_attention_heads, head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, num_key_value_heads, head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, num_key_value_heads, head_dim).transpose(1, 2)

        # Apply RoPE
        freqs_cos, freqs_sin = _build_rope_cache(seq_len, head_dim, rope_theta, device=torch.device(device))
        q = _apply_rotary_emb(q, freqs_cos, freqs_sin)
        k = _apply_rotary_emb(k, freqs_cos, freqs_sin)

        # GQA: repeat KV heads
        n_rep = num_attention_heads // num_key_value_heads  # 8
        k = k.repeat_interleave(n_rep, dim=1)
        v = v.repeat_interleave(n_rep, dim=1)

        # Scaled dot-product attention
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=True
        )

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, hidden_size)

        # Capture o_proj input activation
        activations['o_proj'] = attn_output.detach().cpu().float()

        # Output projection
        attn_result = torch.nn.functional.linear(attn_output, proj_weights['o_proj'])

        # Residual connection
        x = residual + attn_result

        # Post-attention layernorm
        residual2 = x
        x_normed2 = rms_norm(x, post_ln_weight)

        # Capture MLP input activations
        activations['gate_proj'] = x_normed2.detach().cpu().float()
        activations['up_proj'] = x_normed2.detach().cpu().float()

    print(f"\nCaptured {len(activations)} activation tensors:")
    for name, tensor in activations.items():
        print(f"  {name}: shape={list(tensor.shape)}, "
              f"max={tensor.abs().max():.2f}, mean={tensor.abs().mean():.4f}, "
              f"max/mean={tensor.abs().max()/tensor.abs().mean():.0f}x")

    # Collect weight matrices (only those with input_dim=8192)
    weight_tensors = {}
    if return_weights:
        for proj_name, w in proj_weights.items():
            # w shape: [out_features, in_features]
            # Only include if in_features == 8192 (power of 2)
            if w.shape[1] == hidden_size:
                weight_tensors[proj_name] = w.detach().cpu().float()
        print(f"\nCollected {len(weight_tensors)} weight matrices:")
        for name, w in weight_tensors.items():
            print(f"  {name}: shape={list(w.shape)}")

    # Save if requested
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        for name, tensor in activations.items():
            path = os.path.join(save_dir, f"{name}.pt")
            torch.save(tensor, path)
            print(f"  Saved {path}")

        if return_weights:
            weights_dir = os.path.join(save_dir, "weights")
            os.makedirs(weights_dir, exist_ok=True)
            for name, tensor in weight_tensors.items():
                path = os.path.join(weights_dir, f"{name}.pt")
                torch.save(tensor, path)
                print(f"  Saved {path}")

        meta = {name: list(t.shape) for name, t in activations.items()}
        if return_weights:
            meta['weights'] = {name: list(t.shape) for name, t in weight_tensors.items()}
        with open(os.path.join(save_dir, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)

    if return_weights:
        return activations, weight_tensors
    return activations


def load_cached_activations(
    save_dir: str,
    load_weights: bool = False,
) -> Union[Dict[str, torch.Tensor], Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]:
    """Load previously saved activations (and optionally weights) from disk.

    Args:
        save_dir: Directory containing .pt files from load_activations().
        load_weights: If True, also load weight matrices from weights/ subdir.

    Returns:
        If load_weights is False: Dict mapping layer names to activation tensors.
        If load_weights is True: (activations, weights) tuple.
    """
    activations = {}
    save_path = Path(save_dir)

    for pt_file in sorted(save_path.glob("*.pt")):
        name = pt_file.stem
        activations[name] = torch.load(pt_file, map_location='cpu', weights_only=True)
        print(f"  Loaded {name}: shape={list(activations[name].shape)}")

    if load_weights:
        weights = {}
        weights_dir = save_path / "weights"
        if weights_dir.is_dir():
            for pt_file in sorted(weights_dir.glob("*.pt")):
                name = pt_file.stem
                weights[name] = torch.load(pt_file, map_location='cpu', weights_only=True)
                print(f"  Loaded weight {name}: shape={list(weights[name].shape)}")
        return activations, weights

    return activations


def load_multilayer_activations(
    model_name: str = "NousResearch/Meta-Llama-3.1-8B",
    cache_dir: str = None,
    save_dir: str = None,
    device: str = "cuda",
    layer_indices: list = None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Load activations and weights from multiple layers using the full model.

    Uses HuggingFace model with forward pre-hooks to capture the input
    to each linear projection at selected layers. Returns flattened dicts
    with keys like 'L0_q_proj', 'L8_q_proj', etc.

    Args:
        model_name: HuggingFace model ID.
        cache_dir: Where model files are cached.
        save_dir: If provided, save activations and weights as .pt files.
        device: Target device for inference.
        layer_indices: Which layers to capture. Default: 8 evenly spaced.

    Returns:
        (activations, weights) tuple with flattened layer-prefixed keys.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    if cache_dir is None:
        cache_dir = os.environ.get('HF_HOME', '/mnt/data/huggingface')

    config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
    hidden_size = config.hidden_size
    n_layers = config.num_hidden_layers

    if layer_indices is None:
        # 8 evenly spaced layers
        layer_indices = sorted(set(
            int(i * (n_layers - 1) / 7) for i in range(8)
        ))

    print(f"Loading {model_name} for multi-layer activation capture...")
    print(f"  Layers: {layer_indices} (of {n_layers} total)")
    print(f"  Hidden size: {hidden_size}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name, cache_dir=cache_dir,
        torch_dtype=torch.float16, device_map=device,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    tokens = tokenizer(SAMPLE_PROMPT, return_tensors="pt")
    input_ids = tokens["input_ids"].to(device)
    seq_len = input_ids.shape[1]
    print(f"  Tokenized prompt: {seq_len} tokens")

    # Register forward pre-hooks on selected layers
    activations = {}
    hooks = []

    proj_names_attn = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
    proj_names_mlp = ['gate_proj', 'up_proj']

    for layer_idx in layer_indices:
        layer = model.model.layers[layer_idx]

        for proj_name in proj_names_attn:
            module = getattr(layer.self_attn, proj_name)
            key = f"L{layer_idx}_{proj_name}"

            def make_hook(k):
                def hook_fn(mod, inp):
                    activations[k] = inp[0].detach().cpu().float()
                return hook_fn

            hooks.append(module.register_forward_pre_hook(make_hook(key)))

        for proj_name in proj_names_mlp:
            module = getattr(layer.mlp, proj_name)
            key = f"L{layer_idx}_{proj_name}"

            def make_hook(k):
                def hook_fn(mod, inp):
                    activations[k] = inp[0].detach().cpu().float()
                return hook_fn

            hooks.append(module.register_forward_pre_hook(make_hook(key)))

    # Forward pass to capture activations
    print("  Running forward pass...")
    with torch.no_grad():
        model(input_ids)

    for h in hooks:
        h.remove()

    # Extract weights (only those with in_features == hidden_size)
    weights = {}
    for layer_idx in layer_indices:
        layer = model.model.layers[layer_idx]
        for proj_name in proj_names_attn:
            w = getattr(layer.self_attn, proj_name).weight
            key = f"L{layer_idx}_{proj_name}"
            if w.shape[1] == hidden_size:
                weights[key] = w.detach().cpu().float()

        for proj_name in proj_names_mlp:
            w = getattr(layer.mlp, proj_name).weight
            key = f"L{layer_idx}_{proj_name}"
            if w.shape[1] == hidden_size:
                weights[key] = w.detach().cpu().float()

    # Free model
    del model
    torch.cuda.empty_cache()

    print(f"\nCaptured {len(activations)} activation tensors, "
          f"{len(weights)} weight matrices:")
    for name in sorted(activations.keys()):
        t = activations[name]
        print(f"  {name}: shape={list(t.shape)}, "
              f"max={t.abs().max():.2f}, mean={t.abs().mean():.4f}")

    # Save if requested
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        for name, tensor in activations.items():
            path = os.path.join(save_dir, f"{name}.pt")
            torch.save(tensor, path)
        weights_dir = os.path.join(save_dir, "weights")
        os.makedirs(weights_dir, exist_ok=True)
        for name, tensor in weights.items():
            path = os.path.join(weights_dir, f"{name}.pt")
            torch.save(tensor, path)
        meta = {
            'model': model_name,
            'layer_indices': layer_indices,
            'hidden_size': hidden_size,
            'n_layers_total': n_layers,
            'activations': {name: list(t.shape) for name, t in activations.items()},
            'weights': {name: list(t.shape) for name, t in weights.items()},
        }
        with open(os.path.join(save_dir, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"  Saved to {save_dir}")

    return activations, weights


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Capture layer activations")
    parser.add_argument('--model', type=str, default="NousResearch/Meta-Llama-3.1-70B")
    parser.add_argument('--save-dir', type=str, default=None)
    parser.add_argument('--return-weights', action='store_true')
    parser.add_argument('--multilayer', action='store_true',
                        help="Capture from multiple layers (requires full model)")
    parser.add_argument('--layer-indices', type=int, nargs='+', default=None,
                        help="Which layers to capture (default: 8 evenly spaced)")
    cli_args = parser.parse_args()

    os.environ.setdefault('HF_HOME', '/mnt/data/huggingface')
    model_short = cli_args.model.split('/')[-1].lower()

    if cli_args.multilayer:
        save_dir = cli_args.save_dir or f"/mnt/data/activations/{model_short}_multilayer"
        activations, weights = load_multilayer_activations(
            model_name=cli_args.model, save_dir=save_dir,
            layer_indices=cli_args.layer_indices)
        print(f"\nDone. Saved {len(activations)} activation tensors + "
              f"{len(weights)} weight matrices to {save_dir}")
    else:
        save_dir = cli_args.save_dir or f"/mnt/data/activations/{model_short}_layer0"
        result = load_activations(
            model_name=cli_args.model, save_dir=save_dir,
            return_weights=cli_args.return_weights)
        n = len(result[0]) if isinstance(result, tuple) else len(result)
        print(f"\nDone. Saved {n} activation tensors to {save_dir}")
