"""FP8 E4M3FNUZ, INT8, and INT4 quantization routines for AMD MI300X.

Uses torch.float8_e4m3fnuz (AMD CDNA3 native, max=240.0), NOT float8_e4m3fn (NVIDIA, max=448).
Provides naive per-tensor, per-channel, and rotated quantization with error metrics.

Key insight: FP8 is a floating-point format with relative precision (~6.25%), so rotation's
benefit shows in max error reduction (3x) and underflow elimination, not MSE. For INT8
(uniform grid), rotation reduces MSE by 10-100x because outliers no longer waste the grid.
For INT4, rotation is even more critical: 16 quantization levels mean every bit is precious,
and outliers cause catastrophic clipping.
"""

import torch
from typing import Dict, Tuple
from .hadamard import random_hadamard_transform, inverse_random_hadamard_transform

# MI300X native FP8 format
FP8_DTYPE = torch.float8_e4m3fnuz
FP8_MAX = 240.0

# INT8 symmetric quantization range
INT8_MAX = 127

# INT4 symmetric quantization range: [-8, 7]
INT4_MAX = 7


def quantize_fp8_naive(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-tensor FP8 quantization. Simplest baseline.

    Args:
        x: Input tensor in float16/float32.

    Returns:
        (x_fp8, scale) where x ≈ x_fp8.float() * scale
    """
    amax = x.abs().amax()
    scale = amax / FP8_MAX
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    x_scaled = x / scale
    x_fp8 = x_scaled.to(FP8_DTYPE)
    return x_fp8, scale


def quantize_fp8_per_channel(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-channel (per-row) FP8 quantization. Better for outlier channels.

    Args:
        x: Input tensor of shape (..., dim).

    Returns:
        (x_fp8, scales) where scales has shape (..., 1).
    """
    amax = x.abs().amax(dim=-1, keepdim=True)
    scales = amax / FP8_MAX
    scales = torch.where(scales > 0, scales, torch.ones_like(scales))
    x_scaled = x / scales
    x_fp8 = x_scaled.to(FP8_DTYPE)
    return x_fp8, scales


def quantize_fp8_rotated(x: torch.Tensor, signs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rotated FP8 quantization: Random Hadamard → per-tensor FP8.

    The rotation spreads outliers across all channels, making per-tensor
    quantization nearly as good as per-channel on the original data.

    Args:
        x: Input tensor with power-of-2 last dim.
        signs: Random sign vector from generate_random_signs().

    Returns:
        (x_fp8, scale, signs) for later dequantization.
    """
    x_rotated = random_hadamard_transform(x, signs)
    x_fp8, scale = quantize_fp8_naive(x_rotated)
    return x_fp8, scale, signs


def dequantize_fp8(x_fp8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize FP8 back to float.

    Args:
        x_fp8: FP8 tensor.
        scale: Scale factor (scalar or per-channel).

    Returns:
        Reconstructed float tensor.
    """
    return x_fp8.float() * scale


def dequantize_fp8_rotated(x_fp8: torch.Tensor, scale: torch.Tensor,
                           signs: torch.Tensor) -> torch.Tensor:
    """Dequantize rotated FP8: dequant → inverse rotation.

    Args:
        x_fp8: FP8 tensor (in rotated space).
        scale: Quantization scale.
        signs: Sign vector used during rotation.

    Returns:
        Reconstructed tensor in original space.
    """
    x_deq = dequantize_fp8(x_fp8, scale)
    return inverse_random_hadamard_transform(x_deq.to(torch.float32), signs.float())


def quantize_int8_naive(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-tensor symmetric INT8 quantization.

    Uses a uniform grid — outliers waste the entire dynamic range,
    making this the worst case for outlier-heavy activations.

    Args:
        x: Input tensor.

    Returns:
        (x_int8, scale) where x ≈ x_int8.float() * scale
    """
    amax = x.abs().amax()
    scale = amax / INT8_MAX
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    x_scaled = (x / scale).round().clamp(-128, 127).to(torch.int8)
    return x_scaled, scale


def quantize_int8_rotated(x: torch.Tensor, signs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rotated INT8 quantization: Random Hadamard → per-tensor INT8.

    This is where rotation truly shines: the uniform INT8 grid benefits
    enormously from the compressed dynamic range after rotation.
    """
    x_rotated = random_hadamard_transform(x, signs)
    x_int8, scale = quantize_int8_naive(x_rotated)
    return x_int8, scale, signs


def dequantize_int8(x_int8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize INT8 back to float."""
    return x_int8.float() * scale


def dequantize_int8_rotated(x_int8: torch.Tensor, scale: torch.Tensor,
                            signs: torch.Tensor) -> torch.Tensor:
    """Dequantize rotated INT8: dequant → inverse rotation."""
    x_deq = dequantize_int8(x_int8, scale)
    return inverse_random_hadamard_transform(x_deq, signs.float())


def quantize_int4_naive(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-tensor symmetric INT4 quantization.

    With only 16 levels ([-8, 7]), outliers waste the entire dynamic range.
    This is the worst case — rotation is essential for INT4.

    Args:
        x: Input tensor.

    Returns:
        (x_int4, scale) where x ≈ x_int4.float() * scale.
        x_int4 is stored as int8 (PyTorch has no int4 dtype).
    """
    amax = x.abs().amax()
    scale = amax / INT4_MAX
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    x_scaled = (x / scale).round().clamp(-8, 7).to(torch.int8)
    return x_scaled, scale


def quantize_int4_per_group(x: torch.Tensor, group_size: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-group symmetric INT4 quantization (standard for W4 weights).

    Reshapes the last dim into groups and quantizes each group independently.
    This is the standard approach for weight quantization (GPTQ, AWQ, etc.).

    Args:
        x: Input tensor with last dim divisible by group_size.
        group_size: Number of elements per quantization group.

    Returns:
        (x_int4, scales) where scales has shape (..., n_groups, 1).
        x_int4 is stored as int8 with values in [-8, 7].
    """
    orig_shape = x.shape
    last_dim = orig_shape[-1]
    assert last_dim % group_size == 0, f"Last dim {last_dim} not divisible by group_size {group_size}"

    # Reshape: (..., last_dim) -> (..., n_groups, group_size)
    x = x.view(*orig_shape[:-1], last_dim // group_size, group_size)

    amax = x.abs().amax(dim=-1, keepdim=True)
    scales = amax / INT4_MAX
    scales = torch.where(scales > 0, scales, torch.ones_like(scales))
    x_scaled = (x / scales).round().clamp(-8, 7).to(torch.int8)

    # Reshape back to original shape
    x_scaled = x_scaled.view(orig_shape)
    return x_scaled, scales


def quantize_int4_per_token(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-token (per-row) symmetric INT4 quantization.

    Standard for activation quantization in W4A4 systems (QuaRot, SpinQuant).
    Each token (row) gets its own scale factor, preventing one outlier token
    from destroying quantization precision for all other tokens.

    Args:
        x: Input tensor of shape (..., dim).

    Returns:
        (x_int4, scales) where scales has shape (..., 1).
        x_int4 is stored as int8 with values in [-8, 7].
    """
    amax = x.abs().amax(dim=-1, keepdim=True)
    scales = amax / INT4_MAX
    scales = torch.where(scales > 0, scales, torch.ones_like(scales))
    x_scaled = (x / scales).round().clamp(-8, 7).to(torch.int8)
    return x_scaled, scales


def dequantize_int4(x_int4: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize per-tensor INT4 back to float."""
    return x_int4.float() * scale


def dequantize_int4_per_token(x_int4: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Dequantize per-token INT4 back to float."""
    return x_int4.float() * scales


def dequantize_int4_per_group(x_int4: torch.Tensor, scales: torch.Tensor, group_size: int = 128) -> torch.Tensor:
    """Dequantize per-group INT4 back to float.

    Args:
        x_int4: INT4 tensor stored as int8, original shape (..., last_dim).
        scales: Per-group scales, shape (..., n_groups, 1).
        group_size: Group size used during quantization.

    Returns:
        Dequantized float tensor with original shape.
    """
    orig_shape = x_int4.shape
    x = x_int4.view(*orig_shape[:-1], -1, group_size).float()
    x = x * scales
    return x.view(orig_shape)


def compute_error_metrics(original: torch.Tensor, reconstructed: torch.Tensor) -> Dict[str, float]:
    """Compute comprehensive error metrics between original and reconstructed tensors.

    Args:
        original: Ground truth tensor.
        reconstructed: Reconstructed tensor after quantize/dequantize.

    Returns:
        Dict with MSE, MAE, max_error, SNR_dB, and relative_error.
    """
    orig = original.float()
    recon = reconstructed.float()
    diff = orig - recon

    mse = (diff ** 2).mean().item()
    mae = diff.abs().mean().item()
    max_error = diff.abs().max().item()

    signal_power = (orig ** 2).mean().item()
    noise_power = mse
    if noise_power > 0:
        snr_db = 10 * torch.log10(torch.tensor(signal_power / noise_power)).item()
    else:
        snr_db = float('inf')

    norm_orig = torch.norm(orig).item()
    norm_diff = torch.norm(diff).item()
    relative_error = norm_diff / norm_orig if norm_orig > 0 else float('inf')

    return {
        'mse': mse,
        'mae': mae,
        'max_error': max_error,
        'snr_db': snr_db,
        'relative_error': relative_error,
    }
