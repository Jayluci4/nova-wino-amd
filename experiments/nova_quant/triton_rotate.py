"""Fused Triton kernels for rotation + quantization on MI300X.

Provides three levels of fusion:
1. Element-wise fusion: sign_flip + scale + FP8/INT8 cast (always works)
2. Block-matmul fusion: sign_flip + block_WHT + normalize (block_size <= 128)
3. Butterfly fusion: Kronecker-factored H_{PQ} = H_P ⊗ H_Q via two matmuls
   (128 < block_size <= 16384)

For block_size > 16384, falls back to PyTorch WHT.
The butterfly approach decomposes large Hadamard transforms into two smaller
matmuls that each fit in registers, using tl.dot → MFMA on MI300X.
"""

import torch
import triton
import triton.language as tl
import math
from typing import Tuple

from .rotation import NOVARotation, block_hadamard_transform


# --- Element-wise kernels ---

@triton.jit
def _sign_scale_fp8_kernel(
    x_ptr, out_ptr, signs_ptr, scale_ptr,
    N,
    BLOCK: tl.constexpr,
):
    """Fused sign_flip + FP8 scale + cast.

    After WHT is done in PyTorch, this kernel fuses the remaining
    element-wise operations into a single pass.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N

    x = tl.load(x_ptr + offsets, mask=mask)
    signs = tl.load(signs_ptr + offsets % tl.load(scale_ptr + 1).to(tl.int32), mask=mask)
    scale = tl.load(scale_ptr)

    # Sign flip + scale
    x = x * signs / scale

    tl.store(out_ptr + offsets, x, mask=mask)


@triton.jit
def _fused_sign_normalize_kernel(
    x_ptr, out_ptr, signs_ptr,
    N, inv_sqrt_block: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Fused: x * signs * (1/sqrt(block_size))."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N

    x = tl.load(x_ptr + offsets, mask=mask)
    signs = tl.load(signs_ptr + offsets, mask=mask)

    x = x * signs * inv_sqrt_block

    tl.store(out_ptr + offsets, x, mask=mask)


# --- Block-matmul kernel for small blocks (≤ 128) ---

def _build_hadamard_matrix(n: int) -> torch.Tensor:
    """Build the n×n Hadamard matrix (normalized by 1/sqrt(n)).

    For small n (≤128), precomputing the matrix and using matmul is
    faster than the butterfly algorithm due to better memory access patterns
    and ability to use matrix core instructions (MFMA on MI300X).
    """
    if n == 1:
        return torch.tensor([[1.0]])
    half = _build_hadamard_matrix(n // 2)
    return torch.cat([
        torch.cat([half, half], dim=1),
        torch.cat([half, -half], dim=1),
    ], dim=0) / math.sqrt(2)


# Cache precomputed Hadamard matrices
_H_CACHE = {}


def get_hadamard_matrix(n: int, device: str = 'cuda') -> torch.Tensor:
    """Get cached normalized Hadamard matrix."""
    key = (n, device)
    if key not in _H_CACHE:
        H = _build_hadamard_matrix(n)
        # Undo the recursive normalization (we normalize separately)
        log2n = int(math.log2(n))
        H = H * (math.sqrt(2) ** log2n)  # Back to unnormalized ±1 entries
        _H_CACHE[key] = H.to(device, dtype=torch.float32)
    return _H_CACHE[key]


@triton.jit
def _block_rotate_matmul_kernel(
    x_ptr, out_ptr, signs_ptr, H_ptr,
    M, N, stride_m,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Fused sign_flip + block_hadamard via matmul for small blocks.

    Each program handles BLOCK_M rows × one Hadamard block (BLOCK_N cols).
    Uses tl.dot which maps to AMD MFMA instructions for high throughput.

    Args:
        x_ptr: Input activations (M, N).
        out_ptr: Output buffer (M, N).
        signs_ptr: Sign vector (N,).
        H_ptr: Hadamard matrix (BLOCK_N, BLOCK_N), unnormalized.
        M: Number of rows.
        N: Total columns.
        stride_m: Row stride.
        BLOCK_M: Rows per program (tile height).
        BLOCK_N: Hadamard block size (tile width).
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Row and column ranges for this tile
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Load x tile: (BLOCK_M, BLOCK_N)
    x_offsets = rows[:, None] * stride_m + cols[None, :]
    mask = (rows[:, None] < M) & (cols[None, :] < N)
    x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)

    # Load signs for this block: (BLOCK_N,)
    signs = tl.load(signs_ptr + cols, mask=cols < N, other=1.0).to(tl.float32)
    x = x * signs[None, :]

    # Load Hadamard matrix: (BLOCK_N, BLOCK_N)
    h_rows = tl.arange(0, BLOCK_N)[:, None]
    h_cols = tl.arange(0, BLOCK_N)[None, :]
    H = tl.load(H_ptr + h_rows * BLOCK_N + h_cols).to(tl.float32)

    # Matmul: out = x @ H^T / sqrt(BLOCK_N)
    # x: (BLOCK_M, BLOCK_N), H^T: (BLOCK_N, BLOCK_N)
    out = tl.dot(x, tl.trans(H))
    inv_sqrt = 1.0 / tl.sqrt(BLOCK_N * 1.0)
    out = out * inv_sqrt

    tl.store(out_ptr + x_offsets, out, mask=mask)


# --- Butterfly kernel for large blocks (> 128) ---

@triton.jit
def _butterfly_rotate_kernel(
    x_ptr, out_ptr, signs_ptr,
    H_P_ptr, H_Q_ptr,
    M, N, stride_m,
    P: tl.constexpr,
    Q: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    INVERSE: tl.constexpr,
):
    """Fused sign_flip + block-WHT via Kronecker-factored two-pass matmul.

    Decomposes H_{P*Q} = H_P ⊗ H_Q. In row-major layout:
        H_{PQ} @ vec(X) = vec(H_P @ X @ H_Q)
    where X = reshape(x, (P, Q)).

    Forward:  Z = H_P @ reshape(x * signs, (P, Q)) @ H_Q / sqrt(P*Q)
    Inverse:  Z = (H_P @ reshape(x, (P, Q)) @ H_Q / sqrt(P*Q)) * signs

    Each program handles one row × one Hadamard block.
    Both matmuls use tl.dot → AMD MFMA instructions.

    Args:
        x_ptr: Input activations (M, N).
        out_ptr: Output buffer (M, N).
        signs_ptr: Sign vector (N,).
        H_P_ptr: Row Hadamard matrix (P, P), unnormalized.
        H_Q_ptr: Column Hadamard matrix (Q, Q), unnormalized.
        M: Number of rows.
        N: Total columns.
        stride_m: Row stride.
        P: Row factor of block_size (P >= Q, P * Q = BLOCK_SIZE).
        Q: Column factor of block_size.
        BLOCK_SIZE: Hadamard block size = P * Q.
        INVERSE: 0 for forward (signs before WHT), 1 for inverse (signs after WHT).
    """
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)

    base_col = pid_b * BLOCK_SIZE

    # Build (P, Q) index tile — maps block-linear indices to (row, col) in tile
    p_range = tl.arange(0, P)
    q_range = tl.arange(0, Q)
    tile_offsets = p_range[:, None] * Q + q_range[None, :]  # (P, Q)
    global_cols = base_col + tile_offsets
    global_offsets = pid_m * stride_m + global_cols
    mask = (pid_m < M) & (global_cols < N)

    # Load input tile as (P, Q)
    x = tl.load(x_ptr + global_offsets, mask=mask, other=0.0).to(tl.float32)

    # Forward: apply signs before WHT
    if INVERSE == 0:
        signs = tl.load(signs_ptr + global_cols, mask=global_cols < N, other=1.0).to(tl.float32)
        x = x * signs

    # Load H_Q: (Q, Q) — Hadamard is symmetric so H_Q = H_Q^T
    hq_idx = tl.arange(0, Q)[:, None] * Q + tl.arange(0, Q)[None, :]
    H_Q = tl.load(H_Q_ptr + hq_idx).to(tl.float32)

    # Load H_P: (P, P)
    hp_idx = tl.arange(0, P)[:, None] * P + tl.arange(0, P)[None, :]
    H_P = tl.load(H_P_ptr + hp_idx).to(tl.float32)

    # Phase 1: Y = X @ H_Q  — column transform
    # (P, Q) @ (Q, Q) → (P, Q)
    Y = tl.dot(x, H_Q)

    # Phase 2: Z = H_P @ Y  — row transform
    # (P, P) @ (P, Q) → (P, Q)
    Z = tl.dot(H_P, Y)

    # Normalize by 1/sqrt(block_size)
    inv_sqrt = 1.0 / tl.sqrt(BLOCK_SIZE * 1.0)
    Z = Z * inv_sqrt

    # Inverse: apply signs after WHT
    if INVERSE == 1:
        signs = tl.load(signs_ptr + global_cols, mask=global_cols < N, other=1.0).to(tl.float32)
        Z = Z * signs

    tl.store(out_ptr + global_offsets, Z, mask=mask)


# --- High-level API ---

# Maximum block size for the direct Triton matmul approach
# Limited by register/LDS pressure: 128×128 matrix = 64KB in float32
MAX_TRITON_BLOCK = 128

# Maximum block size for the butterfly (Kronecker-factored) Triton approach
# Both factors must be ≤ MAX_TRITON_BLOCK, so max is MAX_TRITON_BLOCK²
MAX_BUTTERFLY_BLOCK = MAX_TRITON_BLOCK * MAX_TRITON_BLOCK  # 16384

# Tile height for small-block matmul kernel
TILE_M = 16


def _factor_block_size(block_size: int) -> Tuple[int, int]:
    """Factor block_size into (P, Q) for Kronecker decomposition H_{PQ} = H_P ⊗ H_Q.

    Both P and Q are powers of 2 with P >= Q, each ≤ MAX_TRITON_BLOCK.
    This allows the large WHT to be applied as two smaller matmuls:
        result = H_P @ reshape(x, (P, Q)) @ H_Q

    Returns:
        (P, Q) where P * Q = block_size, P >= Q, both powers of 2.
    """
    log2_n = int(math.log2(block_size))
    assert block_size == (1 << log2_n), f"block_size must be power of 2, got {block_size}"

    max_log2 = int(math.log2(MAX_TRITON_BLOCK))  # 7 for MAX=128

    # Split log2 as evenly as possible, with P getting the larger half
    q_log = log2_n // 2
    p_log = log2_n - q_log

    # Clamp P to MAX_TRITON_BLOCK, push remainder to Q
    if p_log > max_log2:
        p_log = max_log2
        q_log = log2_n - p_log

    assert p_log <= max_log2 and q_log >= 4, (
        f"Cannot factor block_size={block_size}: needs P=2^{p_log}, Q=2^{q_log}"
    )

    return (1 << p_log, 1 << q_log)


def triton_block_rotate(
    x: torch.Tensor,
    signs: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Apply one rotation stage using Triton kernel if possible.

    For block_size <= 128: uses fused single-matmul kernel.
    For 128 < block_size <= 16384: uses butterfly (Kronecker-factored) kernel.
    For larger blocks: falls back to PyTorch.

    Args:
        x: Input tensor (..., dim), contiguous.
        signs: Sign vector (dim,).
        block_size: Hadamard block size (must be power of 2).

    Returns:
        Rotated tensor, same shape as x.
    """
    orig_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1]).contiguous()
    M, N = x_2d.shape

    if block_size <= MAX_TRITON_BLOCK and block_size >= 16:
        # Use Triton single-matmul kernel
        H = get_hadamard_matrix(block_size, x.device)
        out = torch.empty_like(x_2d)
        signs_gpu = signs.to(x.device, dtype=torch.float32)

        grid = (triton.cdiv(M, TILE_M), N // block_size)
        _block_rotate_matmul_kernel[grid](
            x_2d, out, signs_gpu, H,
            M, N, N,
            BLOCK_M=TILE_M,
            BLOCK_N=block_size,
        )
        return out.view(orig_shape)
    elif block_size > MAX_TRITON_BLOCK and block_size <= MAX_BUTTERFLY_BLOCK:
        # Use Triton butterfly (Kronecker-factored) kernel
        P, Q = _factor_block_size(block_size)
        H_P = get_hadamard_matrix(P, x.device)
        H_Q = get_hadamard_matrix(Q, x.device)
        out = torch.empty_like(x_2d)
        signs_gpu = signs.to(x.device, dtype=torch.float32)

        grid = (M, N // block_size)
        _butterfly_rotate_kernel[grid](
            x_2d, out, signs_gpu, H_P, H_Q,
            M, N, N,
            P=P, Q=Q, BLOCK_SIZE=block_size,
            INVERSE=0,
        )
        return out.view(orig_shape)
    else:
        # Fallback to PyTorch
        x_signed = x * signs.to(x.device, x.dtype)
        result = block_hadamard_transform(x_signed, block_size)
        return result / math.sqrt(block_size)


def triton_block_rotate_inverse(
    x: torch.Tensor,
    signs: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Inverse of triton_block_rotate.

    Inverse: WHT(x) / sqrt(block_size) * signs
    """
    if block_size <= MAX_TRITON_BLOCK and block_size >= 16:
        # For the inverse, we need WHT first then signs
        # Reuse the kernel with identity signs for WHT, then apply signs
        orig_shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1]).contiguous()
        M, N = x_2d.shape

        H = get_hadamard_matrix(block_size, x.device)
        out = torch.empty_like(x_2d)
        identity_signs = torch.ones(N, device=x.device, dtype=torch.float32)

        grid = (triton.cdiv(M, TILE_M), N // block_size)
        _block_rotate_matmul_kernel[grid](
            x_2d, out, identity_signs, H,
            M, N, N,
            BLOCK_M=TILE_M,
            BLOCK_N=block_size,
        )
        result = out.view(orig_shape)
        return result * signs.to(x.device, x.dtype)
    elif block_size > MAX_TRITON_BLOCK and block_size <= MAX_BUTTERFLY_BLOCK:
        # Use Triton butterfly kernel in inverse mode
        P, Q = _factor_block_size(block_size)
        H_P = get_hadamard_matrix(P, x.device)
        H_Q = get_hadamard_matrix(Q, x.device)
        orig_shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1]).contiguous()
        M, N = x_2d.shape
        out = torch.empty_like(x_2d)
        signs_gpu = signs.to(x.device, dtype=torch.float32)

        grid = (M, N // block_size)
        _butterfly_rotate_kernel[grid](
            x_2d, out, signs_gpu, H_P, H_Q,
            M, N, N,
            P=P, Q=Q, BLOCK_SIZE=block_size,
            INVERSE=1,
        )
        return out.view(orig_shape)
    else:
        result = block_hadamard_transform(x, block_size)
        result = result / math.sqrt(block_size)
        return result * signs.to(x.device, x.dtype)


def benchmark_rotation_methods(
    dim: int = 8192,
    batch: int = 200,
    block_size: int = 128,
    n_warmup: int = 20,
    n_measure: int = 200,
) -> dict:
    """Compare PyTorch vs Triton rotation latency.

    Returns dict with timing results for each method.
    """
    x = torch.randn(batch, dim, device='cuda', dtype=torch.float32)
    signs = (torch.randint(0, 2, (dim,), device='cuda') * 2 - 1).float()

    results = {}

    # PyTorch reference
    for _ in range(n_warmup):
        _ = block_hadamard_transform(x * signs, block_size) / math.sqrt(block_size)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_measure):
        _ = block_hadamard_transform(x * signs, block_size) / math.sqrt(block_size)
    end.record()
    torch.cuda.synchronize()
    results['pytorch_ms'] = start.elapsed_time(end) / n_measure

    # Triton (direct matmul for small blocks, butterfly for large blocks)
    triton_supported = (16 <= block_size <= MAX_TRITON_BLOCK) or \
                       (block_size > MAX_TRITON_BLOCK and block_size <= MAX_BUTTERFLY_BLOCK)
    if triton_supported:
        for _ in range(n_warmup):
            _ = triton_block_rotate(x, signs, block_size)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(n_measure):
            _ = triton_block_rotate(x, signs, block_size)
        end.record()
        torch.cuda.synchronize()
        results['triton_ms'] = start.elapsed_time(end) / n_measure
        results['speedup'] = results['pytorch_ms'] / results['triton_ms']
    else:
        results['triton_ms'] = None
        results['speedup'] = None

    results['block_size'] = block_size
    results['dim'] = dim
    results['batch'] = batch

    return results
