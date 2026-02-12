"""Walsh-Hadamard transform for activation rotation.

Implements the iterative butterfly algorithm for WHT in O(n log n).
Used to "smear" outlier activations across all channels before FP8 quantization,
dramatically reducing reconstruction error.
"""

import torch
import math


def hadamard_transform(x: torch.Tensor) -> torch.Tensor:
    """Apply unnormalized Walsh-Hadamard transform along the last dimension.

    Uses the iterative butterfly (fast) algorithm. Requires last dim to be power of 2.

    Args:
        x: Input tensor with power-of-2 last dimension.

    Returns:
        WHT of x along last dimension (unnormalized).
    """
    n = x.shape[-1]
    if n & (n - 1) != 0:
        raise ValueError(f"Last dimension must be power of 2, got {n}")

    result = x.clone()
    h = 1
    while h < n:
        # Reshape to apply butterfly at stride h
        result = result.view(*result.shape[:-1], n // (2 * h), 2, h)
        a = result[..., 0, :]  # even
        b = result[..., 1, :]  # odd
        result = torch.stack([a + b, a - b], dim=-2)
        result = result.view(*x.shape)
        h *= 2

    return result


def random_hadamard_transform(x: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    """Apply randomized Hadamard transform: normalize(WHT(sign_flip(x))).

    The random sign flip followed by WHT produces a near-Gaussian distribution
    from any input distribution, which is ideal for uniform quantization.

    Args:
        x: Input tensor with power-of-2 last dimension.
        signs: Tensor of +1/-1 values, broadcastable to x's last dim.

    Returns:
        Normalized randomized Hadamard transform of x.
    """
    n = x.shape[-1]
    # Apply random sign flip
    x_signed = x * signs
    # Apply WHT
    result = hadamard_transform(x_signed)
    # Normalize so transform is orthonormal (preserves norms)
    return result / math.sqrt(n)


def inverse_random_hadamard_transform(x: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    """Inverse of random_hadamard_transform. WHT is self-inverse up to scaling.

    Since H @ H = n * I, and sign flips are self-inverse:
        inverse = sign_flip(WHT(x) / sqrt(n))

    Args:
        x: Rotated tensor.
        signs: Same sign tensor used in the forward transform.

    Returns:
        Original tensor (within floating point tolerance).
    """
    n = x.shape[-1]
    # Apply WHT (self-inverse up to scale)
    result = hadamard_transform(x)
    # Normalize
    result = result / math.sqrt(n)
    # Undo sign flip
    return result * signs


def generate_random_signs(dim: int, device: torch.device, dtype: torch.dtype = torch.float16,
                          seed: int = 42) -> torch.Tensor:
    """Generate a deterministic random sign vector for the Hadamard rotation.

    Args:
        dim: Dimension (must be power of 2).
        device: Target device.
        dtype: Target dtype.
        seed: Random seed for reproducibility.

    Returns:
        Tensor of shape (dim,) with values in {-1, +1}.
    """
    gen = torch.Generator(device='cpu').manual_seed(seed)
    signs = torch.randint(0, 2, (dim,), generator=gen, device='cpu') * 2 - 1
    return signs.to(device=device, dtype=dtype)
