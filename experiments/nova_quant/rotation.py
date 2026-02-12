"""Structured rotation matrices for NOVA-Quant discovery.

Parameterizes rotations as cascaded block-diagonal Hadamard transforms with
learnable sign vectors. This is the search space for Evolution Strategy
discovery of hardware-optimal rotations.

Key insight: A single random Hadamard is already good at spreading outliers.
NOVA-Quant searches for *structured* multi-stage rotations that:
1. Achieve lower quantization error (optimized signs for the activation distribution)
2. Run faster on MI300X (block-diagonal structure, cache-aligned)
3. Use hardware-friendly patterns (block sizes matching CU cache lines)
"""

import torch
import math
import json
import copy
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from ..rotated_fp8.hadamard import hadamard_transform


def block_hadamard_transform(x: torch.Tensor, block_size: int) -> torch.Tensor:
    """Apply block-diagonal Hadamard transform.

    Splits the last dimension into blocks and applies WHT independently to each.
    More cache-friendly than full WHT for large dimensions.

    Args:
        x: Input tensor with last dim divisible by block_size.
        block_size: Size of each Hadamard block (must be power of 2).

    Returns:
        Block-diagonal WHT of x.
    """
    n = x.shape[-1]
    if n % block_size != 0:
        raise ValueError(f"Dim {n} not divisible by block_size {block_size}")
    if block_size & (block_size - 1) != 0:
        raise ValueError(f"block_size must be power of 2, got {block_size}")

    orig_shape = x.shape
    x = x.view(*orig_shape[:-1], n // block_size, block_size)
    x = hadamard_transform(x)
    return x.view(orig_shape)


@dataclass
class RotationStage:
    """A single rotation stage: sign_flip -> block_hadamard -> normalize.

    Forward:  f(x) = WHT_block(signs * x) / sqrt(block_size)
    Inverse:  f^{-1}(y) = signs * WHT_block(y) / sqrt(block_size)

    The inverse follows from WHT being self-inverse: WHT(WHT(x)) = n * x.
    """
    block_size: int
    signs: torch.Tensor  # shape (dim,), values in {-1, +1}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        signs = self.signs.to(x.device, x.dtype)
        x = x * signs
        x = block_hadamard_transform(x, self.block_size)
        return x / math.sqrt(self.block_size)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        signs = self.signs.to(x.device, x.dtype)
        x = block_hadamard_transform(x, self.block_size)
        x = x / math.sqrt(self.block_size)
        return x * signs


class NOVARotation:
    """Multi-stage structured rotation discovered by Evolution Strategy.

    R(x) = Stage_K(Stage_{K-1}(...Stage_1(x)...))

    Each stage applies: sign_flip -> block_hadamard -> normalize.
    Multiple stages with different block sizes create a rich family of
    orthogonal transforms that the ES can optimize.

    Why multi-stage beats single-stage:
    - Two passes at different block sizes mix info more thoroughly
    - Small blocks are cache-friendly; large blocks spread globally
    - Each stage's sign vector is independently optimized
    """

    def __init__(self, dim: int, stages: Optional[List[RotationStage]] = None):
        self.dim = dim
        self.stages = stages or []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for stage in self.stages:
            x = stage.forward(x)
        return x

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        for stage in reversed(self.stages):
            x = stage.inverse(x)
        return x

    @property
    def n_stages(self) -> int:
        return len(self.stages)

    @property
    def block_sizes(self) -> List[int]:
        return [s.block_size for s in self.stages]

    @property
    def total_sign_params(self) -> int:
        return sum(s.signs.numel() for s in self.stages)

    def clone(self) -> 'NOVARotation':
        return copy.deepcopy(self)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize rotation for saving."""
        return {
            'dim': self.dim,
            'n_stages': self.n_stages,
            'stages': [
                {
                    'block_size': s.block_size,
                    'signs': s.signs.cpu().tolist(),
                }
                for s in self.stages
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NOVARotation':
        """Deserialize rotation."""
        stages = [
            RotationStage(
                block_size=s['block_size'],
                signs=torch.tensor(s['signs'], dtype=torch.float32),
            )
            for s in data['stages']
        ]
        return cls(dim=data['dim'], stages=stages)

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path: str) -> 'NOVARotation':
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def __repr__(self):
        blocks = ', '.join(str(b) for b in self.block_sizes)
        return f"NOVARotation(dim={self.dim}, stages={self.n_stages}, blocks=[{blocks}])"


def make_random_hadamard_rotation(dim: int, seed: int = 42) -> NOVARotation:
    """Create the baseline: single-stage full-dimension random Hadamard.

    This is what QuIP# and the rotated_fp8 experiment use.
    NOVA-Quant's goal is to beat this.
    """
    gen = torch.Generator(device='cpu').manual_seed(seed)
    signs = torch.randint(0, 2, (dim,), generator=gen, device='cpu') * 2 - 1
    stage = RotationStage(block_size=dim, signs=signs.float())
    return NOVARotation(dim=dim, stages=[stage])


def make_identity_rotation(dim: int) -> NOVARotation:
    """No rotation (identity transform). Used as a control."""
    return NOVARotation(dim=dim, stages=[])
