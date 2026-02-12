"""Tests for NOVA-Quant rotation discovery modules."""

import pytest
import torch
import math

from .rotation import (
    NOVARotation,
    RotationStage,
    block_hadamard_transform,
    make_random_hadamard_rotation,
    make_identity_rotation,
)
from .cost_function import (
    evaluate_rotation_quality,
    measure_rotation_latency,
    nova_cost,
    RotationMetrics,
)
from .es_search import NOVAQuantES, SearchConfig, SearchResult


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DIM = 256  # Small dim for fast tests
SEED = 42


# --- Rotation tests ---

class TestBlockHadamard:
    """Tests for block-diagonal Hadamard transform."""

    def test_involution(self):
        """H(H(x)) = block_size * x  (WHT is self-inverse up to scale)."""
        x = torch.randn(4, DIM, dtype=torch.float64)
        for bs in [16, 64, DIM]:
            y = block_hadamard_transform(x, bs)
            z = block_hadamard_transform(y, bs)
            torch.testing.assert_close(z, x * bs, rtol=1e-10, atol=1e-10)

    def test_full_block_matches_full_wht(self):
        """block_size=dim should match full WHT."""
        from ..rotated_fp8.hadamard import hadamard_transform
        x = torch.randn(4, DIM, dtype=torch.float64)
        full_wht = hadamard_transform(x)
        block_wht = block_hadamard_transform(x, DIM)
        torch.testing.assert_close(full_wht, block_wht, rtol=1e-10, atol=1e-10)

    def test_norm_preservation(self):
        """||H(x)/sqrt(bs)|| ≈ ||x||  (normalized WHT preserves norm)."""
        x = torch.randn(8, DIM, dtype=torch.float64)
        for bs in [16, 64, DIM]:
            y = block_hadamard_transform(x, bs) / math.sqrt(bs)
            x_norms = torch.norm(x, dim=-1)
            y_norms = torch.norm(y, dim=-1)
            torch.testing.assert_close(x_norms, y_norms, rtol=1e-10, atol=1e-10)

    def test_dim_not_divisible_raises(self):
        """Should raise ValueError if dim not divisible by block_size."""
        x = torch.randn(4, 100)  # 100 not divisible by 64
        with pytest.raises(ValueError, match="not divisible"):
            block_hadamard_transform(x, 64)

    def test_non_power_of_two_raises(self):
        """Should raise ValueError if block_size not power of 2."""
        x = torch.randn(4, 256)
        with pytest.raises(ValueError):
            block_hadamard_transform(x, 48)


class TestRotationStage:
    """Tests for single rotation stage."""

    def test_forward_inverse_roundtrip(self):
        """stage.inverse(stage.forward(x)) ≈ x."""
        signs = (torch.randint(0, 2, (DIM,)) * 2 - 1).float()
        stage = RotationStage(block_size=64, signs=signs)
        x = torch.randn(4, DIM, dtype=torch.float32)
        y = stage.forward(x)
        x_recon = stage.inverse(y)
        torch.testing.assert_close(x_recon, x, rtol=1e-4, atol=1e-4)

    def test_forward_changes_tensor(self):
        """Forward should produce a different tensor."""
        signs = (torch.randint(0, 2, (DIM,)) * 2 - 1).float()
        stage = RotationStage(block_size=DIM, signs=signs)
        x = torch.randn(4, DIM)
        y = stage.forward(x)
        assert not torch.allclose(x, y, atol=1e-3)

    def test_norm_preservation_stage(self):
        """Rotation stage should preserve norms."""
        signs = (torch.randint(0, 2, (DIM,)) * 2 - 1).float()
        stage = RotationStage(block_size=64, signs=signs)
        x = torch.randn(8, DIM, dtype=torch.float64)
        # Cast signs to float64 for precision
        stage.signs = signs.double()
        y = stage.forward(x)
        torch.testing.assert_close(
            torch.norm(x, dim=-1), torch.norm(y, dim=-1),
            rtol=1e-10, atol=1e-10,
        )


class TestNOVARotation:
    """Tests for multi-stage rotation."""

    def test_identity_is_noop(self):
        """Identity rotation should return x unchanged."""
        rot = make_identity_rotation(DIM)
        x = torch.randn(4, DIM)
        y = rot.forward(x)
        torch.testing.assert_close(y, x)

    def test_single_stage_roundtrip(self):
        """Single-stage rotation inverse roundtrip."""
        rot = make_random_hadamard_rotation(DIM, seed=SEED)
        x = torch.randn(4, DIM, dtype=torch.float32)
        y = rot.forward(x)
        x_recon = rot.inverse(y)
        torch.testing.assert_close(x_recon, x, rtol=1e-4, atol=1e-4)

    def test_multi_stage_roundtrip(self):
        """Multi-stage rotation inverse roundtrip."""
        stages = []
        for bs in [64, DIM]:
            signs = (torch.randint(0, 2, (DIM,)) * 2 - 1).float()
            stages.append(RotationStage(block_size=bs, signs=signs))
        rot = NOVARotation(dim=DIM, stages=stages)

        x = torch.randn(4, DIM, dtype=torch.float32)
        y = rot.forward(x)
        x_recon = rot.inverse(y)
        torch.testing.assert_close(x_recon, x, rtol=1e-3, atol=1e-3)

    def test_multi_stage_norm_preservation(self):
        """Multi-stage rotation should preserve norms."""
        stages = []
        for bs in [64, DIM]:
            signs = (torch.randint(0, 2, (DIM,)) * 2 - 1).double()
            stages.append(RotationStage(block_size=bs, signs=signs))
        rot = NOVARotation(dim=DIM, stages=stages)

        x = torch.randn(4, DIM, dtype=torch.float64)
        y = rot.forward(x)
        torch.testing.assert_close(
            torch.norm(x, dim=-1), torch.norm(y, dim=-1),
            rtol=1e-8, atol=1e-8,
        )

    def test_serialization_roundtrip(self):
        """to_dict -> from_dict should preserve rotation."""
        rot = make_random_hadamard_rotation(DIM, seed=SEED)
        d = rot.to_dict()
        rot2 = NOVARotation.from_dict(d)

        assert rot2.dim == rot.dim
        assert rot2.n_stages == rot.n_stages
        for s1, s2 in zip(rot.stages, rot2.stages):
            assert s1.block_size == s2.block_size
            torch.testing.assert_close(s1.signs, s2.signs)

    def test_save_load_roundtrip(self, tmp_path):
        """save -> load should preserve rotation."""
        rot = make_random_hadamard_rotation(DIM, seed=SEED)
        path = str(tmp_path / "rot.json")
        rot.save(path)
        rot2 = NOVARotation.load(path)

        x = torch.randn(4, DIM)
        y1 = rot.forward(x)
        y2 = rot2.forward(x)
        torch.testing.assert_close(y1, y2)

    def test_clone_is_independent(self):
        """Cloned rotation should be independent."""
        rot = make_random_hadamard_rotation(DIM, seed=SEED)
        rot2 = rot.clone()
        rot2.stages[0].signs[0] *= -1
        assert rot.stages[0].signs[0] != rot2.stages[0].signs[0]

    def test_properties(self):
        """Test n_stages, block_sizes, total_sign_params."""
        stages = [
            RotationStage(block_size=64, signs=torch.ones(DIM)),
            RotationStage(block_size=DIM, signs=torch.ones(DIM)),
        ]
        rot = NOVARotation(dim=DIM, stages=stages)
        assert rot.n_stages == 2
        assert rot.block_sizes == [64, DIM]
        assert rot.total_sign_params == 2 * DIM


# --- Cost function tests ---

@pytest.fixture
def synthetic_activations():
    """Create synthetic activations with outliers."""
    torch.manual_seed(SEED)
    acts = {}
    for name in ['layer0_q', 'layer0_k']:
        x = torch.randn(32, DIM)
        # Inject outliers
        x[:, 0] *= 20
        x[:, 1] *= 15
        acts[name] = x
    return acts


class TestCostFunction:
    """Tests for cost function and metrics."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_evaluate_rotation_quality(self, synthetic_activations):
        """evaluate_rotation_quality should return valid metrics."""
        rot = make_random_hadamard_rotation(DIM, seed=SEED)
        metrics = evaluate_rotation_quality(rot, synthetic_activations, DEVICE)

        assert metrics.fp8_mse >= 0
        assert metrics.fp8_max_error >= 0
        assert metrics.int8_mse >= 0
        assert metrics.int8_max_error >= 0
        assert metrics.n_layers == 2

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_rotation_reduces_int8_error(self, synthetic_activations):
        """Random Hadamard should reduce INT8 error vs identity."""
        identity = make_identity_rotation(DIM)
        rotated = make_random_hadamard_rotation(DIM, seed=SEED)

        m_id = evaluate_rotation_quality(identity, synthetic_activations, DEVICE)
        m_rot = evaluate_rotation_quality(rotated, synthetic_activations, DEVICE)

        # Rotation should reduce INT8 MSE for outlier-heavy activations
        assert m_rot.int8_mse < m_id.int8_mse, (
            f"Rotation INT8 MSE {m_rot.int8_mse} should be < identity {m_id.int8_mse}"
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_nova_cost_returns_tuple(self, synthetic_activations):
        """nova_cost should return (cost, metrics, latency)."""
        rot = make_random_hadamard_rotation(DIM, seed=SEED)
        x_sample = synthetic_activations['layer0_q'].to(DEVICE, dtype=torch.float32)

        cost, metrics, latency = nova_cost(
            rot, synthetic_activations, x_sample,
            alpha=1.0, beta=0.1, target='int8', device=DEVICE,
        )

        assert isinstance(cost, float)
        assert isinstance(metrics, RotationMetrics)
        assert isinstance(latency, float)
        assert cost > 0
        assert latency > 0


# --- ES search tests ---

class TestESSearch:
    """Tests for Evolution Strategy search."""

    def test_random_rotation_validity(self):
        """Random rotations should be valid."""
        config = SearchConfig(dim=DIM, pop_size=4, n_generations=2, n_restarts=1)
        es = NOVAQuantES(config)

        for i in range(10):
            rot = es.random_rotation(seed=i)
            assert rot.dim == DIM
            assert 1 <= rot.n_stages <= 3
            for stage in rot.stages:
                assert DIM % stage.block_size == 0
                assert stage.signs.shape == (DIM,)
                assert set(stage.signs.unique().tolist()).issubset({-1.0, 1.0})

    def test_mutate_preserves_validity(self):
        """Mutation should produce valid rotations."""
        config = SearchConfig(dim=DIM, pop_size=4, n_generations=2, n_restarts=1)
        es = NOVAQuantES(config)
        rot = es.random_rotation(seed=0)

        for _ in range(20):
            rot = es.mutate(rot)
            assert rot.dim == DIM
            for stage in rot.stages:
                assert DIM % stage.block_size == 0
                assert stage.signs.shape == (DIM,)
                assert set(stage.signs.unique().tolist()).issubset({-1.0, 1.0})

    def test_crossover_produces_valid_child(self):
        """Crossover should produce valid rotation."""
        config = SearchConfig(dim=DIM, pop_size=4, n_generations=2, n_restarts=1)
        es = NOVAQuantES(config)
        a = es.random_rotation(seed=0)
        b = es.random_rotation(seed=1)

        for _ in range(10):
            child = es.crossover(a, b)
            assert child.dim == DIM
            for stage in child.stages:
                assert DIM % stage.block_size == 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_es_reduces_cost(self, synthetic_activations):
        """ES should reduce cost over generations."""
        config = SearchConfig(
            dim=DIM, pop_size=8, n_generations=10, n_restarts=1, seed=SEED,
        )
        es = NOVAQuantES(config)

        x_sample = synthetic_activations['layer0_q'].to(DEVICE, dtype=torch.float32)

        def cost_fn(rotation):
            return nova_cost(
                rotation, synthetic_activations, x_sample,
                alpha=1.0, beta=0.0, target='int8', device=DEVICE,
            )

        result = es.run(cost_fn)

        assert isinstance(result, SearchResult)
        assert result.best_rotation is not None
        assert result.best_cost < float('inf')
        assert result.total_evaluations > 0

        # Check that history shows improvement
        if len(result.history) >= 2:
            assert result.history[-1]['best_cost'] <= result.history[0]['best_cost']

    def test_search_config_defaults(self):
        """SearchConfig should have reasonable defaults."""
        cfg = SearchConfig()
        assert cfg.dim == 8192
        assert cfg.pop_size == 32
        assert cfg.n_generations == 200
        assert cfg.elite_fraction == 0.25
        assert cfg.crossover_prob == 0.7


# --- Triton kernel tests ---

class TestTritonRotate:
    """Tests for Triton rotation kernels."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_triton_matches_pytorch(self):
        """Triton block rotation should match PyTorch reference."""
        from .triton_rotate import triton_block_rotate

        torch.manual_seed(SEED)
        x = torch.randn(8, DIM, device=DEVICE, dtype=torch.float32)
        signs = (torch.randint(0, 2, (DIM,), device=DEVICE) * 2 - 1).float()

        for bs in [16, 32, 64, 128]:
            if DIM % bs != 0:
                continue
            # Triton
            y_triton = triton_block_rotate(x, signs, bs)

            # PyTorch reference
            x_signed = x * signs
            y_ref = block_hadamard_transform(x_signed, bs) / math.sqrt(bs)

            torch.testing.assert_close(
                y_triton, y_ref, rtol=1e-3, atol=1e-3,
                msg=f"Triton mismatch at block_size={bs}",
            )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_triton_inverse_roundtrip(self):
        """triton_block_rotate_inverse(triton_block_rotate(x)) ≈ x."""
        from .triton_rotate import triton_block_rotate, triton_block_rotate_inverse

        torch.manual_seed(SEED)
        x = torch.randn(8, DIM, device=DEVICE, dtype=torch.float32)
        signs = (torch.randint(0, 2, (DIM,), device=DEVICE) * 2 - 1).float()

        for bs in [16, 64, 128]:
            if DIM % bs != 0:
                continue
            y = triton_block_rotate(x, signs, bs)
            x_recon = triton_block_rotate_inverse(y, signs, bs)
            # Tolerance scales with block size due to float32 matmul accumulation
            tol = 1e-2 if bs >= 128 else 1e-3
            torch.testing.assert_close(
                x_recon, x, rtol=tol, atol=tol,
                msg=f"Triton inverse mismatch at block_size={bs}",
            )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_triton_large_block_no_crash(self):
        """block_size > 128 should work (butterfly kernel or fallback)."""
        from .triton_rotate import triton_block_rotate

        x = torch.randn(4, DIM, device=DEVICE, dtype=torch.float32)
        signs = (torch.randint(0, 2, (DIM,), device=DEVICE) * 2 - 1).float()

        # block_size = DIM = 256 > 128 — uses butterfly kernel
        y = triton_block_rotate(x, signs, DIM)
        assert y.shape == x.shape
        assert not torch.isnan(y).any()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_benchmark_runs(self):
        """Benchmark should complete without error."""
        from .triton_rotate import benchmark_rotation_methods

        results = benchmark_rotation_methods(
            dim=DIM, batch=4, block_size=64,
            n_warmup=2, n_measure=5,
        )
        assert 'pytorch_ms' in results
        assert 'triton_ms' in results
        assert results['pytorch_ms'] > 0
        assert results['triton_ms'] > 0
        assert results['speedup'] > 0


# --- Butterfly (Kronecker-factored) kernel tests ---

W4A4_DIM = 256  # Small dim for fast W4A4 tests

BUTTERFLY_DIM = 4096  # Large enough for block_size up to 4096


class TestButterflyRotate:
    """Tests for butterfly (Kronecker-factored) Triton kernel for large blocks."""

    def test_factor_block_size(self):
        """_factor_block_size should produce valid (P, Q) pairs."""
        from .triton_rotate import _factor_block_size, MAX_TRITON_BLOCK

        cases = {
            256: (16, 16),
            512: (32, 16),
            1024: (32, 32),
            2048: (64, 32),
            4096: (64, 64),
            8192: (128, 64),
        }
        for bs, expected in cases.items():
            P, Q = _factor_block_size(bs)
            assert (P, Q) == expected, f"block_size={bs}: got ({P}, {Q}), expected {expected}"
            assert P * Q == bs
            assert P <= MAX_TRITON_BLOCK
            assert Q <= MAX_TRITON_BLOCK

    def test_factor_rejects_non_power_of_2(self):
        """_factor_block_size should reject non-power-of-2 inputs."""
        from .triton_rotate import _factor_block_size

        with pytest.raises(AssertionError):
            _factor_block_size(300)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_butterfly_matches_pytorch(self):
        """Butterfly Triton kernel should match PyTorch WHT for large blocks."""
        from .triton_rotate import triton_block_rotate

        torch.manual_seed(SEED)
        x = torch.randn(8, BUTTERFLY_DIM, device=DEVICE, dtype=torch.float32)
        signs = (torch.randint(0, 2, (BUTTERFLY_DIM,), device=DEVICE) * 2 - 1).float()

        for bs in [256, 512, 1024, 2048, 4096]:
            y_triton = triton_block_rotate(x, signs, bs)

            # PyTorch reference
            x_signed = x * signs
            y_ref = block_hadamard_transform(x_signed, bs) / math.sqrt(bs)

            torch.testing.assert_close(
                y_triton, y_ref, rtol=1e-2, atol=1e-2,
                msg=f"Butterfly mismatch at block_size={bs}",
            )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_butterfly_inverse_roundtrip(self):
        """Butterfly forward + inverse should recover original tensor."""
        from .triton_rotate import triton_block_rotate, triton_block_rotate_inverse

        torch.manual_seed(SEED)
        x = torch.randn(8, BUTTERFLY_DIM, device=DEVICE, dtype=torch.float32)
        signs = (torch.randint(0, 2, (BUTTERFLY_DIM,), device=DEVICE) * 2 - 1).float()

        for bs in [256, 1024, 4096]:
            y = triton_block_rotate(x, signs, bs)
            x_recon = triton_block_rotate_inverse(y, signs, bs)
            torch.testing.assert_close(
                x_recon, x, rtol=2e-2, atol=2e-2,
                msg=f"Butterfly inverse roundtrip failed at block_size={bs}",
            )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_butterfly_norm_preservation(self):
        """Butterfly rotation should preserve vector norms."""
        from .triton_rotate import triton_block_rotate

        torch.manual_seed(SEED)
        x = torch.randn(16, BUTTERFLY_DIM, device=DEVICE, dtype=torch.float32)
        signs = (torch.randint(0, 2, (BUTTERFLY_DIM,), device=DEVICE) * 2 - 1).float()

        for bs in [256, 1024, 4096]:
            y = triton_block_rotate(x, signs, bs)
            x_norms = torch.norm(x, dim=-1)
            y_norms = torch.norm(y, dim=-1)
            torch.testing.assert_close(
                x_norms, y_norms, rtol=1e-2, atol=1e-2,
                msg=f"Butterfly norm not preserved at block_size={bs}",
            )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_butterfly_benchmark(self):
        """Benchmark should complete for large blocks via butterfly kernel."""
        from .triton_rotate import benchmark_rotation_methods

        results = benchmark_rotation_methods(
            dim=BUTTERFLY_DIM, batch=4, block_size=256,
            n_warmup=2, n_measure=5,
        )
        assert results['triton_ms'] is not None
        assert results['triton_ms'] > 0
        assert results['speedup'] > 0


# --- W4A4 quantization tests ---

class TestW4A4:
    """Tests for W4A4 (4-bit weights + 4-bit activations) quantization."""

    def test_int4_quantize_range(self):
        """INT4 values should be in [-8, 7]."""
        from ..rotated_fp8.fp8_quantize import quantize_int4_naive

        torch.manual_seed(SEED)
        x = torch.randn(32, W4A4_DIM) * 10.0
        x_q, scale = quantize_int4_naive(x)

        assert x_q.dtype == torch.int8
        assert x_q.min().item() >= -8
        assert x_q.max().item() <= 7
        assert scale.item() > 0

    def test_int4_per_group_shapes(self):
        """Per-group quantization should produce correct scale shapes."""
        from ..rotated_fp8.fp8_quantize import quantize_int4_per_group

        torch.manual_seed(SEED)
        group_size = 64
        x = torch.randn(16, W4A4_DIM)
        x_q, scales = quantize_int4_per_group(x, group_size=group_size)

        assert x_q.shape == x.shape
        assert x_q.dtype == torch.int8
        n_groups = W4A4_DIM // group_size
        assert scales.shape == (16, n_groups, 1)
        assert x_q.min().item() >= -8
        assert x_q.max().item() <= 7

    def test_int4_roundtrip_error(self):
        """Dequant(quant(x)) should have bounded error."""
        from ..rotated_fp8.fp8_quantize import (
            quantize_int4_naive, dequantize_int4,
            quantize_int4_per_group, dequantize_int4_per_group,
        )

        torch.manual_seed(SEED)
        x = torch.randn(32, W4A4_DIM)

        # Per-tensor roundtrip
        x_q, scale = quantize_int4_naive(x)
        x_recon = dequantize_int4(x_q, scale)
        mse = ((x - x_recon) ** 2).mean().item()
        assert mse > 0, "INT4 should have non-zero error"
        assert not torch.isnan(x_recon).any()
        assert not torch.isinf(x_recon).any()

        # Per-group roundtrip
        group_size = 64
        x_q_g, scales = quantize_int4_per_group(x, group_size=group_size)
        x_recon_g = dequantize_int4_per_group(x_q_g, scales, group_size=group_size)
        mse_g = ((x - x_recon_g) ** 2).mean().item()
        assert mse_g > 0
        # Per-group should be better than per-tensor
        assert mse_g < mse, (
            f"Per-group MSE {mse_g} should be < per-tensor MSE {mse}"
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_w4a4_rotation_reduces_gemm_error(self):
        """Rotation should reduce W4A4 GEMM output error vs no rotation."""
        from .cost_function import evaluate_w4a4_quality

        torch.manual_seed(SEED)
        # Create activations with outliers
        acts = {}
        weights = {}
        for name in ['layer0_q', 'layer0_k']:
            x = torch.randn(32, W4A4_DIM) * 0.1
            x[:, 0] *= 50  # Strong outlier channels
            x[:, 1] *= 30
            acts[name] = x
            weights[name] = torch.randn(W4A4_DIM, W4A4_DIM) * (2.0 / W4A4_DIM) ** 0.5

        identity = make_identity_rotation(W4A4_DIM)
        rotated = make_random_hadamard_rotation(W4A4_DIM, seed=SEED)

        m_id = evaluate_w4a4_quality(identity, acts, weights, DEVICE, weight_group_size=64)
        m_rot = evaluate_w4a4_quality(rotated, acts, weights, DEVICE, weight_group_size=64)

        assert m_rot.w4a4_gemm_mse < m_id.w4a4_gemm_mse, (
            f"Rotated W4A4 GEMM MSE {m_rot.w4a4_gemm_mse} should be < "
            f"identity {m_id.w4a4_gemm_mse}"
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_w4a4_cost_returns_valid(self):
        """nova_cost with target='w4a4' should return valid results."""
        torch.manual_seed(SEED)
        acts = {'layer0_q': torch.randn(16, W4A4_DIM)}
        weights = {'layer0_q': torch.randn(W4A4_DIM, W4A4_DIM) * 0.01}
        rot = make_random_hadamard_rotation(W4A4_DIM, seed=SEED)
        x_sample = acts['layer0_q'].to(DEVICE, dtype=torch.float32)

        cost, metrics, latency = nova_cost(
            rot, acts, x_sample,
            alpha=1.0, beta=0.1, target='w4a4', device=DEVICE,
            weights=weights, weight_group_size=64,
        )

        assert isinstance(cost, float)
        assert cost > 0
        assert metrics.w4a4_gemm_mse >= 0
        assert not math.isnan(metrics.w4a4_gemm_mse)
        assert latency > 0

    def test_int4_per_token_shapes_and_range(self):
        """Per-token INT4 should produce per-row scales and values in [-8, 7]."""
        from ..rotated_fp8.fp8_quantize import (
            quantize_int4_per_token, dequantize_int4_per_token,
        )

        torch.manual_seed(SEED)
        x = torch.randn(32, W4A4_DIM)
        # Add outlier to one token — should not affect other tokens' scales
        x[0] *= 100.0

        x_q, scales = quantize_int4_per_token(x)

        assert x_q.shape == x.shape
        assert x_q.dtype == torch.int8
        assert x_q.min().item() >= -8
        assert x_q.max().item() <= 7
        assert scales.shape == (32, 1), f"Expected (32, 1), got {scales.shape}"

        # Outlier token should have much larger scale
        assert scales[0].item() > 10 * scales[1].item(), (
            "Outlier token's scale should be much larger than normal tokens"
        )

        # Roundtrip: per-token should be better than per-tensor for outlier-heavy data
        from ..rotated_fp8.fp8_quantize import quantize_int4_naive, dequantize_int4

        x_recon_pt = dequantize_int4_per_token(x_q, scales)
        mse_per_token = ((x - x_recon_pt) ** 2).mean().item()

        x_q_naive, s_naive = quantize_int4_naive(x)
        x_recon_naive = dequantize_int4(x_q_naive, s_naive)
        mse_per_tensor = ((x - x_recon_naive) ** 2).mean().item()

        assert mse_per_token < mse_per_tensor, (
            f"Per-token MSE {mse_per_token} should be < per-tensor MSE {mse_per_tensor}"
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_weight_rotation_correctness(self):
        """W @ R^T applied via rotation.forward should match direct matmul."""
        torch.manual_seed(SEED)
        rot = make_random_hadamard_rotation(W4A4_DIM, seed=SEED)

        # Build explicit rotation matrix by applying rotation to identity rows
        # rotation.forward(x) = x @ R^T, so rotation.forward(I) = R^T
        I = torch.eye(W4A4_DIM, device=DEVICE)
        RT_matrix = rot.forward(I)  # R^T

        W = torch.randn(128, W4A4_DIM, device=DEVICE)

        # Method 1: rotation.forward(W) — computes W @ R^T
        w_rot_1 = rot.forward(W)

        # Method 2: Direct matmul W @ R^T
        w_rot_2 = W @ RT_matrix

        torch.testing.assert_close(
            w_rot_1, w_rot_2, rtol=1e-3, atol=1e-3,
            msg="rotation.forward(W) should equal W @ R^T",
        )
