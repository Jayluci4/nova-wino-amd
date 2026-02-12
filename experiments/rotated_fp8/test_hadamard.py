"""Unit tests for Walsh-Hadamard transform and rotated FP8 quantization.

Tests:
1. WHT is involution: H(H(x)) == n * x
2. Norm preservation: ||H(x)/sqrt(n)|| == ||x||
3. Random Hadamard roundtrip exactness
4. Rotation gaussianizes outliers (kurtosis → ~3)
5. FP8 quantization basic sanity
6. Rotated FP8 error improvement over naive
"""

import torch
import pytest
import math

from .hadamard import (
    hadamard_transform,
    random_hadamard_transform,
    inverse_random_hadamard_transform,
    generate_random_signs,
)
from .fp8_quantize import (
    quantize_fp8_naive,
    quantize_fp8_per_channel,
    quantize_fp8_rotated,
    dequantize_fp8,
    dequantize_fp8_rotated,
    quantize_int8_naive,
    quantize_int8_rotated,
    dequantize_int8,
    dequantize_int8_rotated,
    compute_error_metrics,
    FP8_DTYPE,
)


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42


class TestHadamardTransform:
    """Tests for the Walsh-Hadamard transform."""

    @pytest.mark.parametrize("n", [2, 4, 8, 16, 64, 256, 1024, 8192])
    def test_involution(self, n):
        """H(H(x)) == n * x — WHT is self-inverse up to scaling."""
        torch.manual_seed(SEED)
        x = torch.randn(4, n, device=DEVICE, dtype=torch.float64)
        hx = hadamard_transform(x)
        hhx = hadamard_transform(hx)
        torch.testing.assert_close(hhx, n * x, rtol=1e-8, atol=1e-8)

    @pytest.mark.parametrize("n", [4, 64, 256, 1024, 8192])
    def test_norm_preservation(self, n):
        """||H(x)/sqrt(n)|| == ||x|| — normalized WHT is orthonormal."""
        torch.manual_seed(SEED)
        x = torch.randn(8, n, device=DEVICE, dtype=torch.float32)
        hx_normalized = hadamard_transform(x) / math.sqrt(n)
        norm_x = torch.norm(x, dim=-1)
        norm_hx = torch.norm(hx_normalized, dim=-1)
        torch.testing.assert_close(norm_x, norm_hx, rtol=1e-4, atol=1e-4)

    def test_non_power_of_2_raises(self):
        """WHT should reject non-power-of-2 dimensions."""
        x = torch.randn(3, 7, device=DEVICE)
        with pytest.raises(ValueError, match="power of 2"):
            hadamard_transform(x)

    @pytest.mark.parametrize("n", [64, 256, 1024, 8192])
    def test_random_hadamard_roundtrip(self, n):
        """random_hadamard_transform followed by inverse recovers original."""
        torch.manual_seed(SEED)
        x = torch.randn(4, n, device=DEVICE, dtype=torch.float32)
        signs = generate_random_signs(n, device=DEVICE, dtype=torch.float32)

        rotated = random_hadamard_transform(x, signs)
        recovered = inverse_random_hadamard_transform(rotated, signs)

        torch.testing.assert_close(recovered, x, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("n", [256, 1024, 8192])
    def test_gaussianization(self, n):
        """Rotation should gaussianize outlier-heavy data (kurtosis → ~3)."""
        torch.manual_seed(SEED)

        # Create data with extreme outliers: most values near 0, a few huge
        x = torch.randn(32, n, device=DEVICE, dtype=torch.float32) * 0.1
        # Inject outliers in specific channels
        outlier_channels = torch.randint(0, n, (n // 32,))
        x[:, outlier_channels] = torch.randn(32, len(outlier_channels), device=DEVICE) * 50.0

        signs = generate_random_signs(n, device=DEVICE, dtype=torch.float32)
        rotated = random_hadamard_transform(x, signs)

        # Compute kurtosis of rotated data (should be close to 3 for Gaussian)
        mean = rotated.mean()
        std = rotated.std()
        kurtosis = ((rotated - mean) ** 4).mean() / (std ** 4)

        # Gaussian kurtosis is 3.0; allow generous tolerance since finite sample
        assert 2.0 < kurtosis.item() < 5.0, \
            f"Kurtosis {kurtosis.item():.2f} not near Gaussian (3.0)"

    def test_batched_shapes(self):
        """WHT should work with arbitrary batch dimensions."""
        torch.manual_seed(SEED)
        x = torch.randn(2, 3, 4, 128, device=DEVICE, dtype=torch.float32)
        hx = hadamard_transform(x)
        assert hx.shape == x.shape
        # Verify involution still holds
        hhx = hadamard_transform(hx)
        torch.testing.assert_close(hhx, 128 * x, rtol=1e-4, atol=1e-4)


class TestFP8Quantization:
    """Tests for FP8 quantization routines."""

    def test_naive_no_nan_inf(self):
        """Naive FP8 quantization should never produce NaN or Inf."""
        torch.manual_seed(SEED)
        x = torch.randn(16, 8192, device=DEVICE, dtype=torch.float16) * 10.0
        x_fp8, scale = quantize_fp8_naive(x)
        recon = dequantize_fp8(x_fp8, scale)
        assert not torch.isnan(recon).any(), "NaN in naive FP8 output"
        assert not torch.isinf(recon).any(), "Inf in naive FP8 output"

    def test_per_channel_no_nan_inf(self):
        """Per-channel FP8 should never produce NaN or Inf."""
        torch.manual_seed(SEED)
        x = torch.randn(16, 8192, device=DEVICE, dtype=torch.float16) * 10.0
        x_fp8, scales = quantize_fp8_per_channel(x)
        recon = dequantize_fp8(x_fp8, scales)
        assert not torch.isnan(recon).any(), "NaN in per-channel FP8 output"
        assert not torch.isinf(recon).any(), "Inf in per-channel FP8 output"

    def test_rotated_no_nan_inf(self):
        """Rotated FP8 should never produce NaN or Inf."""
        torch.manual_seed(SEED)
        x = torch.randn(16, 8192, device=DEVICE, dtype=torch.float32) * 10.0
        signs = generate_random_signs(8192, device=DEVICE, dtype=torch.float32)
        x_fp8, scale, _ = quantize_fp8_rotated(x, signs)
        recon = dequantize_fp8_rotated(x_fp8, scale, signs)
        assert not torch.isnan(recon).any(), "NaN in rotated FP8 output"
        assert not torch.isinf(recon).any(), "Inf in rotated FP8 output"

    def test_fp8_dtype_is_amd_native(self):
        """Verify we're using the AMD-native FP8 format."""
        assert FP8_DTYPE == torch.float8_e4m3fnuz
        # e4m3fnuz max representable value is 240
        torch.manual_seed(SEED)
        x = torch.tensor([240.0], device=DEVICE)
        x_fp8 = x.to(FP8_DTYPE)
        assert x_fp8.float().item() == 240.0

    def test_fp8_rotation_reduces_max_error(self):
        """Rotated FP8 should reduce max error for outlier-heavy data.

        FP8 is a floating-point format with relative precision. Rotation's benefit
        for FP8 shows in max error reduction and underflow prevention, not MSE.
        """
        torch.manual_seed(SEED)

        x = torch.randn(32, 8192, device=DEVICE, dtype=torch.float32) * 0.1
        x[:, :8] = torch.randn(32, 8, device=DEVICE) * 100.0

        signs = generate_random_signs(8192, device=DEVICE, dtype=torch.float32)

        # Naive FP8
        x_fp8_naive, scale_naive = quantize_fp8_naive(x)
        recon_naive = dequantize_fp8(x_fp8_naive, scale_naive)
        metrics_naive = compute_error_metrics(x, recon_naive)

        # Rotated FP8
        x_fp8_rot, scale_rot, _ = quantize_fp8_rotated(x, signs)
        recon_rot = dequantize_fp8_rotated(x_fp8_rot, scale_rot, signs)
        metrics_rot = compute_error_metrics(x, recon_rot)

        # Max error should improve significantly
        max_err_improvement = metrics_naive['max_error'] / metrics_rot['max_error']
        assert max_err_improvement > 2.0, \
            f"Expected >2x max error reduction, got {max_err_improvement:.1f}x " \
            f"(naive={metrics_naive['max_error']:.2f}, rotated={metrics_rot['max_error']:.2f})"

    def test_int8_rotation_reduces_mse(self):
        """Rotated INT8 should dramatically reduce MSE for outlier-heavy data.

        INT8 uses a uniform grid where outliers waste the entire dynamic range.
        Rotation compresses the range, giving massive MSE improvement.
        """
        torch.manual_seed(SEED)

        x = torch.randn(32, 8192, device=DEVICE, dtype=torch.float32) * 0.1
        x[:, :8] = torch.randn(32, 8, device=DEVICE) * 100.0

        signs = generate_random_signs(8192, device=DEVICE, dtype=torch.float32)

        # Naive INT8
        x_int8, scale = quantize_int8_naive(x)
        recon = dequantize_int8(x_int8, scale)
        metrics_naive = compute_error_metrics(x, recon)

        # Rotated INT8
        x_int8_r, scale_r, _ = quantize_int8_rotated(x, signs)
        recon_r = dequantize_int8_rotated(x_int8_r, scale_r, signs)
        metrics_rot = compute_error_metrics(x, recon_r)

        improvement = metrics_naive['mse'] / metrics_rot['mse']
        assert improvement > 10.0, \
            f"Expected >10x MSE improvement, got {improvement:.1f}x " \
            f"(naive={metrics_naive['mse']:.6f}, rotated={metrics_rot['mse']:.6f})"

    def test_error_metrics_zero_error(self):
        """Error metrics should report zeros for identical tensors."""
        x = torch.randn(4, 64, device=DEVICE)
        metrics = compute_error_metrics(x, x)
        assert metrics['mse'] == 0.0
        assert metrics['mae'] == 0.0
        assert metrics['max_error'] == 0.0
        assert metrics['snr_db'] == float('inf')
        assert metrics['relative_error'] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
