"""Cost function for NOVA-Quant rotation discovery.

Combines quantization error (MSE, max error) with hardware latency to
produce a single scalar cost that the Evolution Strategy minimizes.

Cost = alpha * quantization_error + beta * latency

The key NOVA insight: optimizing for error alone (SpinQuant/SGD) produces
dense matrices that are slow on hardware. By including latency in the cost,
the ES naturally discovers structured rotations that are both accurate AND fast.
"""

import torch
import time
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

from ..rotated_fp8.fp8_quantize import (
    quantize_fp8_naive,
    dequantize_fp8,
    quantize_int8_naive,
    dequantize_int8,
    quantize_int4_naive,
    dequantize_int4,
    quantize_int4_per_group,
    dequantize_int4_per_group,
    FP8_DTYPE,
    FP8_MAX,
    INT8_MAX,
    INT4_MAX,
)
from .rotation import NOVARotation


@dataclass
class RotationMetrics:
    """Metrics from evaluating a rotation on activations."""
    fp8_mse: float = 0.0
    fp8_max_error: float = 0.0
    fp8_snr_db: float = 0.0
    int8_mse: float = 0.0
    int8_max_error: float = 0.0
    int8_snr_db: float = 0.0
    w4a4_gemm_mse: float = 0.0
    w4a4_gemm_snr_db: float = 0.0
    w4a4_weight_mse: float = 0.0
    w4a4_act_mse: float = 0.0
    latency_ms: float = 0.0
    n_layers: int = 0


def evaluate_rotation_quality(
    rotation: NOVARotation,
    activations: Dict[str, torch.Tensor],
    device: str = 'cuda',
    eval_fp8: bool = True,
    eval_int8: bool = True,
) -> RotationMetrics:
    """Evaluate quantization quality of a rotation on real activations.

    For each activation tensor:
    1. Forward-rotate
    2. Quantize (FP8 and/or INT8)
    3. Dequantize
    4. Inverse-rotate
    5. Measure reconstruction error vs original

    Args:
        rotation: The rotation to evaluate.
        activations: Dict of activation tensors (float32, CPU).
        device: GPU device for computation.
        eval_fp8: Whether to evaluate FP8 quantization.
        eval_int8: Whether to evaluate INT8 quantization.

    Returns:
        RotationMetrics with averaged error metrics.
    """
    metrics = RotationMetrics()
    n = 0

    for name, x_cpu in activations.items():
        x = x_cpu.to(device, dtype=torch.float32)
        signal_power = (x ** 2).mean().item()

        with torch.no_grad():
            x_rot = rotation.forward(x)

            if eval_fp8:
                x_fp8, scale = quantize_fp8_naive(x_rot)
                x_deq = dequantize_fp8(x_fp8, scale)
                x_recon = rotation.inverse(x_deq)
                diff = x - x_recon
                mse = (diff ** 2).mean().item()
                metrics.fp8_mse += mse
                metrics.fp8_max_error += diff.abs().max().item()
                if mse > 0:
                    metrics.fp8_snr_db += 10 * torch.log10(
                        torch.tensor(signal_power / mse)).item()

            if eval_int8:
                x_int8, scale_i = quantize_int8_naive(x_rot)
                x_deq_i = dequantize_int8(x_int8, scale_i)
                x_recon_i = rotation.inverse(x_deq_i)
                diff_i = x - x_recon_i
                mse_i = (diff_i ** 2).mean().item()
                metrics.int8_mse += mse_i
                metrics.int8_max_error += diff_i.abs().max().item()
                if mse_i > 0:
                    metrics.int8_snr_db += 10 * torch.log10(
                        torch.tensor(signal_power / mse_i)).item()

        n += 1

    # Average over layers
    if n > 0:
        metrics.fp8_mse /= n
        metrics.fp8_max_error /= n
        metrics.fp8_snr_db /= n
        metrics.int8_mse /= n
        metrics.int8_max_error /= n
        metrics.int8_snr_db /= n
        metrics.n_layers = n

    return metrics


def evaluate_w4a4_quality(
    rotation: NOVARotation,
    activations: Dict[str, torch.Tensor],
    weights: Dict[str, torch.Tensor],
    device: str = 'cuda',
    weight_group_size: int = 128,
) -> RotationMetrics:
    """Evaluate W4A4 GEMM output error: the true cost of quantization.

    For each (activation, weight) pair with matching proj name:
    1. Compute exact: Y = X @ W^T
    2. Rotate: X' = rotation.forward(X), W' = rotation.forward(W^T)^T
    3. Quantize: X'_q = quant_int4(X'), W'_q = quant_int4_per_group(W')
    4. GEMM: Y_q = dequant(X'_q) @ dequant(W'_q)^T
    5. Error: ||Y - Y_q||

    Weight rotation: W' = W @ R^T. Since R = rotation.forward, we compute
    W' = rotation.forward(W^T)^T, i.e., rotate each column of W, which is
    each row of W^T.

    Args:
        rotation: The rotation to evaluate.
        activations: Dict of activation tensors (float32, CPU).
        weights: Dict of weight matrices (float32, CPU).
        device: GPU device for computation.
        weight_group_size: Group size for per-group weight quantization.

    Returns:
        RotationMetrics with W4A4 fields populated.
    """
    metrics = RotationMetrics()
    n = 0

    for name, x_cpu in activations.items():
        if name not in weights:
            continue

        w_cpu = weights[name]
        x = x_cpu.to(device, dtype=torch.float32)
        w = w_cpu.to(device, dtype=torch.float32)

        # x: (..., in_features), w: (out_features, in_features)
        # Exact GEMM: Y = X @ W^T
        x_flat = x.view(-1, x.shape[-1])  # (tokens, in_features)

        with torch.no_grad():
            y_exact = x_flat @ w.t()  # (tokens, out_features)
            signal_power = (y_exact ** 2).mean().item()

            # Rotate activations: X' = rotation.forward(X) = X @ R^T
            x_rot = rotation.forward(x_flat)

            # Rotate weights: W' = W @ R^T
            # rotation.forward(x) applies the transform to each row of x,
            # which is equivalent to x @ R^T. So rotation.forward(W) = W @ R^T.
            w_rot = rotation.forward(w)  # (out_features, in_features)

            # Quantize activations (per-tensor INT4)
            x_q, x_scale = quantize_int4_naive(x_rot)
            x_deq = dequantize_int4(x_q, x_scale)

            # Quantize weights (per-group INT4)
            in_features = w_rot.shape[1]
            if in_features % weight_group_size == 0:
                w_q, w_scales = quantize_int4_per_group(w_rot, weight_group_size)
                w_deq = dequantize_int4_per_group(w_q, w_scales, weight_group_size)
            else:
                # Fall back to per-tensor if group size doesn't divide evenly
                w_q, w_scale = quantize_int4_naive(w_rot)
                w_deq = dequantize_int4(w_q, w_scale)

            # Quantized GEMM
            y_q = x_deq @ w_deq.t()

            # GEMM output error
            diff = y_exact - y_q
            gemm_mse = (diff ** 2).mean().item()
            metrics.w4a4_gemm_mse += gemm_mse
            if gemm_mse > 0 and signal_power > 0:
                metrics.w4a4_gemm_snr_db += 10 * torch.log10(
                    torch.tensor(signal_power / gemm_mse)).item()

            # Individual component errors for diagnostics
            x_recon = rotation.inverse(x_deq)
            act_diff = x_flat - x_recon
            metrics.w4a4_act_mse += (act_diff ** 2).mean().item()

            w_recon = rotation.inverse(w_deq)
            w_diff = w - w_recon
            metrics.w4a4_weight_mse += (w_diff ** 2).mean().item()

        n += 1

    if n > 0:
        metrics.w4a4_gemm_mse /= n
        metrics.w4a4_gemm_snr_db /= n
        metrics.w4a4_act_mse /= n
        metrics.w4a4_weight_mse /= n
        metrics.n_layers = n

    return metrics


def measure_rotation_latency(
    rotation: NOVARotation,
    x_sample: torch.Tensor,
    n_warmup: int = 20,
    n_measure: int = 100,
) -> float:
    """Measure rotation forward pass latency on MI300X.

    Uses CUDA events for accurate GPU timing. Returns average time in ms.
    """
    x = x_sample.to('cuda', dtype=torch.float32)

    # Warmup
    for _ in range(n_warmup):
        rotation.forward(x)
    torch.cuda.synchronize()

    # Measure
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(n_measure):
        rotation.forward(x)
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / n_measure


# Cache for latency measurements (keyed by structure, not signs)
_latency_cache: Dict[Tuple, float] = {}


def get_cached_latency(
    rotation: NOVARotation,
    x_sample: torch.Tensor,
) -> float:
    """Get latency, caching by rotation structure (block sizes + n_stages).

    Latency depends only on structure (block sizes, n_stages), not on sign
    values. This makes ES much faster since most mutations only change signs.
    """
    key = (rotation.dim, rotation.n_stages, tuple(rotation.block_sizes))
    if key not in _latency_cache:
        _latency_cache[key] = measure_rotation_latency(rotation, x_sample)
    return _latency_cache[key]


def nova_cost(
    rotation: NOVARotation,
    activations: Dict[str, torch.Tensor],
    x_sample: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 0.1,
    target: str = 'int8',
    device: str = 'cuda',
    weights: Optional[Dict[str, torch.Tensor]] = None,
    weight_group_size: int = 128,
) -> Tuple[float, RotationMetrics, float]:
    """Combined NOVA-Quant cost function.

    Cost = alpha * quantization_error + beta * normalized_latency

    The quantization error is scaled so that the baseline random Hadamard
    rotation has a cost of ~1.0 for the error component. Latency is in ms.

    Args:
        rotation: Candidate rotation to evaluate.
        activations: Real activation tensors for error measurement.
        x_sample: Sample tensor for latency measurement.
        alpha: Weight for quantization error.
        beta: Weight for latency (ms).
        target: 'int8', 'fp8', or 'w4a4' — which format to optimize for.
        device: GPU device.
        weights: Weight matrices (required for target='w4a4').
        weight_group_size: Group size for W4A4 per-group weight quantization.

    Returns:
        (cost, metrics, latency_ms)
    """
    if target == 'w4a4':
        assert weights is not None, "weights required for target='w4a4'"
        metrics = evaluate_w4a4_quality(
            rotation, activations, weights, device,
            weight_group_size=weight_group_size,
        )
    else:
        eval_fp8 = (target == 'fp8')
        eval_int8 = (target == 'int8')
        metrics = evaluate_rotation_quality(
            rotation, activations, device,
            eval_fp8=eval_fp8 or True,  # always eval FP8 for reporting
            eval_int8=eval_int8 or True,  # always eval INT8 for reporting
        )

    latency = get_cached_latency(rotation, x_sample)
    metrics.latency_ms = latency

    # Primary error metric
    if target == 'int8':
        error = metrics.int8_mse * 1e6  # Scale to reasonable range
    elif target == 'w4a4':
        error = metrics.w4a4_gemm_mse * 1e6
    else:
        error = metrics.fp8_max_error * 1e3

    cost = alpha * error + beta * latency
    return cost, metrics, latency
