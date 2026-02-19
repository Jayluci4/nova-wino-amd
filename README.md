# NOVA Winograd F(6,3) for AMD MI300X

**Prototype large-tile Winograd convolution in FP16 that works with MIOpen.**

NOVA uses optimized interpolation points to make F(6,3) Winograd numerically stable in half-precision — solving the exact instability that caused AMD, NVIDIA, and every major framework to abandon large-tile Winograd.

MIOpen ships Winograd only at F(2,3). There are no production kernels at larger tile sizes on any AMD GPU, and there never have been. MIOpen’s codebase contains infrastructure for F(4,3) through
F(6,3), but it was abandoned because standard interpolation points are numerically unstable in reduced precision.

This report presents a working F(6,3) Winograd HIP kernel with PyTorch integration that
addresses the numerical stability gap:
• Faster than MIOpen at batch=1: 17% to 57% lower latency across all ResNet-50 layers.
• No accuracy loss: 63.29% top-1 on ImageNetV2 (10K images) vs. 63.15% FP32 baseline.
• No NaN/Inf: Standard F(6,3) produces 221,000 NaN values on the same test. NOVA produces
zero.
• Drop-in replacement: One function call replaces all eligible Conv2d layers in any model.
• Stable Diffusion: 49/49 SD 1.5 UNet convolutions replaced, valid 512×512 images, 0.98× MIOpen
step latency.
• Multiple architectures: Also validated on SDXL (38 layers, 1024×1024) and DenseNet-161 (78
layers, ImageNetV2 accuracy preserved).
The fix is mathematical, not architectural. NOVA selects interpolation points that minimize
condition numbers, bringing the maximum matrix entry from ∼10 down to 2.72, which is within
FP16 dynamic range.

<p align="center">
  <img src="docs/report/figures/latency_b1.png" width="600" alt="NOVA beats MIOpen at batch=1">
</p>

## Quick Start

### Build

```bash
# Requires: AMD GPU (gfx942), ROCm, PyTorch with ROCm support
make
```

### Use

```python
from nova_winograd import replace_conv2d_with_nova

model = torchvision.models.resnet50(pretrained=True).cuda().half()
replace_conv2d_with_nova(model)   # Replaces 13 eligible Conv2d layers
output = model(input_fp16)         # Uses NOVA F(6,3) automatically
```

### Test

```bash
make test    # 11 correctness tests
make bench   # Performance vs MIOpen
```

## Installation

```bash
git clone https://github.com/jayantsh/nova-wino-amd.git
cd nova-wino-amd
pip install -e .
make          # Builds HIP kernel → nova_winograd/lib/libnova_winograd.so
```

### Requirements

- AMD Instinct GPU (tested on MI300X, gfx942)
- ROCm 6.3+ with rocBLAS
- PyTorch 2.x with ROCm support
- hipcc (from ROCm toolkit)

## Architecture

NOVA provides two pipelines selected automatically at runtime:

### Fused MFMA Pipeline (K,C ≤ 64 — conv2_x layers)

Single-kernel: fuses input transform, GEMM, and output transform via `mfma_f32_16x16x16f16` intrinsics. Zero intermediate buffers.

```
Input [B,C,H,W]  →  Fused Kernel (1024 threads, LDS-staged)  →  Output [B,K,H,W]
                     Phase 1: B^T · tile · B  (__shfl → LDS)
                     Phase 2: MFMA accumulate  (LDS → registers)
                     Phase 3: A · M · A^T     (LDS → __shfl → global)
```

### 3-Pass Pipeline (K,C > 64 — conv3_x+ layers)

```
Input [B,C,H,W]  →  Input Transform (HIP)  →  rocBLAS Batched GEMM  →  Output Transform (HIP)  →  Output [B,K,H,W]
                     B^T · tile · B              64× [K,C]·[C,P]          A · M · A^T
                     wave shuffles               FP32 accumulate          wave shuffles
                     zero LDS                    via MFMA                 zero LDS
```

- **Transforms**: Custom HIP kernels, 4 tiles/workgroup (256 threads), register-only via `__shfl`
- **GEMM**: rocBLAS `gemm_strided_batched_ex` — FP16 in, FP32 accumulation, FP16 out
- **Filter transform**: Computed once, cached across forward passes
- **Runtime dispatch**: `NovaWinogradFused` selects fused or 3-pass based on layer dimensions

## API Reference

### Modules

| Class | Use Case |
|-------|----------|
| `NovaWinogradConv2d` | Inference (workspace caching, weight versioning) |
| `NovaWinogradConv2dTrainable` | Training (HIP forward, FP32 native backward) |
| `NovaWinogradConv2dCompilable` | `torch.compile(fullgraph=True)` compatible |

### Functions

| Function | Description |
|----------|-------------|
| `replace_conv2d_with_nova(model)` | Replace all eligible 3x3 s1 Conv2d layers in-place |
| `nova_forward(input_fp16, weight_fp32, padding)` | Functional API for single forward pass |
| `NovaWinogradConv2d.from_conv2d(conv)` | Create from existing nn.Conv2d (copies weights) |

## Repository Structure

```
nova-wino-amd/
├── csrc/
│   ├── nova_winograd.hip          # 3-pass pipeline: transforms + rocBLAS GEMM + C API
│   └── nova_winograd_fused.hip    # Fused MFMA kernel with runtime dispatch + C API
├── nova_winograd/
│   ├── __init__.py                 # Package exports
│   ├── ops.py                      # ctypes bridge to .so
│   ├── conv2d.py                   # nn.Module implementations
│   ├── surgery.py                  # replace_conv2d_with_nova()
│   └── lib/                        # Built .so goes here
├── tests/
│   └── test_correctness.py         # 11 pytest tests
├── benchmarks/
│   ├── bench_layers.py             # Per-layer performance vs MIOpen
│   └── bench_sd_unet.py            # Stable Diffusion UNet demo
├── examples/
│   ├── quickstart.py               # Three usage patterns
│   └── resnet50_inference.py       # ResNet-50 with NOVA
├── docs/
│   └── report/                     # Technical report (LaTeX + PDF)
├── Makefile
├── setup.py
└── README.md
```

## Performance

### Batch=1 (Latency-Critical Inference)

| Layer | MIOpen F(2,3) | NOVA F(6,3) | Speedup |
|-------|:------------:|:-----------:|:-------:|
| conv2_x (64ch, 56x56) | 0.035 ms | **0.029 ms** | 1.2x |
| conv3_x (128ch, 28x28) | 0.036 ms | **0.025 ms** | 1.4x |
| conv4_x (256ch, 14x14) | 0.056 ms | **0.024 ms** | 2.3x |
| conv5_x (512ch, 7x7) | 0.050 ms | **0.026 ms** | 1.9x |

### Batch > 1

The fused MFMA kernel eliminates the batch>1 performance gap for conv2_x layers while saving 60+ MB of workspace memory:

| Layer (B=32) | 3-Pass (original) | Fused MFMA | Speedup | Workspace Saved |
|--------------|:-----------------:|:----------:|:-------:|:---------------:|
| conv2_x (64ch, 56×56) | 0.585 ms | **0.385 ms** | 1.52× | 60.5 MB |

For larger layers (conv3_x+), the 3-pass pipeline with pre-allocated workspace is used automatically via runtime dispatch. MIOpen's fused single-kernel F(2,3) remains competitive at these sizes.

## Technical Report

Full technical report with figures, tables, and analysis:
[`docs/report/main.pdf`](docs/report/main.pdf)

## License

Copyright (c) 2026 Jayant Lohia. All rights reserved. See [LICENSE](LICENSE).

## Citation

If you use NOVA Winograd in your work, please cite the NOVA paper on arXiv.
