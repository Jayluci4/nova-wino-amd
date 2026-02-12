# NOVA-Quant Discovery Sprint: Results & Analysis

**Hardware**: AMD Instinct MI300X (192GB HBM3, 5.3 TB/s)
**Model**: Meta-Llama-3.1-70B, layer 0 activations (q/k/v/o/gate/up_proj, dim=8192)
**Software**: PyTorch 2.9.1+ROCm 6.3, Triton

## 1. Background

Standard LLM quantization (FP8/INT8) fails on activations with outliers — a few channels
fire at 100x the mean, forcing either clipping (loss of large values) or precision loss
(underflow of small values). Rotation before quantization "smears" outliers across all
channels, gaussianizing the distribution so that a uniform quantization grid works well.

The NOVA-Quant discovery sprint tests whether Evolution Strategies (the same technique
behind NOVA's Winograd point discovery) can find **structured rotations** that outperform
random Hadamard baselines while remaining hardware-friendly on MI300X.

## 2. What Was Built

Seven modules in `experiments/nova_quant/`:

| Module | Purpose |
|--------|---------|
| `rotation.py` | Cascaded block-diagonal Hadamard parameterization with learnable sign vectors |
| `cost_function.py` | Combined cost: `alpha * quant_error + beta * latency`, with structure-aware latency caching |
| `es_search.py` | Evolution Strategy: sign-flip mutation, block-size mutation, stage add/remove, uniform crossover |
| `triton_rotate.py` | Fused Triton kernels: sign-flip + block-Hadamard via `tl.dot` (maps to MFMA on MI300X) |
| `run_discovery.py` | CLI entry point: baselines, ES search, comparison tables, JSON output |
| `test_nova_quant.py` | 41 unit tests covering correctness, roundtrips, Triton matching, ES validity, W4A4, per-token INT4 |
| `__init__.py` | Package |

All 41 tests pass (34 original + 6 W4A4 + 1 per-token INT4). Builds on `experiments/rotated_fp8/` (the validation experiment).

## 3. Results

### 3.1 The Dominant Finding: Rotation Itself Is the Win

From the rotated_fp8 validation experiment on real Llama-3.1-70B activations:

| Method | INT8 MSE | INT8 SNR | FP8 Max Error |
|--------|----------|----------|---------------|
| Naive (no rotation) | 7.76e-6 | 18.8 dB | 0.084 |
| Rotated (random Hadamard) | 1.10e-7 | 37.4 dB | 0.005 |
| **Improvement** | **72.7x** | **+18.6 dB** | **15.2x** |

The activation outlier statistics explain why. Llama-3.1-70B layer 0:
- q/k/v_proj kurtosis: **3,788** (Gaussian = 3)
- Max/mean ratio: **18.8x**
- Dynamic range: **212 dB**

Rotation collapses kurtosis toward 3, making the distribution quantization-friendly.

### 3.2 NOVA-Discovered vs. Random Hadamard

The ES searched over structured rotations (1-3 stages of block-diagonal Hadamard with
optimized sign vectors). Two search configurations were run:

**Single-phase** (100 gen, 3 restarts, pop=32, 7,296 evals, 105s):

| Method | INT8 MSE | INT8 SNR | Latency |
|--------|----------|----------|---------|
| Random Hadamard (full dim) | 1.10e-7 | 37.4 dB | 0.457ms |
| **NOVA-Discovered (block-4096)** | **0.80e-7** | **38.5 dB** | **0.442ms** |
| Improvement | 1.3x | +1.0 dB | 3% faster |

**Two-phase** (structure exploration → sign fine-tuning, 16,440 evals, 232s):

| Method | INT8 MSE | INT8 SNR | Latency |
|--------|----------|----------|---------|
| Random Hadamard (full dim) | 1.10e-7 | 37.4 dB | 0.447ms |
| **NOVA-Discovered (block-4096)** | **0.80e-7** | **38.5 dB** | **0.447ms** |
| Improvement | 1.4x | +1.0 dB | same |

Both converge to the same structure: **single-stage block-4096 Hadamard with optimized
signs**. The ES explores multi-stage configurations but consistently rejects them — the
latency cost of additional stages outweighs the marginal quality gain.

### 3.3 Block Size Spectrum

The discovery reveals a clear quality-vs-speed tradeoff:

| Rotation | INT8 MSE | INT8 SNR | PyTorch Latency |
|----------|----------|----------|-----------------|
| No rotation | 7.76e-6 | 18.8 dB | 0.000ms |
| Block-128 (random signs) | 4.40e-7 | 31.4 dB | 0.284ms |
| Block-4096 (NOVA signs) | 0.80e-7 | 38.5 dB | 0.447ms |
| Full-8192 (random signs) | 1.10e-7 | 37.4 dB | 0.457ms |

Block-4096 with optimized signs beats full-8192 with random signs. The ES found that
the sweet spot is large-but-not-full blocks, where the Hadamard still mixes information
across most channels but avoids the overhead of a full-dimension butterfly.

### 3.4 Triton Kernel Performance

Three kernel tiers, all using `tl.dot` → AMD MFMA instructions:

**Tier 1: Direct matmul** (block_size ≤ 128) — precomputed H matrix fits in LDS:

| Block Size | PyTorch | Triton | Speedup |
|-----------|---------|--------|---------|
| 16 | 0.322ms | 0.025ms | 12.7x |
| 32 | 0.358ms | 0.022ms | 16.5x |
| 64 | 0.379ms | 0.089ms | 4.3x |
| 128 | 0.386ms | 0.028ms | **13.9x** |

**Tier 2: Butterfly (Kronecker-factored)** (128 < block_size ≤ 16384):

Decomposes H_{PQ} = H_P ⊗ H_Q using the Kronecker product identity:
`H_{PQ} @ vec(X) = vec(H_P @ X @ H_Q)`. Two smaller matmuls replace one large one.

| Block Size | Factorization | PyTorch | Triton | Speedup |
|-----------|--------------|---------|--------|---------|
| 256 | 16 × 16 | 0.555ms | 0.032ms | 17.4x |
| 512 | 32 × 16 | 0.658ms | 0.023ms | 28.0x |
| 1024 | 32 × 32 | 0.546ms | 0.023ms | 23.8x |
| 2048 | 64 × 32 | 0.757ms | 0.029ms | 26.2x |
| **4096** | **64 × 64** | **0.804ms** | **0.038ms** | **21.2x** |
| 8192 | 128 × 64 | 0.771ms | 0.026ms | **29.8x** |

The butterfly kernel **closes the gap** identified in §3.3. The ES-optimal block-4096
rotation now runs at 0.038ms via Triton (previously 0.804ms via PyTorch fallback) —
a **21x speedup** that makes the optimal rotation deployable.

### 3.5 Revised Deployment Analysis

With the butterfly kernel, the rotation cost drops dramatically:

| Configuration | Per-layer latency | 80-layer overhead | % of batch-1 forward |
|--------------|-------------------|-------------------|----------------------|
| Block-4096, PyTorch (old) | 0.804ms | 64.3ms | 129% |
| **Block-4096, Triton butterfly** | **0.038ms** | **3.0ms** | **6%** |
| Block-8192, Triton butterfly | 0.026ms | 2.1ms | 4% |

At 3ms total overhead for block-4096 across 80 layers, the rotation cost is negligible
even at batch size 1 (~50ms forward pass → 6% overhead). This changes the deployment
calculus: the **full-quality rotation is now free enough for production use**.

## 4. Interpretation

### What the numbers mean

The results reveal a **three-tier hierarchy of impact**:

1. **Rotation vs. no rotation: 72.7x** (INT8 MSE). This is the fundamental physics —
   gaussianizing outlier-heavy activations before uniform quantization. Any orthogonal
   rotation captures most of this gain.

2. **Sign optimization: 1.4x** (NOVA vs random Hadamard). The specific sign vector
   matters, but the marginal improvement is modest. The Walsh-Hadamard structure already
   does most of the mixing work regardless of signs; the signs provide fine-tuning.

3. **Block size selection: 4x** (block-128 vs. block-4096). Larger blocks mix information
   more completely. Block-128 loses 7 dB of INT8 SNR compared to block-4096 because
   outliers in one 128-element block can't be spread to other blocks.

### The resolved tension

The ES found block-4096 optimal, and the butterfly Triton kernel now accelerates it with a
21x speedup (0.038ms vs 0.804ms PyTorch). The Kronecker factorization H_{4096} = H_{64} ⊗ H_{64}
decomposes the large WHT into two 64×64 matmuls that map directly to MFMA instructions,
with both factor matrices fitting comfortably in registers.

### What this means for deployment

The rotation step adds ~0.038ms per layer with the butterfly kernel. For a 70B model with
80 layers, that's ~3ms added to every forward pass — negligible overhead at any batch size.
Combined with the NOVA sign optimization (+1.0 dB over random), the full pipeline is now
production-ready: high quality (38.5 dB INT8 SNR) at near-zero cost.

## 5. W4A4 Pivot: From FP8 to INT4

### 5.1 Motivation

Research revealed that rotation matters most at **INT4, not FP8** — FP8's floating-point
exponent naturally absorbs outliers. The real pain is W4A4 (4-bit weights + 4-bit activations),
the hardest quantization regime where INT4 has only 16 values ([-8, 7]), making every bit
precious. This is where QuaRot, SpinQuant, and ButterflyQuant focus their efforts.

The correct metric for W4A4 is **GEMM output error**, not standalone quantization error:

```
Y   = X @ W^T                                              (exact)
Y_q = dequant(quant_4bit(X @ R)) @ dequant(quant_4bit(W @ R^T))^T   (quantized)
cost = ||Y - Y_q||²                                        (what we minimize)
```

This captures the full QuaRot pipeline: W' = WR^T (offline, free) and X' = XR (online, costs latency).

### 5.2 Implementation

Extended the NOVA-Quant framework with W4A4 support:

| File | Change |
|------|--------|
| `../rotated_fp8/fp8_quantize.py` | Added INT4 quantize/dequantize (per-tensor and per-group with group_size=128) |
| `../rotated_fp8/load_activations.py` | Extended to return weight matrices alongside activations |
| `cost_function.py` | Added `evaluate_w4a4_quality()` — GEMM output error evaluation |
| `run_discovery.py` | Added `--target w4a4`, `--weight-group-size`, synthetic weight generation |
| `test_nova_quant.py` | Added 6 W4A4 tests (INT4 range, per-group shapes, roundtrip, GEMM error, cost, weight rotation correctness) |

Total: 40 tests pass (34 existing + 6 new).

### 5.3 W4A4 Results on Real Llama-3.1-70B Activations

**Search config**: 200 generations × 5 restarts × pop-size 32 = 24,160 evaluations, 4,725s (79 min)
on MI300X. 6 projection layers (q/k/v/o/gate/up_proj, dim=8192) with real weights (up to 28672×8192).

| Method | W4A4 GEMM MSE | GEMM SNR | Act MSE | Wgt MSE | Latency |
|--------|--------------|----------|---------|---------|---------|
| No Rotation | 0.00705 | 6.8 dB | 1.54e-4 | 8.75e-5 | 0.000ms |
| Random Hadamard (QuaRot) | 0.01289 | 4.2 dB | 3.62e-5 | 2.40e-4 | 2.392ms |
| Block-128 Hadamard | 0.00594 | 2.8 dB | 7.08e-5 | 6.26e-5 | 2.118ms |
| **NOVA-Discovered (block-256)** | **0.00473** | **3.6 dB** | **6.29e-5** | **9.27e-5** | **0.309ms** |

**Improvement over QuaRot baseline**: 2.7× lower GEMM MSE, 7.7× faster (0.31ms vs 2.39ms).
**Improvement over no rotation**: 1.5× lower GEMM MSE while adding only 0.31ms latency.
**Improvement over block-128**: 1.3× lower GEMM MSE, 6.9× faster.

### 5.4 Critical Finding: QuaRot Hurts at W4A4

The most surprising result: **full-dimension random Hadamard (QuaRot) is worse than no rotation
at W4A4**. GEMM MSE 0.01289 vs 0.00705 — nearly 2× worse.

The mechanism is clear from the component errors:
- **Activation MSE**: QuaRot wins (3.62e-5 vs 1.54e-4) — rotation gaussianizes activations as expected.
- **Weight MSE**: QuaRot loses badly (2.40e-4 vs 8.75e-5) — full-dimension rotation destroys
  weight structure, and with only 16 INT4 levels, the damage is catastrophic.

The GEMM error is dominated by **weight quantization noise amplified through the matrix multiply**.
At INT8 (256 levels), the weight damage from rotation is tolerable and the activation benefit
dominates. At INT4 (16 levels), weight damage overwhelms activation benefit.

This explains why NOVA converged to **block-256** — large enough to smooth activation outliers
across groups of 256 channels, small enough to preserve weight structure within blocks.

### 5.5 W4A4 Search Convergence

| Restart | Structure | Final Cost | Notes |
|---------|-----------|------------|-------|
| R1 | block-128 | 5141 | Converged by gen 140 |
| R2 | block-256 | 4805 | Found block-256 superior |
| R3 | block-128 | 5101 | Block-128 ceiling hit |
| R4 | block-128 | 5170 | Consistent block-128 range |
| R5 | **block-256** | **4730** | **Global best** |

Block-256 consistently outperforms block-128 by ~8%. The ES never converged to block sizes
larger than 256, confirming that the weight-damage tradeoff caps the useful block size at W4A4.

### 5.6 Comparison: INT8 vs W4A4 Optimal Structures

| Metric | INT8 Optimal | W4A4 Optimal | Explanation |
|--------|-------------|-------------|-------------|
| Block size | 4096 | 256 | W4A4 penalizes weight rotation much more |
| Sign optimization gain | 1.4× | ~1.3× | Similar marginal benefit |
| Rotation vs identity | 72.7× | 1.5× | INT8 has headroom; W4A4 weight damage limits gain |
| QuaRot (full-dim) | Best baseline | Worst baseline | Full rotation destroys weights at INT4 |

The INT8-optimal and W4A4-optimal rotations are **fundamentally different**. This validates the
NOVA approach of target-aware search — a rotation optimized for INT8 would be harmful at W4A4.

## 6. Multi-Layer Validation

### 6.1 Full-Model Perplexity (WikiText-2)

Applied each rotation strategy to ALL 480 eligible linear layers of Llama-3.1-70B and measured
WikiText-2 perplexity (288,937 tokens, sliding window, stride=2048):

| Config | PPL | vs QuaRot |
|--------|-----|-----------|
| FP16 baseline | **2.79** | — |
| W4A4 no rotation | 128,386,408 | 8,172× worse |
| W4A4 NOVA (block-256, uniform) | 22,143 | 1.4× worse |
| W4A4 QuaRot (block-8192) | 15,713 | baseline |
| **W4A4 adaptive** (skip L0, block-4096) | **2,522** | **6.2× better** |

**Surprise #1**: NOVA block-256 (uniform across all layers) has *worse* perplexity than QuaRot,
despite having 2.7× lower GEMM MSE on layer 0. The single-layer metric does not predict multi-layer
behavior.

**Surprise #2**: A simple adaptive strategy — skip rotation at layer 0, use block-4096 everywhere
else — delivers **6.2× lower perplexity than QuaRot**. This insight came directly from the per-layer
analysis below.

All W4A4 PPL numbers above use per-tensor activation quantization (one scale for the entire activation
matrix) — unrealistically coarse. Section 6.6 shows the dramatic improvement with per-token A4.
The *relative ordering* reveals fundamental dynamics about how rotation interacts with depth.

### 6.2 Per-Layer Universality Analysis

Captured real activations and weights at layers [0, 10, 20, 40, 60, 79], swept block sizes
[128, 256, 512, 1024, 4096, 8192] with random signs at each layer:

| Layer | Best Block | Best MSE | Identity MSE | Improvement |
|-------|-----------|----------|-------------|-------------|
| 0 | **identity** | 0.005440 | 0.005440 | 1.0x |
| 10 | **4096** | 0.002327 | 0.084802 | 36.4x |
| 20 | **8192** | 0.004411 | 0.205801 | 46.7x |
| 40 | **4096** | 0.006467 | 0.215420 | 33.3x |
| 60 | **4096** | 0.007868 | 0.184075 | 23.4x |
| 79 | **8192** | 0.005093 | 1.159592 | 227.7x |

**Finding: Block-256 is NOT universal.** Layer 0 prefers identity (no rotation); all other layers
prefer large blocks (4096-8192). The ES search on layer 0 was misled by an extreme outlier layer.

**The mechanism — weight damage sensitivity**:

| Layer | Wgt MSE (identity) | Wgt MSE (block-8192) | Damage Ratio |
|-------|-------------------|---------------------|-------------|
| 0 | 8.75e-5 | 2.39e-4 | **2.7×** |
| 10 | 3.46e-6 | 3.04e-6 | 0.9× |
| 40 | 3.76e-6 | 3.52e-6 | 0.9× |
| 79 | 3.97e-6 | 3.57e-6 | 0.9× |

Layer 0 weights are 25× more sensitive to rotation than deeper layers. Rotation triples layer 0
weight MSE while barely affecting layers 10+. For 79 of 80 layers, rotation is free for weights
and only activation smoothing matters — favoring maximum mixing (large blocks).

### 6.3 Transferability of Signs

Tested layer-0 NOVA rotation (optimized signs, block-256) on all layers vs random signs (block-256):

| Layer | Random-256 MSE | NOVA-256 MSE | Sign Benefit |
|-------|---------------|-------------|-------------|
| 0 | 0.008999 | 0.008119 | 1.11× |
| 10 | 0.002921 | 0.003083 | **0.95×** (hurts) |
| 20 | 0.008155 | 0.007303 | 1.12× |
| 40 | 0.010765 | 0.009700 | 1.11× |
| 60 | 0.011848 | 0.011705 | 1.01× |
| 79 | 0.006430 | 0.006233 | 1.03× |

**Finding: Random signs work just as well.** NOVA's optimized signs provide 0-12% benefit and
actually hurt at layer 10. Per-layer sign optimization is unnecessary.

### 6.4 Activation Distribution Across Depth

| Layer | Mean Kurtosis | Mean Max/Mean | Implication |
|-------|:------------:|:------------:|-------------|
| 0 | **1,601** | **641.5** | Extreme outliers → rotation damages weights |
| 10 | 59 | 91.9 | Moderate → rotation helps significantly |
| 20 | 119 | 74.8 | Moderate → rotation helps significantly |
| 40 | 85 | 95.2 | Moderate → rotation helps significantly |
| 60 | 81 | 86.5 | Moderate → rotation helps significantly |
| 79 | 33 | 82.0 | Near-Gaussian → rotation helps the most (228×) |

Layer 0 has **16-48× higher kurtosis** than all other layers. This explains why:
1. The ES search on layer 0 found block-256 (minimizing weight damage from rotation)
2. QuaRot (block-8192) wins at full-model perplexity (layers 1-79 dominate, and they want max mixing)
3. The single-layer GEMM MSE metric was misleading for system-level performance

### 6.5 Revised Deployment Strategy

The data suggests a **per-layer adaptive** approach:

| Layer Range | Rotation | Rationale |
|-------------|----------|-----------|
| Layer 0 | Identity (no rotation) | Extreme kurtosis causes weight damage |
| Layers 1-79 | Block-4096 or block-8192, random signs | Near-Gaussian activations; max mixing optimal |

This eliminates the need for per-layer ES search entirely — just skip rotation at layer 0 and
use a large random Hadamard everywhere else. The "discovery" is that **layer-aware deployment
matters more than sign optimization**.

The adaptive strategy was validated with full-model perplexity: **PPL = 2,522 vs QuaRot's 15,713**
(6.2× improvement). The recipe is trivial: one `if layer_idx == 0: skip` condition.

### 6.6 Per-Token Activation Quantization

The results in §6.1 used per-tensor A4 (one scale for the entire activation matrix). All real
W4A4 systems (QuaRot, SpinQuant, AQLM) use **per-token** quantization — one scale per row/token.
Per-tensor is unrealistically coarse: a single outlier token forces all other tokens into a few
quantization bins, wasting most of the 16 INT4 levels.

**Implementation**: Added `quantize_int4_per_token()` / `dequantize_int4_per_token()` to the INT4
quantize module, and parameterized `apply_w4a4()` / `apply_w4a4_adaptive()` with `--act-quant`
(choices: `per_tensor`, `per_token`).

**Results** (same 288,937 tokens, WikiText-2, stride=2048):

| Config | Per-Tensor PPL | Per-Token PPL | Improvement |
|--------|:--------------:|:------------:|:-----------:|
| FP16 baseline | 2.79 | 2.79 | — |
| W4A4 no rotation | 128,386,408 | 233,919 | 549× |
| W4A4 QuaRot (block-8192) | 15,713 | 15,154 | 1.04× |
| **W4A4 adaptive** (skip L0, block-4096) | 2,522 | **53.94** | **47×** |

**Key findings**:

1. **Per-token massively helps the adaptive strategy** (2,522 → 53.94, 47× improvement). Layer 0's
   extreme per-token outlier variance (kurtosis=1,601) is now handled by independent per-token
   scales instead of one global scale. Each token quantizes against its own dynamic range.

2. **Per-token barely helps QuaRot** (15,713 → 15,154, 1.04×). QuaRot's problem is weight damage
   from full-dimension rotation, not activation granularity. Per-token scales can't fix corrupted weights.

3. **Per-token helps identity a lot** (128M → 234K, 549× improvement) — per-token isolates outlier
   tokens from each other. But without rotation for layers 1-79, it's still catastrophic.

4. **Adaptive is now 281× better than QuaRot** with per-token (53.94 vs 15,154). The hierarchy is
   unambiguous: skip-L0 + block-4096 + per-token is the correct recipe.

**The remaining gap to FP16** (53.94 vs 2.79, 19× degradation) is expected — this is naive simulated
W4A4 without calibration, GPTQ, SmoothQuant, or any weight-aware optimization. The key result is that
the adaptive rotation strategy **produces a coherent model** (PPL < 100) rather than noise (PPL > 15,000),
confirming the rotation framework's value for real W4A4 deployment.

**Comparison to published QuaRot results**: QuaRot (Ashkboos et al., 2024) reports Llama-2-70B
W4A4 perplexity of ~6-8 with GPTQ + learned scales + calibration. Our 53.94 without any calibration
suggests the adaptive rotation strategy should close the gap or beat QuaRot once standard weight
optimization is applied.

### 6.7 Error Budget Decomposition: W4-Only vs A4-Only

To diagnose the remaining 19× gap from FP16 (53.94 vs 2.79), we isolated the contribution
of weight quantization and activation quantization by running each independently. In both
configs, rotation is applied to **both** sides (lossless in FP16) — only quantization is toggled.

| Config | PPL | Degradation vs FP16 |
|--------|:---:|:-------------------:|
| FP16 baseline | 2.79 | 1.0× |
| A4-only adaptive (per-token) | 7.45 | 2.7× |
| W4-only adaptive (per-token) | 10.27 | 3.7× |
| W4A4 adaptive (per-token) | 53.94 | 19.3× |

**Key findings**:

1. **Weights are the larger bottleneck** (3.7× vs 2.7× degradation). Per-group INT4 weight
   quantization (g=128) causes more damage than per-token INT4 activation quantization. This
   is expected: weights participate in every token's computation, so their error compounds across
   the sequence, while activation errors are local to each token.

2. **Errors are multiplicative, not additive**. If errors were independent and additive, combined
   PPL would be ~2.79 + (10.27 - 2.79) + (7.45 - 2.79) = 14.93. The actual 53.94 implies
   quantization noise in weights amplifies activation quantization noise (and vice versa) through
   the GEMM: `error(X_q @ W_q^T) ≈ error(X_q) @ W^T + X @ error(W_q^T) + error(X_q) @ error(W_q^T)`.
   The cross-term `error(X_q) @ error(W_q^T)` is non-negligible at INT4.

3. **A4-only PPL = 7.45 is remarkably good**. Per-token INT4 activations with adaptive rotation
   lose only 2.7× — the rotation framework effectively handles activation quantization. This
   validates the core thesis: rotation + per-token scales tames even 4-bit activations.

4. **GPTQ / learned rounding would have the most impact**. Since weights are the dominant error
   source and weight quantization is a one-time offline process, applying GPTQ (calibration-based
   weight rounding) or AdaRound should yield the largest perplexity improvement per unit of
   engineering effort. This is exactly what published QuaRot and SpinQuant systems do on top of
   rotation.

## 7. Artifacts

| File | Description |
|------|-------------|
| `discovery_results.json` | INT8 single-phase search: config, baselines, NOVA result, full search history |
| `discovery_results_twophase.json` | INT8 two-phase search: same format, includes phase 1+2 history |
| `best_rotation.json` | Best INT8 rotation (block-4096, 8192 optimized signs) |
| `best_rotation_twophase.json` | Best INT8 two-phase rotation |
| `/mnt/data/activations/llama70b_layer0/w4a4_discovery_results.json` | W4A4 search: config, baselines, NOVA result, 24,160 evals |
| `/mnt/data/activations/llama70b_layer0/w4a4_best_rotation.json` | Best W4A4 rotation (block-256, optimized signs) |
| `/mnt/data/activations/llama70b_layer0/perplexity_results.json` | Full-model WikiText-2 perplexity: 4 configs (FP16, identity, QuaRot, NOVA) |
| `/mnt/data/activations/llama70b_layer0/perplexity_adaptive.json` | Adaptive strategy perplexity: skip L0 rotation, block-4096 elsewhere → PPL 2,522 |
| `/mnt/data/activations/llama70b_layer0/perplexity_per_token.json` | Per-token A4 perplexity: identity, QuaRot, adaptive — adaptive achieves PPL 53.94 |
| `/mnt/data/activations/llama70b_layer0/perplexity_decomposition.json` | Error budget decomposition: W4-only (PPL 10.27) and A4-only (PPL 7.45) |
| `/mnt/data/activations/llama70b_layer0/perlayer_results.json` | Per-layer universality/transferability analysis: 6 layers × 8 rotations |
| `/mnt/data/activations/llama70b_layer0/weights/` | Llama-3.1-70B layer 0 weight matrices (.pt) |
| `perplexity_eval.py` | Full-model perplexity evaluation script |
| `perlayer_analysis.py` | Per-layer universality and transferability analysis script |
| `../rotated_fp8/results.json` | Per-layer validation results from the initial experiment |
