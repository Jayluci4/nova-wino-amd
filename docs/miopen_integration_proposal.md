# MIOpen Integration Proposal: NOVA Points for Multi-Pass Winograd

**Author**: Jayant Lohia
**Date**: February 2026
**Scope**: Replace transform matrix constants in existing MIOpen multi-pass Winograd solvers

---

## What Already Exists in MIOpen

MIOpen contains a complete multi-pass bidirectional Winograd framework that was
never shipped to production:

| Component | Status |
|-----------|--------|
| `ConvMPBidirectWinograd<2-3>` through `<6-3>` | Implemented, never enabled |
| `ConvMPBidirectWinograd_xdlops<2-3>` through `<6-3>` | Implemented, never enabled |
| Assembly transform templates (macro-parameterized) | Complete |
| rocBLAS strided batched GEMM integration | Complete |
| Performance database infrastructure | Complete (zero entries for >f2x3) |
| `MIOPEN_DEBUG_AMD_MP_BD_WINOGRAD_EXPEREMENTAL_FP16_TRANSFORM` | Flag exists (note typo) |

**Evidence of abandonment**: Zero performance DB entries on gfx942, gfx90a,
gfx908, and gfx906 for any multi-pass solver. Zero pre-compiled kernel binaries
for tile sizes larger than f2x3. The experimental FP16 flag confirms that FP16
transform paths were explored.

## Why It Was Abandoned

Standard Cook-Toom interpolation points (0, ±1, ±2, ±3) produce transform
matrices with rapidly growing condition numbers:

| Tile | Standard κ(B^T) | Max Matrix Entry | FP16 Viable? |
|------|----------------|------------------|--------------|
| F(2,3) | 3.2 | 1 | Yes (production) |
| F(4,3) | 42.5 | 4 | Marginal |
| F(6,3) | 2,075 | ~10 | **No** — 221K NaN on ImageNet |
| F(8,3) | 196,900 | ~243 | No |

At F(6,3), the condition number of 2,075 means FP16 rounding errors are
amplified ~2,075× during the inverse transform. This produces NaN values,
destroys accuracy (31% top-1 vs. 63% baseline), and makes the output unusable.

**The decision not to ship was correct.** Standard points are genuinely broken
at FP16 for large tiles.

## What NOVA Changes

NOVA selects real-valued interpolation points that minimize condition numbers.
The transform matrices (A, B^T, G) change; the algorithm and architecture do not.

| Tile | Standard κ | NOVA κ | Improvement | FP16 Status |
|------|-----------|--------|-------------|-------------|
| F(4,3) | 42.5 | 14.5 | 2.9× | Reliable |
| F(6,3) | 2,075 | **77** | **27×** | **Verified on 10K images** |
| F(8,3) | 196,900 | 474 | 415× | Still insufficient |

For 2D convolutions (Kronecker products), the improvements square:
- F(4×4, 3×3): 8.5× improvement
- F(6×6, 3×3): **733× improvement**

## Concrete Integration Steps

### Step 1: Matrix Constant Swap (~500 bytes)

Replace the A, B^T, G matrices in the multi-pass solver with NOVA values.
These are compile-time constants in the assembly templates. The NOVA matrices
are provided as exact rational numbers (reproducible via the Cook-Toom
algorithm with NOVA's point set).

### Step 2: Enable Existing Solver

The `ConvMPBidirectWinograd<6-3>` (or `_xdlops<6-3>`) solver class already has:
- `IsApplicable()` — check convolution parameters
- `FindSolution()` — compute workspace and invoke transforms + GEMM
- `GetInvokeFactory()` — dispatch to the kernel pipeline

These need to be enabled in the solver registry for target architectures
(gfx942 initially, then gfx90a).

### Step 3: Performance Database Tuning

Run the standard MIOpen tuning pipeline to populate performance DB entries.
The solver already integrates with this infrastructure—it just has zero
entries because it was never benchmarked.

### Step 4: Validation

- Single-layer correctness vs. FP32 direct convolution
- ImageNet accuracy (my results: 63.29% top-1, zero NaN, zero degradation)
- Stable Diffusion end-to-end (my results: 49/49 layers, valid images)

## Risk Assessment

| Risk | Level | Rationale |
|------|-------|-----------|
| Numerical correctness | **Low** | Verified on 10K ImageNet images + Stable Diffusion |
| Architecture change | **None** | Same multi-pass pipeline MIOpen team designed |
| Performance regression | **None** | Only affects new solver; existing F(2,3) unchanged |
| Maintenance burden | **Low** | ~500 bytes of constants; same code paths |
| Testing scope | **Medium** | Need to validate across spatial sizes and channel configs |

## What I'm Providing

1. **NOVA F(6,3) transform matrices** — exact rational values, verified via
   Kronecker product squaring (ratio = 1.000000 for all matrices)
2. **Working HIP kernel** — 906 lines, same multi-pass architecture as MIOpen,
   demonstrates the algorithm at production performance
3. **Validation results** — ResNet-50, DenseNet-161, SD 1.5, SDXL; 10K-image
   ImageNet accuracy; NaN/Inf counts
4. **This proposal** — mapping from my implementation to MIOpen's existing code

Engineering decisions about assembly optimization, solver priorities, and
release scheduling are for the MIOpen team.

---

*The infrastructure is built. The numerical fix is proven. The delta is small.*
