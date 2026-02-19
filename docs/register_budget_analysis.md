# Register Budget Analysis: Fully-Fused F(6,3) Winograd Kernel on MI300X

## Hardware Constraints (CDNA3 / gfx942)

| Resource | Per CU | Per SIMD | Per Wavefront (max) |
|----------|--------|----------|---------------------|
| VGPRs    | 512 KB total (shared VGPR+AGPR) | 128 KB | 256 VGPRs |
| AGPRs    | (shared with above) | — | 256 AGPRs |
| VGPR+AGPR combined | 512 KB | 128 KB | 512 total |
| LDS      | 64 KB  | — | — |
| Wavefront size | — | — | 64 lanes |
| Max waves/SIMD | — | 8 | — |
| SIMDs/CU | 4 | — | — |
| Max waves/CU | 32 | — | — |

### Occupancy vs Register Usage

Each VGPR is 64 lanes × 4 bytes = 256 bytes per wave.
Each SIMD has 512 VGPRs + 512 AGPRs = 1024 registers × 64 lanes × 4B = 256 KB.

| Waves/SIMD | VGPRs/wave | AGPRs/wave | Total regs/wave |
|------------|-----------|-----------|----------------|
| 8 (max)    | 64        | 64        | 128            |
| 4          | 128       | 128       | 256            |
| 2          | 256       | 256       | 512            |
| 1          | 256       | 256       | 512            |

**Target**: 4 waves/SIMD (16 waves/CU) — good balance of occupancy and register space.
This gives us **128 VGPRs + 128 AGPRs per wavefront**.

---

## Strategy A: Fully-Fused Single-Kernel (Input Transform → MFMA → Output Transform)

### Overview
One workgroup processes a batch of (b, k, tile) output tiles.
Each wavefront handles one output tile: loads the 8×8 input patch, applies B^T·tile·B,
multiplies by filter across all channel chunks, then applies A·M·A^T and writes output.

### VGPR Budget (per wavefront of 64 lanes)

Each lane in the 64-lane wavefront "owns" one position in the 8×8 grid (lane = row*8 + col).

**Input Transform (B^T · tile · B):**
| Purpose | VGPRs | Notes |
|---------|-------|-------|
| tile_val (loaded input element) | 1 | One FP32 per lane |
| temp (B^T · tile intermediate) | 1 | Reuse via __shfl |
| v_val (transform output) | 1 | V[pos] for this lane |
| Address computation | 4 | b, c, tile indices, pointers |
| **Subtotal** | **7** | |

**MFMA Accumulation (channel reduction):**
For the GEMM: M[pos] = Σ_c U[pos][k][c] · V[pos][c][bp]

The tricky part: each lane holds V[pos] for one position. The MFMA instruction needs
data in a specific layout. For `mfma_f32_16x16x16f16`:
- Input A: 16×16 tile of U (filter)
- Input B: 16×16 tile of V (input transform)
- Output C: 16×16 tile of M (accumulated)

But our "GEMM" per tile position is: M[k] = Σ_c U[k][c] × V[c]
This is a matrix-vector product (K × C matmul with C-vector), NOT a matrix-matrix product.
MFMA is designed for mat-mat, not mat-vec.

**Key insight**: To use MFMA efficiently, we must batch across MULTIPLE tile positions (bp dimension):
- M[k][bp] = Σ_c U[k][c] × V[c][bp]  — this IS a GEMM: [K,C] × [C,BP_chunk]

So each workgroup should process a chunk of bp values (multiple tiles) to form a proper GEMM.

| Purpose | VGPRs | Notes |
|---------|-------|-------|
| V_fragment (chunk of V for MFMA input B) | 8-16 | Depends on tile size |
| U_fragment (chunk of U for MFMA input A) | 8-16 | Streamed from global |
| Address/loop counters | 4 | Channel loop index, pointers |
| **Subtotal** | **20-36** | |

| Purpose | AGPRs | Notes |
|---------|-------|-------|
| MFMA accumulators (M fragment) | 4-16 | Depends on MFMA tile size |
| **Subtotal** | **4-16** | Per output tile |

For `mfma_f32_16x16x16f16`: 4 AGPRs per output tile.
If we want to accumulate a 16×16 output block: 4 AGPRs.
If we want multiple output blocks (to increase reuse): multiply accordingly.

**Output Transform (A · M · A^T):**
After MFMA accumulation, M[pos] values are in AGPRs. Need to:
1. Move from AGPRs to VGPRs
2. Apply A · M · A^T via wave shuffles (same as current output_transform_kernel_mt)

| Purpose | VGPRs | Notes |
|---------|-------|-------|
| m_val (from AGPR) | 1 | Moved from accumulator |
| temp (A · M intermediate) | 1 | Via __shfl |
| y_val (output) | 1 | Final 6×6 element |
| Output address | 3 | h_out, w_out, pointer |
| **Subtotal** | **6** | Reuses input transform VGPRs |

### Total Budget (Strategy A)

| Phase | VGPRs | AGPRs |
|-------|-------|-------|
| Input transform | 7 | 0 |
| MFMA (channel loop) | 20-36 | 4-16 |
| Output transform | 6 | 0 |
| Overlap (max concurrent) | ~40-50 | 4-16 |
| **Budget at 4 waves/SIMD** | **128** | **128** |
| **Margin** | **78-88** | **112-124** |

**Verdict: FEASIBLE at 4 waves/SIMD occupancy.** Significant headroom in AGPRs.

---

## Strategy B: Partial Fusion — Fused per-position GEMM + Output Transform

### Overview
Instead of fusing everything, fuse only the GEMM and output transform.
Input transform remains a separate kernel writing V_gemm to global memory.
But the GEMM + output transform runs as one kernel, eliminating M_gemm.

Each workgroup processes one tile position p (0..63).
For this position, compute M[p][k][bp] = Σ_c U[p][k][c] · V[p][c][bp] using MFMA,
then immediately apply the output transform across all 64 positions.

**Problem**: The output transform needs all 64 positions for a given (k, bp) to compute
A · M · A^T. But each workgroup only has position p. Cross-workgroup synchronization is not
possible within a kernel.

**Solution**: Each workgroup handles ALL 64 positions for a subset of (k, bp) pairs.
This means the workgroup must run 64 separate MFMA accumulations (one per position).

| Purpose | VGPRs | AGPRs |
|---------|-------|-------|
| V_gemm reads (64 positions) | 16 | 0 |
| U_gemm reads (64 positions) | 16 | 0 |
| MFMA accumulators × 64 positions | 0 | 64 × 4 = 256 (!) |
| Output transform | 8 | 0 |

**Problem**: 256 AGPRs needed for 64 position accumulators. This limits to 2 waves/SIMD.

**Alternative**: Process positions in groups of 16 (4 iterations of 16 positions):
- 16 positions × 4 AGPRs = 64 AGPRs
- After each group: do partial output transform, write intermediate to LDS, continue
- Final output transform reads all 64 positions from LDS

| Purpose | VGPRs | AGPRs |
|---------|-------|-------|
| Data loading | 24 | 0 |
| MFMA (16 positions at a time) | 16 | 64 |
| LDS staging | 0 (LDS) | 0 |
| Output transform | 8 | 0 |
| **Total** | **~48** | **64** |
| **Budget at 4 waves/SIMD** | **128** | **128** |

**Verdict: FEASIBLE with position-chunking.**

---

## Strategy C: Fully-Fused with Cooperative Workgroups

### Overview
Each workgroup has 4 wavefronts (256 threads). Assign:
- Each wavefront handles one tile (b, c_chunk, tile_idx)
- All 4 wavefronts cooperate on the channel reduction via LDS

Each wavefront:
1. Loads its input tile patch, applies B^T · tile · B → 64 V values in VGPRs
2. Writes V values to LDS for MFMA input staging
3. Cooperative MFMA: all wavefronts stream U from global, multiply against V in LDS
4. Apply output transform from MFMA accumulators

### LDS Budget
- 4 tiles × 64 positions × 2 bytes (FP16) = 512 bytes per tile group → trivial
- Double buffer: 1 KB
- Filter staging: K_chunk × C_chunk × 2 bytes

With 64 KB LDS per CU and target 4 workgroups/CU: 16 KB LDS per workgroup.
Can stage 16 KB / 2 = 8192 FP16 values in LDS.

**Verdict: FEASIBLE. LDS is not the bottleneck.**

---

## Recommendation

**Strategy A (fully-fused)** is the clear winner:
- VGPR budget: ~50 of 128 available → comfortable
- AGPR budget: ~16 of 128 available → very comfortable
- 4 waves/SIMD occupancy achievable
- Eliminates ALL intermediate global memory traffic
- Single kernel launch

The critical design choice is how to batch the MFMA:
- **Option 1**: Each wavefront processes ONE (b, k, tile) tuple, loops over channels.
  MFMA operates on [1, C] × [C, 1] which is just a dot product — terrible MFMA utilization.
- **Option 2**: Each wavefront processes MULTIPLE bp values for one k, forming
  [K_chunk, C_chunk] × [C_chunk, BP_chunk] GEMM tiles. This is the right approach.
  BP_chunk = 16 gives mfma_f32_16x16x16f16 tiles naturally.

**Recommended MFMA tile**: `mfma_f32_16x16x16f16`
- 16×16 output tile, K-reduction of 16 per instruction
- Maps to K_chunk=16, BP_chunk=16, C_reduction=16 per MFMA
- 4 AGPRs per output tile = very register-efficient
- Each wavefront accumulates one 16×16 tile of M[k_chunk][bp_chunk]

**Workgroup structure**: 4 wavefronts per workgroup
- Wavefront 0-3 each handle a different k_chunk or bp_chunk
- Share filter data via LDS to reduce global memory bandwidth
- After channel reduction loop: apply output transform via wave shuffles

### Register Summary for Recommended Design

| Resource | Used | Budget (4 waves/SIMD) | Utilization |
|----------|------|-----------------------|-------------|
| VGPRs | ~50 | 128 | 39% |
| AGPRs | ~16-32 | 128 | 13-25% |
| LDS | ~4 KB | 16 KB (per WG) | 25% |

This leaves ample headroom for double-buffering (adds ~8 VGPRs for prefetch registers)
and other optimizations.
