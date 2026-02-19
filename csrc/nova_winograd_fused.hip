/**
 * NOVA Winograd F(6,3) — Fused Single-Kernel MFMA Implementation
 *
 * Fuses Input Transform + MFMA GEMM + Output Transform into ONE kernel.
 * Eliminates V_gemm and M_gemm intermediate buffers entirely.
 *
 * Architecture:
 *   Workgroup: 1024 threads = 16 wavefronts, handles BP_PER_WG=16 bp values
 *
 *   Phase 1 (Input Transform): 16 wavefronts do __shfl-based B^T·d·B
 *     → V[c_local][bp_local] per position stored in LDS
 *
 *   Phase 2 (MFMA GEMM): 16 wavefronts handle 64 positions in 4 batches of 16
 *     → MFMA accumulate M[k][bp] = Σ_c U[k][c] × V[c][bp] per position
 *
 *   Phase 3 (Output Transform): Write M to LDS, __shfl A·M·A^T, write output
 *
 * Memory traffic: read input + read U_gemm + write output (same as fused MIOpen)
 * Zero intermediate global memory buffers.
 *
 * Best for conv2_x-sized layers (K,C ≤ 64) at batch > 1.
 * For larger layers, the original 3-pass pipeline is preferred.
 *
 * Build standalone test:
 *   hipcc -DSTANDALONE_TEST -o nova_fused_test nova_winograd_fused.hip \
 *     -std=c++17 -L/opt/rocm/lib -lrocblas -lamdhip64 -I/opt/rocm/include \
 *     --offload-arch=gfx942 -O3
 *
 * Build shared library:
 *   hipcc -shared -fPIC -o nova_winograd_fused.so nova_winograd_fused.hip \
 *     -std=c++17 -L/opt/rocm/lib -lrocblas -lamdhip64 -I/opt/rocm/include \
 *     --offload-arch=gfx942 -O3
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <rocblas/rocblas.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>
#include <algorithm>

#define HIP_CHECK(cmd) do { \
    hipError_t e = cmd; \
    if (e != hipSuccess) { \
        fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

#define ROCBLAS_CHECK(cmd) do { \
    rocblas_status s = cmd; \
    if (s != rocblas_status_success) { \
        fprintf(stderr, "rocBLAS error %d at %s:%d\n", (int)s, __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

// ═══════════════════════════════════════════════════════════════════════════
// MFMA type aliases
// ═══════════════════════════════════════════════════════════════════════════

typedef _Float16 fp16x4_t __attribute__((ext_vector_type(4)));
typedef float    fp32x4_t __attribute__((ext_vector_type(4)));

// ═══════════════════════════════════════════════════════════════════════════
// NOVA F(6,3) Transform Matrices
// ═══════════════════════════════════════════════════════════════════════════

__constant__ __half c_BT[8][8];
__constant__ __half c_B[8][8];
__constant__ float c_G[8][3];
__constant__ float c_GT[3][8];
__constant__ __half c_A[6][8];
__constant__ __half c_AT[8][6];

static const float nova_A_h[6][8] = {
    {1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 0.f},
    {0.f, 0.6f, -0.6f, 1.f, -1.f, 1.1666666667f, -1.1666666667f, 0.f},
    {0.f, 0.36f, 0.36f, 1.f, 1.f, 1.3611111111f, 1.3611111111f, 0.f},
    {0.f, 0.216f, -0.216f, 1.f, -1.f, 1.587962963f, -1.587962963f, 0.f},
    {0.f, 0.1296f, 0.1296f, 1.f, 1.f, 1.8526234568f, 1.8526234568f, 0.f},
    {0.f, 0.07776f, -0.07776f, 1.f, -1.f, 2.1613940329f, -2.1613940329f, 1.f}
};

static const float nova_B_h[8][8] = {
    {-0.49f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f},
    {0.f, 0.8166666667f, -0.8166666667f, 0.49f, -0.49f, 0.42f, -0.42f, -0.49f},
    {2.2111111111f, 1.3611111111f, 1.3611111111f, 0.49f, 0.49f, 0.36f, 0.36f, 0.f},
    {0.f, -1.4166666667f, 1.4166666667f, -1.7211111111f, 1.7211111111f, -1.5866666667f, 1.5866666667f, 2.2111111111f},
    {-2.7211111111f, -2.3611111111f, -2.3611111111f, -1.7211111111f, -1.7211111111f, -1.36f, -1.36f, 0.f},
    {0.f, 0.6f, -0.6f, 1.f, -1.f, 1.1666666667f, -1.1666666667f, -2.7211111111f},
    {1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 0.f},
    {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 1.f}
};

static const float nova_G_h[8][3] = {
    {-2.0408163265f, 0.f, 0.f},
    {2.1677302997f, 1.3006381798f, 0.7803829079f},
    {2.1677302997f, -1.3006381798f, 0.7803829079f},
    {-2.1634615385f, -2.1634615385f, -2.1634615385f},
    {-2.1634615385f, 2.1634615385f, -2.1634615385f},
    {1.0161394021f, 1.1854959691f, 1.3830786306f},
    {1.0161394021f, -1.1854959691f, 1.3830786306f},
    {0.f, 0.f, 1.f}
};

void upload_transform_matrices() {
    __half h_A[6][8], h_AT[8][6], h_B[8][8], h_BT[8][8];
    float h_GT[3][8];
    for (int i = 0; i < 6; i++)
        for (int j = 0; j < 8; j++) {
            h_A[i][j] = __float2half(nova_A_h[i][j]);
            h_AT[j][i] = __float2half(nova_A_h[i][j]);
        }
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 8; j++) {
            h_B[i][j] = __float2half(nova_B_h[i][j]);
            h_BT[j][i] = __float2half(nova_B_h[i][j]);
        }
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 3; j++) h_GT[j][i] = nova_G_h[i][j];
    HIP_CHECK(hipMemcpyToSymbol(c_A, h_A, sizeof(h_A)));
    HIP_CHECK(hipMemcpyToSymbol(c_AT, h_AT, sizeof(h_AT)));
    HIP_CHECK(hipMemcpyToSymbol(c_B, h_B, sizeof(h_B)));
    HIP_CHECK(hipMemcpyToSymbol(c_BT, h_BT, sizeof(h_BT)));
    HIP_CHECK(hipMemcpyToSymbol(c_G, nova_G_h, sizeof(nova_G_h)));
    HIP_CHECK(hipMemcpyToSymbol(c_GT, h_GT, sizeof(h_GT)));
}

// ═══════════════════════════════════════════════════════════════════════════
// Filter Transform Kernel (shared by both pipelines)
// ═══════════════════════════════════════════════════════════════════════════

__global__ __launch_bounds__(64, 16)
void filter_transform_kernel(
    const float* __restrict__ weight, __half* __restrict__ U_gemm, int K, int C)
{
    int tid = threadIdx.x, row = tid >> 3, col = tid & 7;
    int gi = blockIdx.x;
    if (gi >= K * C) return;
    int k = gi / C, c = gi % C;
    float w[3][3];
    const float* wp = weight + (k * C + c) * 9;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) w[i][j] = wp[i * 3 + j];
    float r = 0.0f;
    #pragma unroll
    for (int p = 0; p < 3; p++) {
        float gp = c_G[row][p];
        #pragma unroll
        for (int q = 0; q < 3; q++) r += gp * w[p][q] * c_GT[q][col];
    }
    U_gemm[(long long)tid * K * C + k + (long long)c * K] = __float2half(r);
}

// ═══════════════════════════════════════════════════════════════════════════
// Input/Output Transform Kernels (for 3-pass fallback pipeline)
// ═══════════════════════════════════════════════════════════════════════════

#define TILES_PER_WG 4

__global__ __launch_bounds__(64 * TILES_PER_WG, 4)
void input_transform_kernel_mt(
    const __half* __restrict__ input,
    __half* __restrict__ V_gemm,
    int batch, int C, int H, int W,
    int pad, int nh, int nw, int BP,
    int total_tiles)
{
    const int wf_id = threadIdx.x >> 6;
    const int lane = threadIdx.x & 63;
    const int row = lane >> 3;
    const int col = lane & 7;

    const int global_idx = blockIdx.x * TILES_PER_WG + wf_id;
    if (global_idx >= total_tiles) return;

    const int P = nh * nw;
    const int b = global_idx / (C * P);
    const int rem = global_idx % (C * P);
    const int c = rem / P;
    const int tile_idx = rem % P;
    const int th = tile_idx / nw;
    const int tw = tile_idx % nw;

    const int h_start = th * 6 - pad;
    const int w_start = tw * 6 - pad;
    const int h_idx = h_start + row;
    const int w_idx = w_start + col;

    float tile_val = 0.0f;
    if (h_idx >= 0 && h_idx < H && w_idx >= 0 && w_idx < W)
        tile_val = __half2float(input[((b * C + c) * H + h_idx) * W + w_idx]);

    float temp = 0.0f;
    #pragma unroll
    for (int k = 0; k < 8; k++)
        temp += __half2float(c_BT[row][k]) * __shfl(tile_val, k * 8 + col, 64);

    float v_val = 0.0f;
    #pragma unroll
    for (int k = 0; k < 8; k++)
        v_val += __shfl(temp, row * 8 + k, 64) * __half2float(c_B[k][col]);

    const int pos = lane;
    const int bp = b * P + tile_idx;
    V_gemm[pos * C * BP + c + bp * C] = __float2half(v_val);
}

__global__ __launch_bounds__(64 * TILES_PER_WG, 4)
void output_transform_kernel_mt(
    const __half* __restrict__ M_gemm,
    __half* __restrict__ output,
    int batch, int K, int H_out, int W_out,
    int nh, int nw, int BP,
    int total_tiles)
{
    const int wf_id = threadIdx.x >> 6;
    const int lane = threadIdx.x & 63;
    const int row = lane >> 3;
    const int col = lane & 7;

    const int global_idx = blockIdx.x * TILES_PER_WG + wf_id;
    if (global_idx >= total_tiles) return;

    const int P = nh * nw;
    const int b = global_idx / (K * P);
    const int rem = global_idx % (K * P);
    const int k = rem / P;
    const int tile_idx = rem % P;
    const int th = tile_idx / nw;
    const int tw = tile_idx % nw;

    const int pos = lane;
    const int bp = b * P + tile_idx;
    float m_val = __half2float(M_gemm[pos * K * BP + k + bp * K]);

    float temp = 0.0f;
    #pragma unroll
    for (int p = 0; p < 8; p++) {
        float m_p_col = __shfl(m_val, p * 8 + col, 64);
        if (row < 6) temp += __half2float(c_A[row][p]) * m_p_col;
    }

    float y_val = 0.0f;
    #pragma unroll
    for (int p = 0; p < 8; p++) {
        float temp_row_p = __shfl(temp, row * 8 + p, 64);
        if (row < 6 && col < 6) y_val += temp_row_p * __half2float(c_AT[p][col]);
    }
    if (row < 6 && col < 6) {
        int h_out = th * 6 + row;
        int w_out = tw * 6 + col;
        if (h_out < H_out && w_out < W_out)
            output[((b * K + k) * H_out + h_out) * W_out + w_out] = __float2half(y_val);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Fully Fused MFMA Winograd Kernel
//
// Workgroup: 1024 threads = 16 wavefronts
// Grid: ceil(BP / BP_PER_WG) workgroups
//
// LDS layout for V:
//   V_lds[64_pos][C_CHUNK_c][17_bp_padded]  (FP16)
//   Size: 64 × 16 × 17 × 2 = 34816 bytes ≈ 34KB
//
// LDS layout for M (reuses V space):
//   M_lds[16_k_local][16_bp][64_pos]  (FP16)
//   Size: 16 × 16 × 64 × 2 = 32768 bytes = 32KB
// ═══════════════════════════════════════════════════════════════════════════

#define BP_PER_WG  16
#define C_CHUNK    16
#define POS_PER_WF 4
#define K_SUPER    64

__global__ __launch_bounds__(1024, 1)
void fused_winograd_mfma_kernel(
    const __half* __restrict__ input,     // [B, C, H, W]
    const __half* __restrict__ U_gemm,    // [64, K, C] col-major per pos
    __half* __restrict__ output,          // [B, K, H_out, W_out]
    int batch, int C, int K, int H, int W,
    int pad, int nh, int nw,
    int H_out, int W_out)
{
    const int P = nh * nw;
    const int BP = batch * P;

    const int bp_base = blockIdx.x * BP_PER_WG;
    if (bp_base >= BP) return;

    const int tid    = threadIdx.x;
    const int wf_id  = tid >> 6;
    const int lane   = tid & 63;
    const int row    = lane >> 3;
    const int col    = lane & 7;
    const int pos    = lane;

    const int bp_local = wf_id;
    const int bp_global = bp_base + bp_local;

    const int b_val = (bp_global < BP) ? bp_global / P : 0;
    const int tile_idx = (bp_global < BP) ? bp_global % P : 0;
    const int th = tile_idx / nw;
    const int tw = tile_idx % nw;
    const int h_start = th * 6 - pad;
    const int w_start = tw * 6 - pad;
    const int h_idx = h_start + row;
    const int w_idx = w_start + col;
    const bool bp_valid = (bp_global < BP);

    extern __shared__ __half lds_raw[];
    __half* V_lds = lds_raw;

    for (int k_super = 0; k_super < K; k_super += K_SUPER) {
        const int k_super_end = min(k_super + K_SUPER, K);
        const int n_k_chunks = (k_super_end - k_super + 15) / 16;

        // Accumulators: up to 4 k_chunks × 4 pos_batches
        fp32x4_t acc[4][POS_PER_WF];
        #pragma unroll
        for (int kc = 0; kc < 4; kc++)
            #pragma unroll
            for (int pb = 0; pb < POS_PER_WF; pb++)
                acc[kc][pb] = {0.0f, 0.0f, 0.0f, 0.0f};

        // ════════ Phase 1 + 2: Transform once per c_chunk, accumulate all k_chunks ════════
        for (int c_base = 0; c_base < C; c_base += C_CHUNK) {
            // Phase 1: Input Transform → LDS (computed ONCE per c_chunk)
            for (int c_local = 0; c_local < C_CHUNK && c_base + c_local < C; c_local++) {
                int c = c_base + c_local;
                float tile_val = 0.0f;
                if (bp_valid && h_idx >= 0 && h_idx < H && w_idx >= 0 && w_idx < W)
                    tile_val = __half2float(input[((b_val * C + c) * H + h_idx) * W + w_idx]);

                float temp = 0.0f;
                #pragma unroll
                for (int kk = 0; kk < 8; kk++)
                    temp += __half2float(c_BT[row][kk]) * __shfl(tile_val, kk * 8 + col, 64);

                float v_val = 0.0f;
                #pragma unroll
                for (int kk = 0; kk < 8; kk++)
                    v_val += __shfl(temp, row * 8 + kk, 64) * __half2float(c_B[kk][col]);

                V_lds[pos * (C_CHUNK * 17) + c_local * 17 + bp_local] = __float2half(v_val);
            }
            __syncthreads();

            // Phase 2: MFMA for ALL k_chunks (reusing V from LDS)
            for (int pos_batch = 0; pos_batch < 4; pos_batch++) {
                int p = pos_batch * 16 + wf_id;
                int t_row_mfma = lane % 16;
                int t_grp_mfma = lane / 16;

                // Load B from LDS (same for all k_chunks!)
                fp16x4_t b;
                int bp_col = lane % 16;
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    int c_local = 4 * t_grp_mfma + i;
                    b[i] = (_Float16)V_lds[p * (C_CHUNK * 17) + c_local * 17 + bp_col];
                }

                // MFMA for each k_chunk (different A, same B)
                for (int kc = 0; kc < n_k_chunks; kc++) {
                    int k_base = k_super + kc * 16;
                    fp16x4_t a;
                    int k_g = k_base + t_row_mfma;
                    #pragma unroll
                    for (int i = 0; i < 4; i++) {
                        int c_g = c_base + 4 * t_grp_mfma + i;
                        if (k_g < K && c_g < C)
                            a[i] = (_Float16)U_gemm[(long long)p * K * C + k_g + (long long)c_g * K];
                        else
                            a[i] = (_Float16)0;
                    }
                    acc[kc][pos_batch] = __builtin_amdgcn_mfma_f32_16x16x16f16(
                        a, b, acc[kc][pos_batch], 0, 0, 0);
                }
            }
            __syncthreads();
        }

        // ════════ Phase 3: Output Transform per k_chunk ════════
        __half* M_lds = lds_raw;

        for (int kc = 0; kc < n_k_chunks; kc++) {
            int k_base = k_super + kc * 16;

            // Write M to LDS: M_lds[k_local][bp_col][p] = acc[kc][pos_batch][i]
            for (int pos_batch = 0; pos_batch < 4; pos_batch++) {
                int p = pos_batch * 16 + wf_id;
                int bp_col = lane % 16;
                int k_grp = lane / 16;
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    int k_local = 4 * k_grp + i;
                    M_lds[k_local * 1024 + bp_col * 64 + p] = __float2half(acc[kc][pos_batch][i]);
                }
            }
            __syncthreads();

            // Output transform per k value
            for (int k_local = 0; k_local < 16 && k_base + k_local < K; k_local++) {
                int k = k_base + k_local;
                float m_val = __half2float(M_lds[k_local * 1024 + bp_local * 64 + pos]);

                float out_temp = 0.0f;
                #pragma unroll
                for (int p = 0; p < 8; p++) {
                    float m_p_col = __shfl(m_val, p * 8 + col, 64);
                    if (row < 6) out_temp += __half2float(c_A[row][p]) * m_p_col;
                }

                float y_val = 0.0f;
                #pragma unroll
                for (int p = 0; p < 8; p++) {
                    float temp_row_p = __shfl(out_temp, row * 8 + p, 64);
                    if (row < 6 && col < 6)
                        y_val += temp_row_p * __half2float(c_AT[p][col]);
                }

                if (bp_valid && row < 6 && col < 6) {
                    int h_out = th * 6 + row;
                    int w_out = tw * 6 + col;
                    if (h_out < H_out && w_out < W_out)
                        output[((b_val * K + k) * H_out + h_out) * W_out + w_out] = __float2half(y_val);
                }
            }
            __syncthreads();
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Host wrapper with runtime dispatch
// ═══════════════════════════════════════════════════════════════════════════

class NovaWinogradFused {
public:
    int K_, C_;
    __half* d_U_gemm_;
    rocblas_handle handle_;
    bool initialized_;

    NovaWinogradFused() : K_(0), C_(0), d_U_gemm_(nullptr), initialized_(false) {
        ROCBLAS_CHECK(rocblas_create_handle(&handle_));
    }

    ~NovaWinogradFused() {
        if (d_U_gemm_) hipFree(d_U_gemm_);
        rocblas_destroy_handle(handle_);
    }

    void set_weights(const float* d_weight, int K, int C) {
        K_ = K;
        C_ = C;
        if (d_U_gemm_) hipFree(d_U_gemm_);
        HIP_CHECK(hipMalloc(&d_U_gemm_, 64LL * K * C * sizeof(__half)));
        filter_transform_kernel<<<K * C, 64>>>(d_weight, d_U_gemm_, K, C);
        HIP_CHECK(hipGetLastError());
        initialized_ = true;
    }

    // Forward with runtime dispatch: fused MFMA for small K,C; 3-pass for large
    void forward(
        const __half* d_input, __half* d_output,
        int batch, int H, int W, int pad,
        hipStream_t stream = 0)
    {
        if (!initialized_) {
            fprintf(stderr, "ERROR: call set_weights() before forward()\n");
            exit(1);
        }

        const int m = 6, n = 8;
        int H_out = H + 2 * pad - 2;
        int W_out = W + 2 * pad - 2;
        int H_pad = H_out + n - 1, W_pad = W_out + n - 1;
        int extra_h = (m - (H_pad - n) % m) % m;
        int extra_w = (m - (W_pad - n) % m) % m;
        H_pad += extra_h; W_pad += extra_w;
        int nh = (H_pad - n) / m + 1;
        int nw = (W_pad - n) / m + 1;
        int P = nh * nw;
        int BP = batch * P;

        // Use fused MFMA for conv2_x-sized layers (K,C ≤ 64)
        // with sufficient batch parallelism (BP ≥ 16)
        bool use_fused = (K_ <= 64 && C_ <= 64 && BP >= 16);

        if (use_fused) {
            forward_fused(d_input, d_output, batch, H, W, pad,
                          nh, nw, H_out, W_out, BP, stream);
        } else {
            forward_3pass(d_input, d_output, batch, H, W, pad,
                          nh, nw, H_out, W_out, BP, stream);
        }
    }

private:
    void forward_fused(
        const __half* d_input, __half* d_output,
        int batch, int H, int W, int pad,
        int nh, int nw, int H_out, int W_out, int BP,
        hipStream_t stream)
    {
        dim3 grid((BP + BP_PER_WG - 1) / BP_PER_WG);
        dim3 block(1024);
        int lds_bytes = 64 * C_CHUNK * 17 * sizeof(__half);
        lds_bytes = std::max(lds_bytes, 16 * 16 * 64 * (int)sizeof(__half));

        hipLaunchKernelGGL(fused_winograd_mfma_kernel, grid, block, lds_bytes, stream,
                           d_input, d_U_gemm_, d_output,
                           batch, C_, K_, H, W,
                           pad, nh, nw, H_out, W_out);
        HIP_CHECK(hipGetLastError());
    }

    void forward_3pass(
        const __half* d_input, __half* d_output,
        int batch, int H, int W, int pad,
        int nh, int nw, int H_out, int W_out, int BP,
        hipStream_t stream)
    {
        __half *d_V_gemm, *d_M_gemm;
        HIP_CHECK(hipMalloc(&d_V_gemm, 64LL * C_ * BP * sizeof(__half)));
        HIP_CHECK(hipMalloc(&d_M_gemm, 64LL * K_ * BP * sizeof(__half)));

        rocblas_set_stream(handle_, stream);

        // Input transform
        int P = nh * nw;
        int input_tiles = batch * C_ * P;
        int input_blocks = (input_tiles + TILES_PER_WG - 1) / TILES_PER_WG;
        hipLaunchKernelGGL(input_transform_kernel_mt, dim3(input_blocks), dim3(64 * TILES_PER_WG),
                           0, stream,
                           d_input, d_V_gemm,
                           batch, C_, H, W, pad, nh, nw, BP, input_tiles);

        // rocBLAS strided batched GEMM
        float alpha = 1.0f, beta = 0.0f;
        ROCBLAS_CHECK(rocblas_gemm_strided_batched_ex(
            handle_,
            rocblas_operation_none, rocblas_operation_none,
            K_, BP, C_,
            &alpha,
            d_U_gemm_, rocblas_datatype_f16_r, K_, (rocblas_stride)(K_ * C_),
            d_V_gemm,  rocblas_datatype_f16_r, C_, (rocblas_stride)(C_ * BP),
            &beta,
            d_M_gemm,  rocblas_datatype_f16_r, K_, (rocblas_stride)(K_ * BP),
            d_M_gemm,  rocblas_datatype_f16_r, K_, (rocblas_stride)(K_ * BP),
            64, rocblas_datatype_f32_r,
            rocblas_gemm_algo_standard, 0, 0
        ));

        // Output transform
        int output_tiles = batch * K_ * P;
        int output_blocks = (output_tiles + TILES_PER_WG - 1) / TILES_PER_WG;
        hipLaunchKernelGGL(output_transform_kernel_mt, dim3(output_blocks), dim3(64 * TILES_PER_WG),
                           0, stream,
                           d_M_gemm, d_output,
                           batch, K_, H_out, W_out, nh, nw, BP, output_tiles);

        HIP_CHECK(hipFree(d_V_gemm));
        HIP_CHECK(hipFree(d_M_gemm));
    }
};

// ═══════════════════════════════════════════════════════════════════════════
// C API
// ═══════════════════════════════════════════════════════════════════════════

#ifndef STANDALONE_TEST

extern "C" {

typedef void* nova_fused_handle_t;

nova_fused_handle_t nova_fused_create() {
    upload_transform_matrices();
    return (nova_fused_handle_t)(new NovaWinogradFused());
}

void nova_fused_destroy(nova_fused_handle_t h) {
    delete (NovaWinogradFused*)h;
}

void nova_fused_set_weights(nova_fused_handle_t h, const void* d_weight, int K, int C) {
    ((NovaWinogradFused*)h)->set_weights((const float*)d_weight, K, C);
}

void nova_fused_forward(nova_fused_handle_t h,
                        const void* d_input, void* d_output,
                        int batch, int H, int W, int pad) {
    ((NovaWinogradFused*)h)->forward(
        (const __half*)d_input, (__half*)d_output,
        batch, H, W, pad);
}

void nova_fused_compute_tiling(int H, int W, int pad, int* nh, int* nw, int* H_out, int* W_out) {
    const int m = 6, n_tile = 8;
    *H_out = H + 2 * pad - 2;
    *W_out = W + 2 * pad - 2;
    int H_pad = *H_out + n_tile - 1;
    int W_pad = *W_out + n_tile - 1;
    int extra_h = (m - (H_pad - n_tile) % m) % m;
    int extra_w = (m - (W_pad - n_tile) % m) % m;
    H_pad += extra_h;
    W_pad += extra_w;
    *nh = (H_pad - n_tile) / m + 1;
    *nw = (W_pad - n_tile) / m + 1;
}

} // extern "C"

#endif

// ═══════════════════════════════════════════════════════════════════════════
// Standalone Test
// ═══════════════════════════════════════════════════════════════════════════

#ifdef STANDALONE_TEST

__global__ void direct_conv2d_ref(
    const __half* __restrict__ input, const float* __restrict__ weight,
    float* __restrict__ output,
    int batch, int C, int H, int W, int K, int H_out, int W_out, int pad)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * K * H_out * W_out;
    if (idx >= total) return;
    int b = idx / (K * H_out * W_out);
    int rem = idx % (K * H_out * W_out);
    int k = rem / (H_out * W_out);
    int hw = rem % (H_out * W_out);
    int oh = hw / W_out, ow = hw % W_out;
    float sum = 0.0f;
    for (int c = 0; c < C; c++)
        for (int fh = 0; fh < 3; fh++)
            for (int fw = 0; fw < 3; fw++) {
                int ih = oh - pad + fh, iw = ow - pad + fw;
                if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                    sum += __half2float(input[((b * C + c) * H + ih) * W + iw])
                         * weight[(k * C + c) * 9 + fh * 3 + fw];
            }
    output[idx] = sum;
}

int main() {
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    printf("Device: %s (%s), %d CUs\n\n", prop.name, prop.gcnArchName, prop.multiProcessorCount);

    upload_transform_matrices();

    struct Config { int batch, C, K, H, W, pad; const char* name; };
    Config configs[] = {
        {1,   64,  64,  56, 56, 1, "conv2_x B=1 "},
        {1,  128, 128,  28, 28, 1, "conv3_x B=1 "},
        {8,   64,  64,  56, 56, 1, "conv2_x B=8 "},
        {8,  128, 128,  28, 28, 1, "conv3_x B=8 "},
        {8,  256, 256,  14, 14, 1, "conv4_x B=8 "},
        {32,  64,  64,  56, 56, 1, "conv2_x B=32"},
        {32, 128, 128,  28, 28, 1, "conv3_x B=32"},
    };
    int num_configs = sizeof(configs) / sizeof(configs[0]);

    printf("%-14s | %8s %9s | %8s %9s | %7s | %s\n",
           "Config", "3-pass", "GFLOPS", "Fused", "GFLOPS", "Speedup", "Status");
    printf("─────────────────────────────────────────────────────────────────────────────────\n");

    for (int ci = 0; ci < num_configs; ci++) {
        auto& cfg = configs[ci];
        int nh, nw, H_out, W_out;
        const int m = 6, n = 8;
        H_out = cfg.H + 2 * cfg.pad - 2;
        W_out = cfg.W + 2 * cfg.pad - 2;
        int H_pad = H_out + n - 1, W_pad = W_out + n - 1;
        int extra_h = (m - (H_pad - n) % m) % m;
        int extra_w = (m - (W_pad - n) % m) % m;
        H_pad += extra_h; W_pad += extra_w;
        nh = (H_pad - n) / m + 1; nw = (W_pad - n) / m + 1;
        int P = nh * nw, BP = cfg.batch * P;

        // Setup data
        std::mt19937 rng(42 + ci);
        std::normal_distribution<float> dist(0.f, 0.3f);
        int in_sz = cfg.batch * cfg.C * cfg.H * cfg.W;
        int wt_sz = cfg.K * cfg.C * 9;
        int out_sz = cfg.batch * cfg.K * H_out * W_out;

        std::vector<__half> h_in(in_sz);
        std::vector<float> h_wt(wt_sz);
        for (auto& v : h_in) v = __float2half(dist(rng));
        for (auto& v : h_wt) v = dist(rng);

        __half* d_in; float* d_wt; float* d_ref; __half* d_out_3pass; __half* d_out_fused;
        HIP_CHECK(hipMalloc(&d_in, in_sz * 2));
        HIP_CHECK(hipMalloc(&d_wt, wt_sz * 4));
        HIP_CHECK(hipMalloc(&d_ref, out_sz * 4));
        HIP_CHECK(hipMalloc(&d_out_3pass, out_sz * 2));
        HIP_CHECK(hipMalloc(&d_out_fused, out_sz * 2));
        HIP_CHECK(hipMemcpy(d_in, h_in.data(), in_sz * 2, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_wt, h_wt.data(), wt_sz * 4, hipMemcpyHostToDevice));

        // Filter transform
        __half* d_U;
        HIP_CHECK(hipMalloc(&d_U, 64LL * cfg.K * cfg.C * 2));
        filter_transform_kernel<<<cfg.K * cfg.C, 64>>>(d_wt, d_U, cfg.K, cfg.C);
        HIP_CHECK(hipDeviceSynchronize());

        // Reference
        direct_conv2d_ref<<<(out_sz + 255) / 256, 256>>>(
            d_in, d_wt, d_ref, cfg.batch, cfg.C, cfg.H, cfg.W, cfg.K, H_out, W_out, cfg.pad);
        HIP_CHECK(hipDeviceSynchronize());

        // Allocate workspace for 3-pass
        __half *d_V_gemm, *d_M_gemm;
        HIP_CHECK(hipMalloc(&d_V_gemm, 64LL * cfg.C * BP * 2));
        HIP_CHECK(hipMalloc(&d_M_gemm, 64LL * cfg.K * BP * 2));

        rocblas_handle rb_handle;
        ROCBLAS_CHECK(rocblas_create_handle(&rb_handle));

        // ── Benchmark 3-pass ──
        // Warmup
        for (int r = 0; r < 3; r++) {
            int input_tiles = cfg.batch * cfg.C * P;
            int input_blocks = (input_tiles + TILES_PER_WG - 1) / TILES_PER_WG;
            hipLaunchKernelGGL(input_transform_kernel_mt, dim3(input_blocks), dim3(64 * TILES_PER_WG), 0, 0,
                               d_in, d_V_gemm, cfg.batch, cfg.C, cfg.H, cfg.W, cfg.pad, nh, nw, BP, input_tiles);
            float alpha = 1.0f, beta = 0.0f;
            rocblas_gemm_strided_batched_ex(
                rb_handle, rocblas_operation_none, rocblas_operation_none,
                cfg.K, BP, cfg.C, &alpha,
                d_U, rocblas_datatype_f16_r, cfg.K, (rocblas_stride)((long long)cfg.K * cfg.C),
                d_V_gemm, rocblas_datatype_f16_r, cfg.C, (rocblas_stride)((long long)cfg.C * BP),
                &beta,
                d_M_gemm, rocblas_datatype_f16_r, cfg.K, (rocblas_stride)((long long)cfg.K * BP),
                d_M_gemm, rocblas_datatype_f16_r, cfg.K, (rocblas_stride)((long long)cfg.K * BP),
                64, rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0);
            int output_tiles = cfg.batch * cfg.K * P;
            int output_blocks = (output_tiles + TILES_PER_WG - 1) / TILES_PER_WG;
            hipLaunchKernelGGL(output_transform_kernel_mt, dim3(output_blocks), dim3(64 * TILES_PER_WG), 0, 0,
                               d_M_gemm, d_out_3pass, cfg.batch, cfg.K, H_out, W_out, nh, nw, BP, output_tiles);
        }
        HIP_CHECK(hipDeviceSynchronize());

        hipEvent_t start, stop;
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));
        int repeats = 20;

        HIP_CHECK(hipEventRecord(start));
        for (int r = 0; r < repeats; r++) {
            int input_tiles = cfg.batch * cfg.C * P;
            int input_blocks = (input_tiles + TILES_PER_WG - 1) / TILES_PER_WG;
            hipLaunchKernelGGL(input_transform_kernel_mt, dim3(input_blocks), dim3(64 * TILES_PER_WG), 0, 0,
                               d_in, d_V_gemm, cfg.batch, cfg.C, cfg.H, cfg.W, cfg.pad, nh, nw, BP, input_tiles);
            float alpha = 1.0f, beta = 0.0f;
            rocblas_gemm_strided_batched_ex(
                rb_handle, rocblas_operation_none, rocblas_operation_none,
                cfg.K, BP, cfg.C, &alpha,
                d_U, rocblas_datatype_f16_r, cfg.K, (rocblas_stride)((long long)cfg.K * cfg.C),
                d_V_gemm, rocblas_datatype_f16_r, cfg.C, (rocblas_stride)((long long)cfg.C * BP),
                &beta,
                d_M_gemm, rocblas_datatype_f16_r, cfg.K, (rocblas_stride)((long long)cfg.K * BP),
                d_M_gemm, rocblas_datatype_f16_r, cfg.K, (rocblas_stride)((long long)cfg.K * BP),
                64, rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0);
            int output_tiles = cfg.batch * cfg.K * P;
            int output_blocks = (output_tiles + TILES_PER_WG - 1) / TILES_PER_WG;
            hipLaunchKernelGGL(output_transform_kernel_mt, dim3(output_blocks), dim3(64 * TILES_PER_WG), 0, 0,
                               d_M_gemm, d_out_3pass, cfg.batch, cfg.K, H_out, W_out, nh, nw, BP, output_tiles);
        }
        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));
        float ms_3pass = 0;
        HIP_CHECK(hipEventElapsedTime(&ms_3pass, start, stop));
        ms_3pass /= repeats;

        // ── Benchmark Fused MFMA ──
        dim3 fused_grid((BP + BP_PER_WG - 1) / BP_PER_WG);
        dim3 fused_block(1024);
        int lds_bytes = 64 * C_CHUNK * 17 * sizeof(__half);
        lds_bytes = std::max(lds_bytes, 16 * 16 * 64 * (int)sizeof(__half));

        for (int r = 0; r < 3; r++)
            hipLaunchKernelGGL(fused_winograd_mfma_kernel, fused_grid, fused_block, lds_bytes, 0,
                               d_in, d_U, d_out_fused, cfg.batch, cfg.C, cfg.K, cfg.H, cfg.W,
                               cfg.pad, nh, nw, H_out, W_out);
        HIP_CHECK(hipDeviceSynchronize());

        HIP_CHECK(hipEventRecord(start));
        for (int r = 0; r < repeats; r++)
            hipLaunchKernelGGL(fused_winograd_mfma_kernel, fused_grid, fused_block, lds_bytes, 0,
                               d_in, d_U, d_out_fused, cfg.batch, cfg.C, cfg.K, cfg.H, cfg.W,
                               cfg.pad, nh, nw, H_out, W_out);
        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));
        float ms_fused = 0;
        HIP_CHECK(hipEventElapsedTime(&ms_fused, start, stop));
        ms_fused /= repeats;

        // ── Verify fused kernel ──
        std::vector<__half> h_out(out_sz);
        std::vector<float> h_ref(out_sz);
        HIP_CHECK(hipMemcpy(h_out.data(), d_out_fused, out_sz * 2, hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(h_ref.data(), d_ref, out_sz * 4, hipMemcpyDeviceToHost));

        double sse = 0, sse_ref = 0;
        int nans = 0;
        for (int i = 0; i < out_sz; i++) {
            float v = __half2float(h_out[i]), r = h_ref[i];
            if (std::isnan(v)) { nans++; continue; }
            sse += (double)(v - r) * (v - r);
            sse_ref += (double)r * r;
        }
        float rel = (float)sqrt(sse / (sse_ref + 1e-12));
        bool pass = (nans == 0 && rel < 0.06f);

        double flops = 2.0 * cfg.batch * cfg.K * cfg.C * H_out * W_out * 9;
        float speedup = ms_3pass / ms_fused;
        printf("%-14s | %6.3fms %7.1f G | %6.3fms %7.1f G | %5.2fx  | %s\n",
               cfg.name,
               ms_3pass, flops / (ms_3pass * 1e6),
               ms_fused, flops / (ms_fused * 1e6),
               speedup,
               pass ? "PASS" : "FAIL");

        rocblas_destroy_handle(rb_handle);
        HIP_CHECK(hipFree(d_in));
        HIP_CHECK(hipFree(d_wt));
        HIP_CHECK(hipFree(d_ref));
        HIP_CHECK(hipFree(d_out_3pass));
        HIP_CHECK(hipFree(d_out_fused));
        HIP_CHECK(hipFree(d_U));
        HIP_CHECK(hipFree(d_V_gemm));
        HIP_CHECK(hipFree(d_M_gemm));
        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));
    }
    return 0;
}

#endif
