/**
 * NOVA Winograd F(6,3) — Full-Performance Multi-Pass HIP Kernel
 *
 * Architecture: Input Transform → rocBLAS Strided Batched GEMM → Output Transform
 * - FP16 transforms (NOVA max entry 2.72, safe for FP16)
 * - FP32 accumulation via rocBLAS MFMA (mfma_f32_16x16x16f16)
 * - Filter transform computed once in FP32, stored as FP16
 *
 * Build standalone test:
 *   hipcc -o nova_winograd_v1 nova_winograd_v1.hip -std=c++17 -lrocblas -L/opt/rocm/lib -I/opt/rocm/include
 * Run:
 *   ./nova_winograd_v1
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <rocblas/rocblas.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <chrono>
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
// NOVA F(6,3) Transform Matrices — Constant Memory
// Points: {0, ±3/5, ±1, ±7/6, ∞}, m=6, r=3, n=8
// ═══════════════════════════════════════════════════════════════════════════

// B^T matrix (8x8) for input transform: V = B^T @ tile @ B
// Stored as B^T so we can directly use rows
__constant__ __half c_BT[8][8];

// B matrix (8x8) for right multiply
__constant__ __half c_B[8][8];

// G matrix (8x3) for filter transform: U = G @ w @ G^T
__constant__ float c_G[8][3];

// G^T matrix (3x8)
__constant__ float c_GT[3][8];

// A matrix (6x8) for output transform: Y = A @ M @ A^T
__constant__ __half c_A[6][8];

// A^T matrix (8x6)
__constant__ __half c_AT[8][6];

// Host-side NOVA matrices
static const float nova_A_h[6][8] = {
    {1.0000000000f, 1.0000000000f, 1.0000000000f, 1.0000000000f, 1.0000000000f, 1.0000000000f, 1.0000000000f, 0.0000000000f},
    {0.0000000000f, 0.6000000000f, -0.6000000000f, 1.0000000000f, -1.0000000000f, 1.1666666667f, -1.1666666667f, 0.0000000000f},
    {0.0000000000f, 0.3600000000f, 0.3600000000f, 1.0000000000f, 1.0000000000f, 1.3611111111f, 1.3611111111f, 0.0000000000f},
    {0.0000000000f, 0.2160000000f, -0.2160000000f, 1.0000000000f, -1.0000000000f, 1.5879629630f, -1.5879629630f, 0.0000000000f},
    {0.0000000000f, 0.1296000000f, 0.1296000000f, 1.0000000000f, 1.0000000000f, 1.8526234568f, 1.8526234568f, 0.0000000000f},
    {0.0000000000f, 0.0777600000f, -0.0777600000f, 1.0000000000f, -1.0000000000f, 2.1613940329f, -2.1613940329f, 1.0000000000f}
};

static const float nova_B_h[8][8] = {
    {-0.4900000000f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
    {0.0f, 0.8166666667f, -0.8166666667f, 0.4900000000f, -0.4900000000f, 0.4200000000f, -0.4200000000f, -0.4900000000f},
    {2.2111111111f, 1.3611111111f, 1.3611111111f, 0.4900000000f, 0.4900000000f, 0.3600000000f, 0.3600000000f, 0.0f},
    {0.0f, -1.4166666667f, 1.4166666667f, -1.7211111111f, 1.7211111111f, -1.5866666667f, 1.5866666667f, 2.2111111111f},
    {-2.7211111111f, -2.3611111111f, -2.3611111111f, -1.7211111111f, -1.7211111111f, -1.3600000000f, -1.3600000000f, 0.0f},
    {0.0f, 0.6000000000f, -0.6000000000f, 1.0000000000f, -1.0000000000f, 1.1666666667f, -1.1666666667f, -2.7211111111f},
    {1.0000000000f, 1.0000000000f, 1.0000000000f, 1.0000000000f, 1.0000000000f, 1.0000000000f, 1.0000000000f, 0.0f},
    {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0000000000f}
};

static const float nova_G_h[8][3] = {
    {-2.0408163265f, 0.0000000000f, 0.0000000000f},
    {2.1677302997f, 1.3006381798f, 0.7803829079f},
    {2.1677302997f, -1.3006381798f, 0.7803829079f},
    {-2.1634615385f, -2.1634615385f, -2.1634615385f},
    {-2.1634615385f, 2.1634615385f, -2.1634615385f},
    {1.0161394021f, 1.1854959691f, 1.3830786306f},
    {1.0161394021f, -1.1854959691f, 1.3830786306f},
    {0.0000000000f, 0.0000000000f, 1.0000000000f}
};

// ═══════════════════════════════════════════════════════════════════════════
// Upload constant memory
// ═══════════════════════════════════════════════════════════════════════════

void upload_transform_matrices() {
    // Convert A to half and upload
    __half h_A[6][8], h_AT[8][6];
    for (int i = 0; i < 6; i++)
        for (int j = 0; j < 8; j++) {
            h_A[i][j] = __float2half(nova_A_h[i][j]);
            h_AT[j][i] = __float2half(nova_A_h[i][j]);
        }
    HIP_CHECK(hipMemcpyToSymbol(c_A, h_A, sizeof(h_A)));
    HIP_CHECK(hipMemcpyToSymbol(c_AT, h_AT, sizeof(h_AT)));

    // Convert B to half and upload B and B^T
    __half h_B[8][8], h_BT[8][8];
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 8; j++) {
            h_B[i][j] = __float2half(nova_B_h[i][j]);
            h_BT[j][i] = __float2half(nova_B_h[i][j]);
        }
    HIP_CHECK(hipMemcpyToSymbol(c_B, h_B, sizeof(h_B)));
    HIP_CHECK(hipMemcpyToSymbol(c_BT, h_BT, sizeof(h_BT)));

    // G stays FP32 (filter transform done in FP32)
    float h_GT[3][8];
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 3; j++)
            h_GT[j][i] = nova_G_h[i][j];
    HIP_CHECK(hipMemcpyToSymbol(c_G, nova_G_h, sizeof(nova_G_h)));
    HIP_CHECK(hipMemcpyToSymbol(c_GT, h_GT, sizeof(h_GT)));
}

// ═══════════════════════════════════════════════════════════════════════════
// Kernel 1: Input Transform
// V = B^T @ tile @ B
// Grid: (batch * C * P), Block: (64) — one wavefront per tile
// Thread tid owns position (row=tid/8, col=tid%8) in 8×8 output
// Output: V_gemm in column-major GEMM-friendly layout
// ═══════════════════════════════════════════════════════════════════════════

__global__ __launch_bounds__(64, 16)
void input_transform_kernel(
    const __half* __restrict__ input,  // [B, C, H, W] NCHW (unpadded)
    __half* __restrict__ V_gemm,       // [64, C, B*P] column-major per pos
    int batch, int C, int H, int W,
    int pad, int nh, int nw, int BP    // BP = batch * P
) {
    const int tid = threadIdx.x;        // 0..63
    const int row = tid >> 3;           // 0..7
    const int col = tid & 7;            // 0..7

    const int global_idx = blockIdx.x;  // 0..(batch*C*P - 1)
    const int P = nh * nw;
    const int total = batch * C * P;
    if (global_idx >= total) return;

    // Decode: global_idx = b * (C * P) + c * P + tile_idx
    const int b = global_idx / (C * P);
    const int rem = global_idx % (C * P);
    const int c = rem / P;
    const int tile_idx = rem % P;
    const int th = tile_idx / nw;
    const int tw = tile_idx % nw;

    // Tile origin in padded coordinates
    const int h_start = th * 6 - pad;
    const int w_start = tw * 6 - pad;

    // Load one element of 8x8 tile with boundary check (fused padding)
    const int h_idx = h_start + row;
    const int w_idx = w_start + col;
    float tile_val = 0.0f;
    if (h_idx >= 0 && h_idx < H && w_idx >= 0 && w_idx < W) {
        tile_val = __half2float(input[((b * C + c) * H + h_idx) * W + w_idx]);
    }

    // Shared memory for 8x8 tile (with +1 padding to avoid bank conflicts)
    __shared__ float smem[8][9];
    smem[row][col] = tile_val;
    __syncthreads();

    // Step 1: temp = B^T @ tile (each thread computes temp[row][col])
    float temp = 0.0f;
    #pragma unroll
    for (int k = 0; k < 8; k++) {
        temp += __half2float(c_BT[row][k]) * smem[k][col];
    }
    __syncthreads();

    smem[row][col] = temp;
    __syncthreads();

    // Step 2: V = temp @ B (each thread computes V[row][col])
    float v_val = 0.0f;
    #pragma unroll
    for (int k = 0; k < 8; k++) {
        v_val += smem[row][k] * __half2float(c_B[k][col]);
    }

    // Scatter-write to GEMM layout: V_gemm[pos][c][bp] at pos*C*BP + c + bp*C
    // pos = row*8 + col, bp = b * P + tile_idx
    const int pos = tid;  // row*8 + col = tid
    const int bp = b * P + tile_idx;
    V_gemm[pos * C * BP + c + bp * C] = __float2half(v_val);
}

// ═══════════════════════════════════════════════════════════════════════════
// Kernel 2: Filter Transform
// U = G @ w @ G^T  (computed in FP32, stored as FP16)
// Grid: (K * C), Block: (64) — one wavefront per (k,c) pair
// Output: U_gemm in column-major GEMM-friendly layout
// ═══════════════════════════════════════════════════════════════════════════

__global__ __launch_bounds__(64, 16)
void filter_transform_kernel(
    const float* __restrict__ weight,  // [K, C, 3, 3]
    __half* __restrict__ U_gemm,       // [64, K, C] column-major per pos
    int K, int C
) {
    const int tid = threadIdx.x;
    const int row = tid >> 3;
    const int col = tid & 7;

    const int global_idx = blockIdx.x;
    if (global_idx >= K * C) return;

    const int k = global_idx / C;
    const int c = global_idx % C;

    // Load 3x3 filter into registers (only threads 0..8 need this, but all compute)
    float w[3][3];
    const float* wp = weight + (k * C + c) * 9;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            w[i][j] = wp[i * 3 + j];

    // Step 1: temp = G @ w (8x3 @ 3x3 = 8x3) — each thread computes if col < 3
    // Step 2: result = temp @ G^T (8x3 @ 3x8 = 8x8)
    // Combined: result[row][col] = sum_p sum_q G[row][p] * w[p][q] * G^T[q][col]
    //         = sum_p sum_q G[row][p] * w[p][q] * G[col][q]

    float result = 0.0f;
    #pragma unroll
    for (int p = 0; p < 3; p++) {
        float g_row_p = c_G[row][p];
        #pragma unroll
        for (int q = 0; q < 3; q++) {
            result += g_row_p * w[p][q] * c_GT[q][col];
        }
    }

    // Write to GEMM layout: U_gemm[pos][k][c] at pos*K*C + k + c*K
    const int pos = tid;
    U_gemm[pos * K * C + k + c * K] = __float2half(result);
}

// ═══════════════════════════════════════════════════════════════════════════
// Kernel 3: Output Transform
// Y = A @ M @ A^T  (A is 6x8, output is 6x6)
// Grid: (batch * K * P), Block: (64)
// Only 36 of 64 threads write output (6x6 tile)
// ═══════════════════════════════════════════════════════════════════════════

__global__ __launch_bounds__(64, 16)
void output_transform_kernel(
    const __half* __restrict__ M_gemm,  // [64, K, B*P] column-major per pos
    __half* __restrict__ output,         // [B, K, H_out, W_out] NCHW
    int batch, int K, int H_out, int W_out,
    int nh, int nw, int BP              // BP = batch * P
) {
    const int tid = threadIdx.x;
    const int row = tid >> 3;
    const int col = tid & 7;

    const int global_idx = blockIdx.x;
    const int P = nh * nw;
    const int total = batch * K * P;
    if (global_idx >= total) return;

    const int b = global_idx / (K * P);
    const int rem = global_idx % (K * P);
    const int k = rem / P;
    const int tile_idx = rem % P;
    const int th = tile_idx / nw;
    const int tw = tile_idx % nw;

    // Load M value from GEMM output
    // M_gemm[pos][k][bp] at pos*K*BP + k + bp*K
    const int pos = tid;
    const int bp = b * P + tile_idx;
    float m_val = __half2float(M_gemm[pos * K * BP + k + bp * K]);

    // Store in shared memory
    __shared__ float smem[8][9];
    smem[row][col] = m_val;
    __syncthreads();

    // Step 1: temp = A @ M (6x8 @ 8x8 = 6x8)
    // Only rows 0-5 participate
    float temp = 0.0f;
    if (row < 6) {
        #pragma unroll
        for (int p = 0; p < 8; p++) {
            temp += __half2float(c_A[row][p]) * smem[p][col];
        }
    }
    __syncthreads();

    smem[row][col] = temp;
    __syncthreads();

    // Step 2: Y = temp @ A^T (6x8 @ 8x6 = 6x6)
    // Only (row < 6 && col < 6) threads write
    if (row < 6 && col < 6) {
        float y_val = 0.0f;
        #pragma unroll
        for (int p = 0; p < 8; p++) {
            y_val += smem[row][p] * __half2float(c_AT[p][col]);
        }

        // Write to output with bounds check
        int h_out = th * 6 + row;
        int w_out = tw * 6 + col;
        if (h_out < H_out && w_out < W_out) {
            output[((b * K + k) * H_out + h_out) * W_out + w_out] = __float2half(y_val);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Kernel 1b: Multi-Tile Input Transform with Wave Shuffles
// 256 threads = 4 wavefronts, each wavefront handles one tile
// Uses __shfl instead of LDS for intra-wavefront communication
// ═══════════════════════════════════════════════════════════════════════════

#define TILES_PER_WG 4

__global__ __launch_bounds__(64 * TILES_PER_WG, 4)
void input_transform_kernel_mt(
    const __half* __restrict__ input,
    __half* __restrict__ V_gemm,
    int batch, int C, int H, int W,
    int pad, int nh, int nw, int BP,
    int total_tiles
) {
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
    if (h_idx >= 0 && h_idx < H && w_idx >= 0 && w_idx < W) {
        tile_val = __half2float(input[((b * C + c) * H + h_idx) * W + w_idx]);
    }

    // Step 1: temp = BT @ tile using wave shuffles
    // All 64 lanes participate — no conditional around __shfl
    float temp = 0.0f;
    #pragma unroll
    for (int k = 0; k < 8; k++) {
        float tile_k_col = __shfl(tile_val, k * 8 + col, 64);
        temp += __half2float(c_BT[row][k]) * tile_k_col;
    }

    // Step 2: V = temp @ B using wave shuffles
    float v_val = 0.0f;
    #pragma unroll
    for (int k = 0; k < 8; k++) {
        float temp_row_k = __shfl(temp, row * 8 + k, 64);
        v_val += temp_row_k * __half2float(c_B[k][col]);
    }

    const int pos = lane;
    const int bp = b * P + tile_idx;
    V_gemm[pos * C * BP + c + bp * C] = __float2half(v_val);
}

// ═══════════════════════════════════════════════════════════════════════════
// Kernel 3b: Multi-Tile Output Transform with Wave Shuffles
// ═══════════════════════════════════════════════════════════════════════════

__global__ __launch_bounds__(64 * TILES_PER_WG, 4)
void output_transform_kernel_mt(
    const __half* __restrict__ M_gemm,
    __half* __restrict__ output,
    int batch, int K, int H_out, int W_out,
    int nh, int nw, int BP,
    int total_tiles
) {
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

    // Step 1: temp = A @ M using wave shuffles
    // ALL 64 lanes must participate in __shfl (AMD wavefront requirement)
    float temp = 0.0f;
    #pragma unroll
    for (int p = 0; p < 8; p++) {
        float m_p_col = __shfl(m_val, p * 8 + col, 64);
        if (row < 6) temp += __half2float(c_A[row][p]) * m_p_col;
    }

    // Step 2: Y = temp @ A^T using wave shuffles
    // ALL 64 lanes must participate in __shfl
    float y_val = 0.0f;
    #pragma unroll
    for (int p = 0; p < 8; p++) {
        float temp_row_p = __shfl(temp, row * 8 + p, 64);
        if (row < 6 && col < 6) y_val += temp_row_p * __half2float(c_AT[p][col]);
    }
    if (row < 6 && col < 6) {
        int h_out = th * 6 + row;
        int w_out = tw * 6 + col;
        if (h_out < H_out && w_out < W_out) {
            output[((b * K + k) * H_out + h_out) * W_out + w_out] = __float2half(y_val);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Host Class: NovaWinogradF63
// ═══════════════════════════════════════════════════════════════════════════

class NovaWinogradF63 {
public:
    int K_, C_;
    __half* d_U_gemm_;      // [64, K, C] filter transform (cached)
    rocblas_handle handle_;
    bool initialized_;

    NovaWinogradF63() : K_(0), C_(0), d_U_gemm_(nullptr), initialized_(false) {
        ROCBLAS_CHECK(rocblas_create_handle(&handle_));
    }

    ~NovaWinogradF63() {
        if (d_U_gemm_) hipFree(d_U_gemm_);
        rocblas_destroy_handle(handle_);
    }

    // Pre-transform weights (call once or when weights change)
    void set_weights(const float* d_weight, int K, int C) {
        K_ = K;
        C_ = C;
        if (d_U_gemm_) hipFree(d_U_gemm_);
        HIP_CHECK(hipMalloc(&d_U_gemm_, 64 * K * C * sizeof(__half)));

        filter_transform_kernel<<<K * C, 64>>>(d_weight, d_U_gemm_, K, C);
        HIP_CHECK(hipGetLastError());
        initialized_ = true;
    }

    // Forward pass
    void forward(
        const __half* d_input,  // [batch, C, H, W]
        __half* d_output,       // [batch, K, H_out, W_out]
        int batch, int H, int W, int pad,
        hipStream_t stream = 0
    ) {
        if (!initialized_) {
            fprintf(stderr, "ERROR: call set_weights() before forward()\n");
            exit(1);
        }

        const int m = 6, n = 8;
        int H_out = H + 2 * pad - 2;  // For 3x3 conv with given padding
        int W_out = W + 2 * pad - 2;
        // For pad=1: H_out = H, W_out = W

        // Compute tiling
        int H_pad = H_out + n - 1;  // minimum padded height to cover output
        int W_pad = W_out + n - 1;
        // Round up so (H_pad - n) is divisible by m
        int extra_h = (m - (H_pad - n) % m) % m;
        int extra_w = (m - (W_pad - n) % m) % m;
        H_pad += extra_h;
        W_pad += extra_w;

        int nh = (H_pad - n) / m + 1;
        int nw = (W_pad - n) / m + 1;
        int P = nh * nw;
        int BP = batch * P;

        // Allocate workspace
        __half *d_V_gemm, *d_M_gemm;
        HIP_CHECK(hipMalloc(&d_V_gemm, 64LL * C_ * BP * sizeof(__half)));
        HIP_CHECK(hipMalloc(&d_M_gemm, 64LL * K_ * BP * sizeof(__half)));

        // Set stream
        rocblas_set_stream(handle_, stream);

        // Launch input transform (multi-tile: 4 tiles per workgroup)
        int input_tiles = batch * C_ * P;
        int input_blocks = (input_tiles + TILES_PER_WG - 1) / TILES_PER_WG;
        hipLaunchKernelGGL(input_transform_kernel_mt, dim3(input_blocks), dim3(64 * TILES_PER_WG),
                           0, stream,
                           d_input, d_V_gemm,
                           batch, C_, H, W, pad, nh, nw, BP, input_tiles);
        HIP_CHECK(hipGetLastError());

        // rocBLAS strided batched GEMM
        float alpha = 1.0f, beta = 0.0f;
        ROCBLAS_CHECK(rocblas_gemm_strided_batched_ex(
            handle_,
            rocblas_operation_none,     // transA
            rocblas_operation_none,     // transB
            K_, BP, C_,                // m, n, k
            &alpha,
            d_U_gemm_, rocblas_datatype_f16_r, K_, (rocblas_stride)(K_ * C_),
            d_V_gemm,  rocblas_datatype_f16_r, C_, (rocblas_stride)(C_ * BP),
            &beta,
            d_M_gemm,  rocblas_datatype_f16_r, K_, (rocblas_stride)(K_ * BP),
            d_M_gemm,  rocblas_datatype_f16_r, K_, (rocblas_stride)(K_ * BP),
            64,                         // batch_count = 8*8 positions
            rocblas_datatype_f32_r,     // FP32 accumulate via MFMA
            rocblas_gemm_algo_standard, 0, 0
        ));

        // Launch output transform (multi-tile: 4 tiles per workgroup)
        int output_tiles = batch * K_ * P;
        int output_blocks = (output_tiles + TILES_PER_WG - 1) / TILES_PER_WG;
        hipLaunchKernelGGL(output_transform_kernel_mt, dim3(output_blocks), dim3(64 * TILES_PER_WG),
                           0, stream,
                           d_M_gemm, d_output,
                           batch, K_, H_out, W_out, nh, nw, BP, output_tiles);
        HIP_CHECK(hipGetLastError());

        // Free workspace
        HIP_CHECK(hipFree(d_V_gemm));
        HIP_CHECK(hipFree(d_M_gemm));
    }

    // Forward with pre-allocated workspace (avoids malloc per call)
    void forward_workspace(
        const __half* d_input, __half* d_output,
        __half* d_V_gemm, __half* d_M_gemm,
        int batch, int H, int W, int pad,
        int nh, int nw,
        hipStream_t stream = 0
    ) {
        int P = nh * nw;
        int BP = batch * P;
        int H_out = H + 2 * pad - 2;
        int W_out = W + 2 * pad - 2;

        rocblas_set_stream(handle_, stream);

        // Input transform (multi-tile: 4 tiles per workgroup)
        int input_tiles = batch * C_ * P;
        int input_blocks = (input_tiles + TILES_PER_WG - 1) / TILES_PER_WG;
        hipLaunchKernelGGL(input_transform_kernel_mt, dim3(input_blocks), dim3(64 * TILES_PER_WG),
                           0, stream,
                           d_input, d_V_gemm,
                           batch, C_, H, W, pad, nh, nw, BP, input_tiles);

        // GEMM
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

        // Output transform (multi-tile: 4 tiles per workgroup)
        int output_tiles = batch * K_ * P;
        int output_blocks = (output_tiles + TILES_PER_WG - 1) / TILES_PER_WG;
        hipLaunchKernelGGL(output_transform_kernel_mt, dim3(output_blocks), dim3(64 * TILES_PER_WG),
                           0, stream,
                           d_M_gemm, d_output,
                           batch, K_, H_out, W_out, nh, nw, BP, output_tiles);
    }
};

// ═══════════════════════════════════════════════════════════════════════════
// Utility: compute tiling params
// ═══════════════════════════════════════════════════════════════════════════

void compute_tiling(int H, int W, int pad, int* nh, int* nw, int* H_out, int* W_out) {
    const int m = 6, n = 8;
    *H_out = H + 2 * pad - 2;
    *W_out = W + 2 * pad - 2;

    int H_pad = *H_out + n - 1;
    int W_pad = *W_out + n - 1;
    int extra_h = (m - (H_pad - n) % m) % m;
    int extra_w = (m - (W_pad - n) % m) % m;
    H_pad += extra_h;
    W_pad += extra_w;

    *nh = (H_pad - n) / m + 1;
    *nw = (W_pad - n) / m + 1;
}

// ═══════════════════════════════════════════════════════════════════════════
// Direct convolution reference (FP32)
// ═══════════════════════════════════════════════════════════════════════════

__global__ void direct_conv2d_ref(
    const __half* __restrict__ input,  // [B, C, H, W]
    const float* __restrict__ weight,  // [K, C, 3, 3]
    float* __restrict__ output,        // [B, K, H_out, W_out]
    int batch, int C, int H, int W, int K, int H_out, int W_out, int pad
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * K * H_out * W_out;
    if (idx >= total) return;

    int b = idx / (K * H_out * W_out);
    int rem = idx % (K * H_out * W_out);
    int k = rem / (H_out * W_out);
    int hw = rem % (H_out * W_out);
    int oh = hw / W_out;
    int ow = hw % W_out;

    float sum = 0.0f;
    for (int c = 0; c < C; c++) {
        for (int fh = 0; fh < 3; fh++) {
            for (int fw = 0; fw < 3; fw++) {
                int ih = oh - pad + fh;
                int iw = ow - pad + fw;
                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                    float in_val = __half2float(input[((b * C + c) * H + ih) * W + iw]);
                    float w_val = weight[(k * C + c) * 9 + fh * 3 + fw];
                    sum += in_val * w_val;
                }
            }
        }
    }
    output[idx] = sum;
}

// ═══════════════════════════════════════════════════════════════════════════
// Standalone Test
// ═══════════════════════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════════════════════
// C API for shared library (used by Python ctypes wrapper)
// ═══════════════════════════════════════════════════════════════════════════

#ifndef STANDALONE_TEST

extern "C" {

// Opaque handle
typedef void* nova_handle_t;

nova_handle_t nova_create() {
    upload_transform_matrices();
    auto* ctx = new NovaWinogradF63();
    return (nova_handle_t)ctx;
}

void nova_destroy(nova_handle_t h) {
    delete (NovaWinogradF63*)h;
}

void nova_set_weights(nova_handle_t h, const void* d_weight, int K, int C) {
    ((NovaWinogradF63*)h)->set_weights((const float*)d_weight, K, C);
}

void nova_forward(nova_handle_t h,
                  const void* d_input, void* d_output,
                  int batch, int H, int W, int pad) {
    ((NovaWinogradF63*)h)->forward(
        (const __half*)d_input, (__half*)d_output,
        batch, H, W, pad);
}

void nova_forward_workspace(nova_handle_t h,
                            const void* d_input, void* d_output,
                            void* d_V_gemm, void* d_M_gemm,
                            int batch, int H, int W, int pad,
                            int nh, int nw) {
    ((NovaWinogradF63*)h)->forward_workspace(
        (const __half*)d_input, (__half*)d_output,
        (__half*)d_V_gemm, (__half*)d_M_gemm,
        batch, H, W, pad, nh, nw);
}

void nova_compute_tiling(int H, int W, int pad, int* nh, int* nw, int* H_out, int* W_out) {
    compute_tiling(H, W, pad, nh, nw, H_out, W_out);
}

// Filter transform only (for caching U_gemm from Python)
void nova_filter_transform(const void* d_weight, void* d_U_gemm, int K, int C) {
    upload_transform_matrices();
    filter_transform_kernel<<<K * C, 64>>>((const float*)d_weight, (__half*)d_U_gemm, K, C);
    hipDeviceSynchronize();
}

// Input transform only
void nova_input_transform(const void* d_input, void* d_V_gemm,
                          int batch, int C, int H, int W,
                          int pad, int nh, int nw, int BP) {
    int P = nh * nw;
    int blocks = batch * C * P;
    input_transform_kernel<<<blocks, 64>>>(
        (const __half*)d_input, (__half*)d_V_gemm,
        batch, C, H, W, pad, nh, nw, BP);
}

// Output transform only
void nova_output_transform(const void* d_M_gemm, void* d_output,
                           int batch, int K, int H_out, int W_out,
                           int nh, int nw, int BP) {
    int P = nh * nw;
    int blocks = batch * K * P;
    output_transform_kernel<<<blocks, 64>>>(
        (const __half*)d_M_gemm, (__half*)d_output,
        batch, K, H_out, W_out, nh, nw, BP);
}

int nova_get_K(nova_handle_t h) { return ((NovaWinogradF63*)h)->K_; }
int nova_get_C(nova_handle_t h) { return ((NovaWinogradF63*)h)->C_; }
void* nova_get_U_gemm(nova_handle_t h) { return ((NovaWinogradF63*)h)->d_U_gemm_; }

} // extern "C"

#endif // !STANDALONE_TEST

#ifdef STANDALONE_TEST

int main() {
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    printf("Device: %s (%s)\n", prop.name, prop.gcnArchName);
    printf("CUs: %d, Memory: %.1f GB\n\n", prop.multiProcessorCount, prop.totalGlobalMem / 1e9);

    upload_transform_matrices();

    // Test configurations (ResNet-50 representative layers)
    struct TestConfig {
        int batch, C, K, H, W, pad;
        const char* name;
    };

    TestConfig configs[] = {
        {1,  64,  64,  56, 56, 1, "conv2_x (B=1)"},
        {1,  128, 128, 28, 28, 1, "conv3_x (B=1)"},
        {1,  256, 256, 14, 14, 1, "conv4_x (B=1)"},
        {1,  512, 512,  7,  7, 1, "conv5_x (B=1)"},
        {8,  64,  64,  56, 56, 1, "conv2_x (B=8)"},
        {8,  128, 128, 28, 28, 1, "conv3_x (B=8)"},
    };
    int num_configs = sizeof(configs) / sizeof(configs[0]);

    for (int ci = 0; ci < num_configs; ci++) {
        auto& cfg = configs[ci];
        printf("═══ %s: B=%d C=%d K=%d %dx%d pad=%d ═══\n",
               cfg.name, cfg.batch, cfg.C, cfg.K, cfg.H, cfg.W, cfg.pad);

        int nh, nw, H_out, W_out;
        compute_tiling(cfg.H, cfg.W, cfg.pad, &nh, &nw, &H_out, &W_out);
        int P = nh * nw;
        printf("  Tiles: %dx%d = %d, Output: %dx%d\n", nh, nw, P, H_out, W_out);

        // Generate random data
        std::mt19937 rng(42 + ci);
        std::normal_distribution<float> dist(0.0f, 0.5f);

        int in_size = cfg.batch * cfg.C * cfg.H * cfg.W;
        int wt_size = cfg.K * cfg.C * 9;
        int out_size = cfg.batch * cfg.K * H_out * W_out;

        std::vector<__half> h_input(in_size);
        std::vector<float> h_weight(wt_size);
        for (auto& v : h_input) v = __float2half(dist(rng));
        for (auto& v : h_weight) v = dist(rng);

        // Upload
        __half* d_input;
        float* d_weight;
        HIP_CHECK(hipMalloc(&d_input, in_size * sizeof(__half)));
        HIP_CHECK(hipMalloc(&d_weight, wt_size * sizeof(float)));
        HIP_CHECK(hipMemcpy(d_input, h_input.data(), in_size * sizeof(__half), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_weight, h_weight.data(), wt_size * sizeof(float), hipMemcpyHostToDevice));

        // Reference: direct conv FP32
        float* d_ref;
        HIP_CHECK(hipMalloc(&d_ref, out_size * sizeof(float)));
        int threads = 256;
        int blocks = (out_size + threads - 1) / threads;
        direct_conv2d_ref<<<blocks, threads>>>(
            d_input, d_weight, d_ref,
            cfg.batch, cfg.C, cfg.H, cfg.W, cfg.K, H_out, W_out, cfg.pad
        );
        HIP_CHECK(hipDeviceSynchronize());

        // NOVA Winograd
        NovaWinogradF63 nova;
        nova.set_weights(d_weight, cfg.K, cfg.C);
        HIP_CHECK(hipDeviceSynchronize());

        __half* d_output;
        HIP_CHECK(hipMalloc(&d_output, out_size * sizeof(__half)));

        // Warmup
        nova.forward(d_input, d_output, cfg.batch, cfg.H, cfg.W, cfg.pad);
        HIP_CHECK(hipDeviceSynchronize());

        // Timed runs
        int repeats = 20;
        hipEvent_t start, stop;
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));

        // Pre-allocate workspace for timed run
        int BP = cfg.batch * P;
        __half *d_V_gemm, *d_M_gemm;
        HIP_CHECK(hipMalloc(&d_V_gemm, 64LL * cfg.C * BP * sizeof(__half)));
        HIP_CHECK(hipMalloc(&d_M_gemm, 64LL * cfg.K * BP * sizeof(__half)));

        HIP_CHECK(hipEventRecord(start));
        for (int r = 0; r < repeats; r++) {
            nova.forward_workspace(d_input, d_output, d_V_gemm, d_M_gemm,
                                   cfg.batch, cfg.H, cfg.W, cfg.pad, nh, nw);
        }
        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));
        float total_ms = 0;
        HIP_CHECK(hipEventElapsedTime(&total_ms, start, stop));
        float avg_ms = total_ms / repeats;

        // Check accuracy
        std::vector<__half> h_output(out_size);
        std::vector<float> h_ref(out_size);
        HIP_CHECK(hipMemcpy(h_output.data(), d_output, out_size * sizeof(__half), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(h_ref.data(), d_ref, out_size * sizeof(float), hipMemcpyDeviceToHost));

        double sse = 0, sse_ref = 0, max_err = 0;
        int nan_count = 0, inf_count = 0;
        for (int i = 0; i < out_size; i++) {
            float v = __half2float(h_output[i]);
            float r = h_ref[i];
            if (std::isnan(v)) { nan_count++; continue; }
            if (std::isinf(v)) { inf_count++; continue; }
            double err = fabs(v - r);
            if (err > max_err) max_err = err;
            sse += (v - r) * (v - r);
            sse_ref += r * r;
        }
        float rel_err = (float)(sqrt(sse) / sqrt(sse_ref + 1e-12));

        // Compute GFLOPS
        double flops = 2.0 * cfg.batch * cfg.K * cfg.C * H_out * W_out * 9;
        double gflops = flops / (avg_ms * 1e6);

        printf("  Time: %.3f ms (%.1f GFLOPS)\n", avg_ms, gflops);
        printf("  Relative error: %.6e\n", rel_err);
        printf("  Max absolute error: %.6e\n", max_err);
        printf("  NaN: %d, Inf: %d\n", nan_count, inf_count);
        printf("  Status: %s\n\n", (nan_count == 0 && inf_count == 0 && rel_err < 0.06) ? "PASS" : "FAIL");

        HIP_CHECK(hipFree(d_input));
        HIP_CHECK(hipFree(d_weight));
        HIP_CHECK(hipFree(d_ref));
        HIP_CHECK(hipFree(d_output));
        HIP_CHECK(hipFree(d_V_gemm));
        HIP_CHECK(hipFree(d_M_gemm));
        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));
    }

    return 0;
}

#endif // STANDALONE_TEST
