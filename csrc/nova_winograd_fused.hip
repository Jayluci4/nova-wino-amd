/**
 * NOVA Winograd F(6,3) — Fused Single-Kernel Implementation
 *
 * Fuses Input Transform + MFMA GEMM + Output Transform into ONE kernel.
 * Eliminates V_gemm and M_gemm intermediate buffers entirely.
 *
 * Architecture:
 *   Each workgroup processes a block of (k_chunk × bp_chunk) output tile pairs
 *   for ALL 64 tile positions. The pipeline:
 *     1. Stream input tiles → register-resident B^T·tile·B → LDS staging
 *     2. MFMA accumulation: M[pos][k][bp] = Σ_c U[pos][k][c] · V[pos][c][bp]
 *     3. Register-resident A·M·A^T → write output directly
 *
 * Memory traffic: read input + read U_gemm + write output (same as fused MIOpen)
 * Zero intermediate global memory buffers.
 *
 * Build:
 *   hipcc -shared -fPIC -o nova_winograd_fused.so nova_winograd_fused.hip \
 *     -std=c++17 -L/opt/rocm/lib -lrocblas -lamdhip64 -I/opt/rocm/include \
 *     --offload-arch=gfx942
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
#include <chrono>

#define HIP_CHECK(cmd) do { \
    hipError_t e = cmd; \
    if (e != hipSuccess) { \
        fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e), __FILE__, __LINE__); \
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

// B^T matrix (8x8) for input transform
__constant__ __half c_BT[8][8];
__constant__ __half c_B[8][8];
__constant__ float c_G[8][3];
__constant__ float c_GT[3][8];
__constant__ __half c_A[6][8];
__constant__ __half c_AT[8][6];

// Host-side NOVA matrices (same as original)
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

void upload_transform_matrices() {
    __half h_A[6][8], h_AT[8][6];
    for (int i = 0; i < 6; i++)
        for (int j = 0; j < 8; j++) {
            h_A[i][j] = __float2half(nova_A_h[i][j]);
            h_AT[j][i] = __float2half(nova_A_h[i][j]);
        }
    HIP_CHECK(hipMemcpyToSymbol(c_A, h_A, sizeof(h_A)));
    HIP_CHECK(hipMemcpyToSymbol(c_AT, h_AT, sizeof(h_AT)));

    __half h_B[8][8], h_BT[8][8];
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 8; j++) {
            h_B[i][j] = __float2half(nova_B_h[i][j]);
            h_BT[j][i] = __float2half(nova_B_h[i][j]);
        }
    HIP_CHECK(hipMemcpyToSymbol(c_B, h_B, sizeof(h_B)));
    HIP_CHECK(hipMemcpyToSymbol(c_BT, h_BT, sizeof(h_BT)));

    float h_GT[3][8];
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 3; j++)
            h_GT[j][i] = nova_G_h[i][j];
    HIP_CHECK(hipMemcpyToSymbol(c_G, nova_G_h, sizeof(nova_G_h)));
    HIP_CHECK(hipMemcpyToSymbol(c_GT, h_GT, sizeof(h_GT)));
}

// ═══════════════════════════════════════════════════════════════════════════
// Filter Transform (unchanged from original — run once, cached)
// ═══════════════════════════════════════════════════════════════════════════

__global__ __launch_bounds__(64, 16)
void filter_transform_kernel(
    const float* __restrict__ weight,
    __half* __restrict__ U_gemm,
    int K, int C)
{
    const int tid = threadIdx.x;
    const int row = tid >> 3;
    const int col = tid & 7;
    const int global_idx = blockIdx.x;
    if (global_idx >= K * C) return;

    const int k = global_idx / C;
    const int c = global_idx % C;

    float w[3][3];
    const float* wp = weight + (k * C + c) * 9;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            w[i][j] = wp[i * 3 + j];

    float result = 0.0f;
    #pragma unroll
    for (int p = 0; p < 3; p++) {
        float g_row_p = c_G[row][p];
        #pragma unroll
        for (int q = 0; q < 3; q++) {
            result += g_row_p * w[p][q] * c_GT[q][col];
        }
    }

    const int pos = tid;
    U_gemm[pos * K * C + k + c * K] = __float2half(result);
}

// ═══════════════════════════════════════════════════════════════════════════
// Fused Winograd Kernel: Input Transform + MFMA GEMM + Output Transform
//
// Strategy: Each workgroup handles one (b, tile) pair across ALL K outputs.
// Within the workgroup, 64 threads form one wavefront that processes one
// tile position at a time, iterating over all 64 positions.
//
// For each position p (0..63):
//   1. Compute V[p][c] for this tile via input transform (wave shuffle)
//   2. Accumulate M[p][k] += U[p][k][c] * V[p][c] across all channels
//      using MFMA: [K_chunk, C_chunk] x [C_chunk, 1]
//   3. After all 64 positions: apply output transform A * M * A^T
//
// Note: With only 1 bp per workgroup, the MFMA is a mat-vec (K×C × C×1).
// For better MFMA utilization, we batch multiple bp values per workgroup.
// ═══════════════════════════════════════════════════════════════════════════

// Version 1: Per-tile-position approach
// Each workgroup handles BP_PER_WG tiles, accumulating all positions.
// This approach processes one position at a time across a batch of bp values.
//
// The key: each wavefront does MFMA [16, C_chunk] × [C_chunk, 16] for 16 bp values.
// With 4 wavefronts per WG, we cover K in chunks of 16.

// ═══════════════════════════════════════════════════════════════════════════
// Fused Winograd Kernel v3: K-Tiled with Input Transform Reuse
//
// Each workgroup = 1 wavefront (64 threads). Each lane owns one position
// in the 8×8 tile grid.
//
// K-tiling: process K_TILE output channels per chunk. For each chunk:
//   For each c (input channel):
//     1. Input transform via __shfl (computed ONCE per c, reused across K_TILE k's)
//     2. Unrolled FMA: m_accum[i] += U[pos][k_base+i][c] * v_val, i=0..K_TILE-1
//   Output transform for each k in chunk, then write.
//
// This reduces input transform computations from K×C to ceil(K/K_TILE)×C.
// For K=64, K_TILE=16: 4× fewer transforms. For K=512: 32× fewer.
// ═══════════════════════════════════════════════════════════════════════════

#define K_TILE 16  // output channels per chunk (register pressure: 16 floats = 16 VGPRs)

__global__ __launch_bounds__(64, 8)
void fused_winograd_kernel(
    const __half* __restrict__ input,     // [B, C, H, W] NCHW
    const __half* __restrict__ U_gemm,    // [64, K, C] column-major per pos
    __half* __restrict__ output,          // [B, K, H_out, W_out] NCHW
    int batch, int C, int K, int H, int W,
    int pad, int nh, int nw,
    int H_out, int W_out
) {
    const int P = nh * nw;
    const int BP = batch * P;

    // Each workgroup handles one bp value
    const int bp = blockIdx.x;
    if (bp >= BP) return;

    const int lane = threadIdx.x;  // 0..63
    const int pos_row = lane / 8;  // 0..7
    const int pos_col = lane % 8;  // 0..7

    const int b = bp / P;
    const int tile_idx = bp % P;
    const int th = tile_idx / nw;
    const int tw = tile_idx % nw;

    const int h_start = th * 6 - pad;
    const int w_start = tw * 6 - pad;
    const int h_idx = h_start + pos_row;
    const int w_idx = w_start + pos_col;

    // Base pointer for this lane's position in U_gemm
    const __half* U_base = U_gemm + lane * K * C;

    // Tile K in chunks of K_TILE
    for (int k_base = 0; k_base < K; k_base += K_TILE) {
        const int k_end = min(k_base + K_TILE, K);
        const int k_count = k_end - k_base;

        // Accumulators for K_TILE output channels
        float m_accum[K_TILE];
        #pragma unroll
        for (int i = 0; i < K_TILE; i++) m_accum[i] = 0.0f;

        // Channel reduction loop — input transform computed ONCE per c
        for (int c = 0; c < C; c++) {
            // Input transform: load tile element and compute B^T·tile·B
            float tile_val = 0.0f;
            if (h_idx >= 0 && h_idx < H && w_idx >= 0 && w_idx < W) {
                tile_val = __half2float(input[((b * C + c) * H + h_idx) * W + w_idx]);
            }

            // B^T · tile using wave shuffles
            float temp = 0.0f;
            #pragma unroll
            for (int kk = 0; kk < 8; kk++) {
                float tile_k_col = __shfl(tile_val, kk * 8 + pos_col, 64);
                temp += __half2float(c_BT[pos_row][kk]) * tile_k_col;
            }

            // · B using wave shuffles
            float v_val = 0.0f;
            #pragma unroll
            for (int kk = 0; kk < 8; kk++) {
                float temp_row_k = __shfl(temp, pos_row * 8 + kk, 64);
                v_val += temp_row_k * __half2float(c_B[kk][pos_col]);
            }

            // FMA across K_TILE output channels (unrolled by compiler)
            const __half* u_ptr = U_base + k_base + c * K;
            #pragma unroll
            for (int i = 0; i < K_TILE; i++) {
                if (k_base + i < K) {
                    m_accum[i] += __half2float(u_ptr[i]) * v_val;
                }
            }
        }

        // Output transform and write for each k in this chunk
        for (int i = 0; i < k_count; i++) {
            const int k = k_base + i;
            float m_val = m_accum[i];

            // A · M · A^T via wave shuffles
            float out_temp = 0.0f;
            #pragma unroll
            for (int p = 0; p < 8; p++) {
                float m_p_col = __shfl(m_val, p * 8 + pos_col, 64);
                if (pos_row < 6) out_temp += __half2float(c_A[pos_row][p]) * m_p_col;
            }

            float y_val = 0.0f;
            #pragma unroll
            for (int p = 0; p < 8; p++) {
                float temp_row_p = __shfl(out_temp, pos_row * 8 + p, 64);
                if (pos_row < 6 && pos_col < 6)
                    y_val += temp_row_p * __half2float(c_AT[p][pos_col]);
            }

            if (pos_row < 6 && pos_col < 6) {
                int h_out = th * 6 + pos_row;
                int w_out = tw * 6 + pos_col;
                if (h_out < H_out && w_out < W_out) {
                    output[((b * K + k) * H_out + h_out) * W_out + w_out] = __float2half(y_val);
                }
            }
        }
    }
}


// ═══════════════════════════════════════════════════════════════════════════
// Host wrapper
// ═══════════════════════════════════════════════════════════════════════════

class NovaWinogradFused {
public:
    int K_, C_;
    __half* d_U_gemm_;
    bool initialized_;

    NovaWinogradFused() : K_(0), C_(0), d_U_gemm_(nullptr), initialized_(false) {}

    ~NovaWinogradFused() {
        if (d_U_gemm_) hipFree(d_U_gemm_);
    }

    void set_weights(const float* d_weight, int K, int C) {
        K_ = K;
        C_ = C;
        if (d_U_gemm_) hipFree(d_U_gemm_);
        HIP_CHECK(hipMalloc(&d_U_gemm_, 64 * K * C * sizeof(__half)));
        filter_transform_kernel<<<K * C, 64>>>(d_weight, d_U_gemm_, K, C);
        HIP_CHECK(hipGetLastError());
        initialized_ = true;
    }

    void forward(
        const __half* d_input,
        __half* d_output,
        int batch, int H, int W, int pad,
        hipStream_t stream = 0
    ) {
        if (!initialized_) {
            fprintf(stderr, "ERROR: call set_weights() before forward()\n");
            exit(1);
        }

        const int m = 6, n = 8;
        int H_out = H + 2 * pad - 2;
        int W_out = W + 2 * pad - 2;

        int H_pad = H_out + n - 1;
        int W_pad = W_out + n - 1;
        int extra_h = (m - (H_pad - n) % m) % m;
        int extra_w = (m - (W_pad - n) % m) % m;
        H_pad += extra_h;
        W_pad += extra_w;

        int nh = (H_pad - n) / m + 1;
        int nw = (W_pad - n) / m + 1;
        int P = nh * nw;
        int BP = batch * P;

        // Launch fused kernel: one workgroup (64 threads) per bp value
        dim3 grid(BP);
        dim3 block(64);

        hipLaunchKernelGGL(fused_winograd_kernel, grid, block, 0, stream,
                           d_input, d_U_gemm_, d_output,
                           batch, C_, K_, H, W,
                           pad, nh, nw, H_out, W_out);
        HIP_CHECK(hipGetLastError());
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

// Reference direct conv for validation
__global__ void direct_conv2d_ref(
    const __half* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
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
    for (int c = 0; c < C; c++)
        for (int fh = 0; fh < 3; fh++)
            for (int fw = 0; fw < 3; fw++) {
                int ih = oh - pad + fh;
                int iw = ow - pad + fw;
                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                    sum += __half2float(input[((b * C + c) * H + ih) * W + iw])
                         * weight[(k * C + c) * 9 + fh * 3 + fw];
                }
            }
    output[idx] = sum;
}

int main() {
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    printf("Device: %s (%s)\n", prop.name, prop.gcnArchName);
    printf("CUs: %d\n\n", prop.multiProcessorCount);

    upload_transform_matrices();

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
        {32, 64,  64,  56, 56, 1, "conv2_x (B=32)"},
    };
    int num_configs = sizeof(configs) / sizeof(configs[0]);

    for (int ci = 0; ci < num_configs; ci++) {
        auto& cfg = configs[ci];
        printf("═══ %s ═══\n", cfg.name);

        int nh, nw, H_out, W_out;
        const int m = 6, n = 8;
        H_out = cfg.H + 2 * cfg.pad - 2;
        W_out = cfg.W + 2 * cfg.pad - 2;
        int H_pad = H_out + n - 1, W_pad = W_out + n - 1;
        int extra_h = (m - (H_pad - n) % m) % m;
        int extra_w = (m - (W_pad - n) % m) % m;
        H_pad += extra_h; W_pad += extra_w;
        nh = (H_pad - n) / m + 1; nw = (W_pad - n) / m + 1;
        int P = nh * nw;
        printf("  Tiles: %dx%d=%d, Output: %dx%d\n", nh, nw, P, H_out, W_out);

        std::mt19937 rng(42 + ci);
        std::normal_distribution<float> dist(0.0f, 0.5f);

        int in_size = cfg.batch * cfg.C * cfg.H * cfg.W;
        int wt_size = cfg.K * cfg.C * 9;
        int out_size = cfg.batch * cfg.K * H_out * W_out;

        std::vector<__half> h_input(in_size);
        std::vector<float> h_weight(wt_size);
        for (auto& v : h_input) v = __float2half(dist(rng));
        for (auto& v : h_weight) v = dist(rng);

        __half* d_input; float* d_weight;
        HIP_CHECK(hipMalloc(&d_input, in_size * sizeof(__half)));
        HIP_CHECK(hipMalloc(&d_weight, wt_size * sizeof(float)));
        HIP_CHECK(hipMemcpy(d_input, h_input.data(), in_size * sizeof(__half), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_weight, h_weight.data(), wt_size * sizeof(float), hipMemcpyHostToDevice));

        // Reference
        float* d_ref;
        HIP_CHECK(hipMalloc(&d_ref, out_size * sizeof(float)));
        direct_conv2d_ref<<<(out_size + 255) / 256, 256>>>(
            d_input, d_weight, d_ref, cfg.batch, cfg.C, cfg.H, cfg.W, cfg.K, H_out, W_out, cfg.pad);
        HIP_CHECK(hipDeviceSynchronize());

        // Fused NOVA
        NovaWinogradFused nova;
        nova.set_weights(d_weight, cfg.K, cfg.C);
        HIP_CHECK(hipDeviceSynchronize());

        __half* d_output;
        HIP_CHECK(hipMalloc(&d_output, out_size * sizeof(__half)));

        // Warmup
        nova.forward(d_input, d_output, cfg.batch, cfg.H, cfg.W, cfg.pad);
        HIP_CHECK(hipDeviceSynchronize());

        // Timed
        int repeats = 20;
        hipEvent_t start, stop;
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));
        HIP_CHECK(hipEventRecord(start));
        for (int r = 0; r < repeats; r++) {
            nova.forward(d_input, d_output, cfg.batch, cfg.H, cfg.W, cfg.pad);
        }
        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));
        float total_ms = 0;
        HIP_CHECK(hipEventElapsedTime(&total_ms, start, stop));
        float avg_ms = total_ms / repeats;

        // Verify
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
        double flops = 2.0 * cfg.batch * cfg.K * cfg.C * H_out * W_out * 9;

        printf("  Time: %.3f ms (%.1f GFLOPS)\n", avg_ms, flops / (avg_ms * 1e6));
        printf("  Relative error: %.6e, Max abs: %.6e\n", rel_err, max_err);
        printf("  NaN: %d, Inf: %d\n", nan_count, inf_count);
        printf("  Status: %s\n\n", (nan_count == 0 && inf_count == 0 && rel_err < 0.06) ? "PASS" : "FAIL");

        HIP_CHECK(hipFree(d_input));
        HIP_CHECK(hipFree(d_weight));
        HIP_CHECK(hipFree(d_ref));
        HIP_CHECK(hipFree(d_output));
        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));
    }
    return 0;
}

#endif
