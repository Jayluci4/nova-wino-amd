/**
 * NOVA Winograd F(6,3) — BP-Tiled Pipeline + MFMA GEMM
 *
 * Key optimization: tile BP dimension so intermediate buffers (V_gemm, M_gemm)
 * fit in L2 cache (4MB per XCD on MI300X).
 *
 * Traffic analysis:
 *   Original: read input + write V + read V + GEMM internal + write M + read M + write output
 *   Tiled:    read input + (V stays in L2) + GEMM + (M stays in L2) + write output
 *   Reduction: eliminates ~2× (V_size + M_size) of HBM traffic
 *
 * Build:
 *   hipcc -o test_bp_tiled test_bp_tiled.hip -std=c++17 -L/opt/rocm/lib -lrocblas
 *         -I/opt/rocm/include --offload-arch=gfx942 -O3
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <rocblas/rocblas.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <random>
#include <vector>

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

typedef _Float16 fp16x4_t __attribute__((ext_vector_type(4)));
typedef float    fp32x4_t __attribute__((ext_vector_type(4)));

// ═══════════════════════════════════════════════════════════════════════════
// Transform matrices
// ═══════════════════════════════════════════════════════════════════════════
__constant__ __half c_BT[8][8];
__constant__ __half c_B[8][8];
__constant__ float c_G[8][3];
__constant__ float c_GT[3][8];
__constant__ __half c_A[6][8];
__constant__ __half c_AT[8][6];

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
// Transform kernels (from original nova_winograd.hip)
// ═══════════════════════════════════════════════════════════════════════════

#define TILES_PER_WG 4

__global__ __launch_bounds__(64, 16)
void filter_transform_kernel(
    const float* __restrict__ weight, __half* __restrict__ U_gemm, int K, int C)
{
    const int tid = threadIdx.x;
    const int row = tid >> 3, col = tid & 7;
    const int global_idx = blockIdx.x;
    if (global_idx >= K * C) return;
    const int k = global_idx / C, c = global_idx % C;
    float w[3][3];
    const float* wp = weight + (k * C + c) * 9;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) w[i][j] = wp[i * 3 + j];
    float result = 0.0f;
    #pragma unroll
    for (int p = 0; p < 3; p++) {
        float g_row_p = c_G[row][p];
        #pragma unroll
        for (int q = 0; q < 3; q++) result += g_row_p * w[p][q] * c_GT[q][col];
    }
    U_gemm[tid * K * C + k + c * K] = __float2half(result);
}

// Input transform: operates on a BP-tile (bp_offset..bp_offset+bp_count-1)
// V_gemm layout: [64, C, bp_count] col-major per pos
__global__ __launch_bounds__(64 * TILES_PER_WG, 4)
void input_transform_tiled(
    const __half* __restrict__ input,
    __half* __restrict__ V_gemm,
    int batch, int C, int H, int W,
    int pad, int nh, int nw,
    int bp_offset, int bp_count,
    int total_tiles)
{
    const int wf_id = threadIdx.x >> 6;
    const int lane = threadIdx.x & 63;
    const int row = lane >> 3, col = lane & 7;

    const int global_idx = blockIdx.x * TILES_PER_WG + wf_id;
    if (global_idx >= total_tiles) return;

    const int P = nh * nw;

    // Decode: which (b, c, tile_idx) does this global_idx map to?
    // We process bp values [bp_offset .. bp_offset+bp_count-1]
    // global_idx = c * bp_count + bp_local  (iterate over C × bp_count)
    const int c = global_idx / bp_count;
    const int bp_local = global_idx % bp_count;

    if (c >= C) return;

    const int bp_global = bp_offset + bp_local;
    const int b = bp_global / P;
    const int tile_idx = bp_global % P;
    const int th = tile_idx / nw, tw = tile_idx % nw;

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

    // Write to TILED V_gemm: [64, C, bp_count] col-major per pos
    const int pos = lane;
    V_gemm[(long long)pos * C * bp_count + c + (long long)bp_local * C] = __float2half(v_val);
}

// Output transform: operates on a BP-tile
__global__ __launch_bounds__(64 * TILES_PER_WG, 4)
void output_transform_tiled(
    const __half* __restrict__ M_gemm,
    __half* __restrict__ output,
    int batch, int K, int H_out, int W_out,
    int nh, int nw,
    int bp_offset, int bp_count,
    int total_tiles)
{
    const int wf_id = threadIdx.x >> 6;
    const int lane = threadIdx.x & 63;
    const int row = lane >> 3, col = lane & 7;

    const int global_idx = blockIdx.x * TILES_PER_WG + wf_id;
    if (global_idx >= total_tiles) return;

    const int P = nh * nw;

    // Decode: global_idx = k * bp_count + bp_local
    const int k = global_idx / bp_count;
    const int bp_local = global_idx % bp_count;
    if (k >= K) return;

    const int bp_global = bp_offset + bp_local;
    const int b = bp_global / P;
    const int tile_idx = bp_global % P;
    const int th = tile_idx / nw, tw = tile_idx % nw;

    // Read from TILED M_gemm: [64, K, bp_count] col-major per pos
    const int pos = lane;
    float m_val = __half2float(M_gemm[(long long)pos * K * bp_count + k + (long long)bp_local * K]);

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

// Original-style input transform (full BP)
__global__ __launch_bounds__(64 * TILES_PER_WG, 4)
void input_transform_kernel_mt(
    const __half* __restrict__ input, __half* __restrict__ V_gemm,
    int batch, int C, int H, int W, int pad, int nh, int nw, int BP, int total_tiles)
{
    const int wf_id = threadIdx.x >> 6;
    const int lane = threadIdx.x & 63;
    const int row = lane >> 3, col = lane & 7;
    const int global_idx = blockIdx.x * TILES_PER_WG + wf_id;
    if (global_idx >= total_tiles) return;
    const int P = nh * nw;
    const int b = global_idx / (C * P);
    const int rem = global_idx % (C * P);
    const int c = rem / P;
    const int tile_idx = rem % P;
    const int th = tile_idx / nw, tw = tile_idx % nw;
    const int h_start = th * 6 - pad, w_start = tw * 6 - pad;
    const int h_idx = h_start + row, w_idx = w_start + col;
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
    V_gemm[(long long)lane * C * BP + c + (long long)(b * P + tile_idx) * C] = __float2half(v_val);
}

// Original-style output transform (full BP)
__global__ __launch_bounds__(64 * TILES_PER_WG, 4)
void output_transform_kernel_mt(
    const __half* __restrict__ M_gemm, __half* __restrict__ output,
    int batch, int K, int H_out, int W_out, int nh, int nw, int BP, int total_tiles)
{
    const int wf_id = threadIdx.x >> 6;
    const int lane = threadIdx.x & 63;
    const int row = lane >> 3, col = lane & 7;
    const int global_idx = blockIdx.x * TILES_PER_WG + wf_id;
    if (global_idx >= total_tiles) return;
    const int P = nh * nw;
    const int b = global_idx / (K * P);
    const int rem = global_idx % (K * P);
    const int k = rem / P;
    const int tile_idx = rem % P;
    const int th = tile_idx / nw, tw = tile_idx % nw;
    float m_val = __half2float(M_gemm[(long long)lane * K * BP + k + (long long)(b * P + tile_idx) * K]);
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
        int h_out = th * 6 + row, w_out = tw * 6 + col;
        if (h_out < H_out && w_out < W_out)
            output[((b * K + k) * H_out + h_out) * W_out + w_out] = __float2half(y_val);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// MFMA Batched GEMM kernel
// ═══════════════════════════════════════════════════════════════════════════
__global__ __launch_bounds__(64, 8)
void mfma_batched_gemm(
    const __half* __restrict__ U,   // [64, K, C] col-major per pos
    const __half* __restrict__ V,   // [64, C, bp_count] col-major per pos
    __half* __restrict__ M,          // [64, K, bp_count] col-major per pos
    int K, int C, int bp_count)
{
    const int pos     = blockIdx.z;
    const int k_tile  = blockIdx.x;
    const int bp_tile = blockIdx.y;
    const int t       = threadIdx.x;
    const int k_base  = k_tile * 16;
    const int bp_base = bp_tile * 16;
    const int t_row   = t % 16;
    const int t_grp   = t / 16;

    const __half* U_pos = U + (long long)pos * K * C;
    const __half* V_pos = V + (long long)pos * C * bp_count;
    __half*       M_pos = M + (long long)pos * K * bp_count;

    __shared__ __half V_lds[16][17];

    fp32x4_t acc = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int c_base = 0; c_base < C; c_base += 16) {
        #pragma unroll
        for (int iter = 0; iter < 4; iter++) {
            int flat     = iter * 64 + t;
            int c_local  = flat % 16;
            int bp_local = flat / 16;
            int c_g      = c_base + c_local;
            int bp_g     = bp_base + bp_local;
            __half val   = (__half)0;
            if (c_g < C && bp_g < bp_count)
                val = V_pos[c_g + (long long)bp_g * C];
            V_lds[c_local][bp_local] = val;
        }

        fp16x4_t a;
        int k_g = k_base + t_row;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int c_g = c_base + 4 * t_grp + i;
            if (k_g < K && c_g < C)
                a[i] = (_Float16)U_pos[k_g + (long long)c_g * K];
            else
                a[i] = (_Float16)0;
        }

        fp16x4_t b;
        #pragma unroll
        for (int i = 0; i < 4; i++)
            b[i] = (_Float16)V_lds[4 * t_grp + i][t % 16];

        acc = __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, acc, 0, 0, 0);
    }

    int bp_g = bp_base + (t % 16);
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int k_g = k_base + 4 * t_grp + i;
        if (k_g < K && bp_g < bp_count)
            M_pos[k_g + (long long)bp_g * K] = __float2half(acc[i]);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Reference direct conv
// ═══════════════════════════════════════════════════════════════════════════
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

// ═══════════════════════════════════════════════════════════════════════════
// Pipeline implementations
// ═══════════════════════════════════════════════════════════════════════════

struct NovaContext {
    __half* d_U_gemm;
    rocblas_handle rb_handle;
    int K, C;
};

void compute_tiling(int H, int W, int pad, int* nh, int* nw, int* H_out, int* W_out) {
    const int m = 6, n = 8;
    *H_out = H + 2 * pad - 2;
    *W_out = W + 2 * pad - 2;
    int H_pad = *H_out + n - 1, W_pad = *W_out + n - 1;
    int extra_h = (m - (H_pad - n) % m) % m;
    int extra_w = (m - (W_pad - n) % m) % m;
    H_pad += extra_h; W_pad += extra_w;
    *nh = (H_pad - n) / m + 1;
    *nw = (W_pad - n) / m + 1;
}

// Original 3-pass (full BP, rocBLAS)
void forward_original(
    NovaContext& ctx, const __half* d_input, __half* d_output,
    int batch, int H, int W, int pad, hipStream_t stream = 0)
{
    int nh, nw, H_out, W_out;
    compute_tiling(H, W, pad, &nh, &nw, &H_out, &W_out);
    int P = nh * nw, BP = batch * P;

    __half *d_V, *d_M;
    HIP_CHECK(hipMalloc(&d_V, 64LL * ctx.C * BP * 2));
    HIP_CHECK(hipMalloc(&d_M, 64LL * ctx.K * BP * 2));

    rocblas_set_stream(ctx.rb_handle, stream);

    int in_tiles = batch * ctx.C * P;
    int in_blocks = (in_tiles + TILES_PER_WG - 1) / TILES_PER_WG;
    hipLaunchKernelGGL(input_transform_kernel_mt, dim3(in_blocks), dim3(64 * TILES_PER_WG),
                       0, stream, d_input, d_V, batch, ctx.C, H, W, pad, nh, nw, BP, in_tiles);

    float alpha = 1.0f, beta = 0.0f;
    ROCBLAS_CHECK(rocblas_gemm_strided_batched_ex(
        ctx.rb_handle, rocblas_operation_none, rocblas_operation_none,
        ctx.K, BP, ctx.C, &alpha,
        ctx.d_U_gemm, rocblas_datatype_f16_r, ctx.K, (rocblas_stride)((long long)ctx.K * ctx.C),
        d_V, rocblas_datatype_f16_r, ctx.C, (rocblas_stride)((long long)ctx.C * BP),
        &beta,
        d_M, rocblas_datatype_f16_r, ctx.K, (rocblas_stride)((long long)ctx.K * BP),
        d_M, rocblas_datatype_f16_r, ctx.K, (rocblas_stride)((long long)ctx.K * BP),
        64, rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0));

    int out_tiles = batch * ctx.K * P;
    int out_blocks = (out_tiles + TILES_PER_WG - 1) / TILES_PER_WG;
    hipLaunchKernelGGL(output_transform_kernel_mt, dim3(out_blocks), dim3(64 * TILES_PER_WG),
                       0, stream, d_M, d_output, batch, ctx.K, H_out, W_out, nh, nw, BP, out_tiles);

    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(d_V));
    HIP_CHECK(hipFree(d_M));
}

// BP-tiled 3-pass with rocBLAS
void forward_tiled_rocblas(
    NovaContext& ctx, const __half* d_input, __half* d_output,
    int batch, int H, int W, int pad, int bp_tile_size, hipStream_t stream = 0)
{
    int nh, nw, H_out, W_out;
    compute_tiling(H, W, pad, &nh, &nw, &H_out, &W_out);
    int P = nh * nw, BP = batch * P;

    // Workspace sized for ONE tile (fits in L2)
    __half *d_V, *d_M;
    HIP_CHECK(hipMalloc(&d_V, 64LL * ctx.C * bp_tile_size * 2));
    HIP_CHECK(hipMalloc(&d_M, 64LL * ctx.K * bp_tile_size * 2));

    rocblas_set_stream(ctx.rb_handle, stream);

    for (int bp_off = 0; bp_off < BP; bp_off += bp_tile_size) {
        int bp_count = std::min(bp_tile_size, BP - bp_off);

        // Input transform for this tile
        int in_tiles = ctx.C * bp_count;
        int in_blocks = (in_tiles + TILES_PER_WG - 1) / TILES_PER_WG;
        hipLaunchKernelGGL(input_transform_tiled, dim3(in_blocks), dim3(64 * TILES_PER_WG),
                           0, stream, d_input, d_V, batch, ctx.C, H, W, pad, nh, nw,
                           bp_off, bp_count, in_tiles);

        // GEMM for this tile: M[K×bp_count] = U[K×C] × V[C×bp_count] × 64 batches
        float alpha = 1.0f, beta = 0.0f;
        ROCBLAS_CHECK(rocblas_gemm_strided_batched_ex(
            ctx.rb_handle, rocblas_operation_none, rocblas_operation_none,
            ctx.K, bp_count, ctx.C, &alpha,
            ctx.d_U_gemm, rocblas_datatype_f16_r, ctx.K, (rocblas_stride)((long long)ctx.K * ctx.C),
            d_V, rocblas_datatype_f16_r, ctx.C, (rocblas_stride)((long long)ctx.C * bp_count),
            &beta,
            d_M, rocblas_datatype_f16_r, ctx.K, (rocblas_stride)((long long)ctx.K * bp_count),
            d_M, rocblas_datatype_f16_r, ctx.K, (rocblas_stride)((long long)ctx.K * bp_count),
            64, rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0));

        // Output transform for this tile
        int out_tiles = ctx.K * bp_count;
        int out_blocks = (out_tiles + TILES_PER_WG - 1) / TILES_PER_WG;
        hipLaunchKernelGGL(output_transform_tiled, dim3(out_blocks), dim3(64 * TILES_PER_WG),
                           0, stream, d_M, d_output, batch, ctx.K, H_out, W_out, nh, nw,
                           bp_off, bp_count, out_tiles);
    }

    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(d_V));
    HIP_CHECK(hipFree(d_M));
}

// BP-tiled 3-pass with MFMA GEMM
void forward_tiled_mfma(
    NovaContext& ctx, const __half* d_input, __half* d_output,
    int batch, int H, int W, int pad, int bp_tile_size, hipStream_t stream = 0)
{
    int nh, nw, H_out, W_out;
    compute_tiling(H, W, pad, &nh, &nw, &H_out, &W_out);
    int P = nh * nw, BP = batch * P;

    __half *d_V, *d_M;
    HIP_CHECK(hipMalloc(&d_V, 64LL * ctx.C * bp_tile_size * 2));
    HIP_CHECK(hipMalloc(&d_M, 64LL * ctx.K * bp_tile_size * 2));

    for (int bp_off = 0; bp_off < BP; bp_off += bp_tile_size) {
        int bp_count = std::min(bp_tile_size, BP - bp_off);

        // Input transform
        int in_tiles = ctx.C * bp_count;
        int in_blocks = (in_tiles + TILES_PER_WG - 1) / TILES_PER_WG;
        hipLaunchKernelGGL(input_transform_tiled, dim3(in_blocks), dim3(64 * TILES_PER_WG),
                           0, stream, d_input, d_V, batch, ctx.C, H, W, pad, nh, nw,
                           bp_off, bp_count, in_tiles);

        // MFMA GEMM
        dim3 grid((ctx.K + 15) / 16, (bp_count + 15) / 16, 64);
        hipLaunchKernelGGL(mfma_batched_gemm, grid, dim3(64), 0, stream,
                           ctx.d_U_gemm, d_V, d_M, ctx.K, ctx.C, bp_count);

        // Output transform
        int out_tiles = ctx.K * bp_count;
        int out_blocks = (out_tiles + TILES_PER_WG - 1) / TILES_PER_WG;
        hipLaunchKernelGGL(output_transform_tiled, dim3(out_blocks), dim3(64 * TILES_PER_WG),
                           0, stream, d_M, d_output, batch, ctx.K, H_out, W_out, nh, nw,
                           bp_off, bp_count, out_tiles);
    }

    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(d_V));
    HIP_CHECK(hipFree(d_M));
}

// ═══════════════════════════════════════════════════════════════════════════
// Main: correctness + benchmark
// ═══════════════════════════════════════════════════════════════════════════

int main() {
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    printf("Device: %s (%s), %d CUs\n\n", prop.name, prop.gcnArchName, prop.multiProcessorCount);
    upload_transform_matrices();

    struct Config { int batch, C, K, H, W, pad; const char* name; };
    Config configs[] = {
        {1,   64,  64,  56, 56, 1, "conv2_x B=1 "},
        {1,  128, 128,  28, 28, 1, "conv3_x B=1 "},
        {1,  256, 256,  14, 14, 1, "conv4_x B=1 "},
        {8,   64,  64,  56, 56, 1, "conv2_x B=8 "},
        {8,  128, 128,  28, 28, 1, "conv3_x B=8 "},
        {8,  256, 256,  14, 14, 1, "conv4_x B=8 "},
        {32,  64,  64,  56, 56, 1, "conv2_x B=32"},
        {32, 128, 128,  28, 28, 1, "conv3_x B=32"},
        {32, 256, 256,  14, 14, 1, "conv4_x B=32"},
    };
    int num_configs = sizeof(configs) / sizeof(configs[0]);

    // Choose BP_TILE to fit V+M in ~4MB (one XCD's L2)
    // V = 64*C*BP_TILE*2, M = 64*K*BP_TILE*2
    // Total = 128*(C+K)*BP_TILE bytes
    // 4MB = 4194304 → BP_TILE = 4194304 / (128*(C+K))

    printf("%-14s | %8s %8s %8s | V+M buf | BP tile\n",
           "Config", "Original", "Tiled-rB", "Tiled-MFMA");
    printf("──────────────────────────────────────────────────────────────────\n");

    for (int ci = 0; ci < num_configs; ci++) {
        auto& cfg = configs[ci];
        int nh, nw, H_out, W_out;
        compute_tiling(cfg.H, cfg.W, cfg.pad, &nh, &nw, &H_out, &W_out);
        int P = nh * nw, BP = cfg.batch * P;

        // Choose BP_TILE
        int bp_tile = 4194304 / (128 * (cfg.C + cfg.K));
        bp_tile = std::max(16, std::min(bp_tile, BP));  // at least 16, at most BP

        long long V_M_bytes = 128LL * (cfg.C + cfg.K) * BP;

        // Setup
        std::mt19937 rng(42 + ci);
        std::normal_distribution<float> dist(0.f, 0.3f);
        int in_size = cfg.batch * cfg.C * cfg.H * cfg.W;
        int wt_size = cfg.K * cfg.C * 9;
        int out_size = cfg.batch * cfg.K * H_out * W_out;

        std::vector<__half> h_input(in_size);
        std::vector<float> h_weight(wt_size);
        for (auto& v : h_input) v = __float2half(dist(rng));
        for (auto& v : h_weight) v = dist(rng);

        __half* d_input; float* d_weight;
        HIP_CHECK(hipMalloc(&d_input, in_size * 2));
        HIP_CHECK(hipMalloc(&d_weight, wt_size * 4));
        HIP_CHECK(hipMemcpy(d_input, h_input.data(), in_size * 2, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_weight, h_weight.data(), wt_size * 4, hipMemcpyHostToDevice));

        // Create context
        NovaContext ctx;
        rocblas_create_handle(&ctx.rb_handle);
        ctx.K = cfg.K; ctx.C = cfg.C;
        HIP_CHECK(hipMalloc(&ctx.d_U_gemm, 64LL * cfg.K * cfg.C * 2));
        filter_transform_kernel<<<cfg.K * cfg.C, 64>>>(d_weight, ctx.d_U_gemm, cfg.K, cfg.C);
        HIP_CHECK(hipDeviceSynchronize());

        // Reference (direct conv)
        float* d_ref;
        HIP_CHECK(hipMalloc(&d_ref, out_size * 4));
        direct_conv2d_ref<<<(out_size + 255) / 256, 256>>>(
            d_input, d_weight, d_ref, cfg.batch, cfg.C, cfg.H, cfg.W, cfg.K, H_out, W_out, cfg.pad);
        HIP_CHECK(hipDeviceSynchronize());

        // Test correctness of tiled MFMA pipeline
        __half* d_out_test;
        HIP_CHECK(hipMalloc(&d_out_test, out_size * 2));
        forward_tiled_mfma(ctx, d_input, d_out_test, cfg.batch, cfg.H, cfg.W, cfg.pad, bp_tile);
        std::vector<__half> h_out(out_size);
        std::vector<float> h_ref(out_size);
        HIP_CHECK(hipMemcpy(h_out.data(), d_out_test, out_size * 2, hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(h_ref.data(), d_ref, out_size * 4, hipMemcpyDeviceToHost));
        double sse = 0, sse_ref = 0;
        int nans = 0;
        for (int i = 0; i < out_size; i++) {
            float v = __half2float(h_out[i]), r = h_ref[i];
            if (std::isnan(v)) { nans++; continue; }
            sse += (double)(v - r) * (v - r);
            sse_ref += (double)r * r;
        }
        float rel_err = (float)sqrt(sse / (sse_ref + 1e-12));
        bool pass = (nans == 0 && rel_err < 0.06f);
        if (!pass) {
            printf("%-14s  FAIL (rel=%.3e nan=%d)\n", cfg.name, rel_err, nans);
            goto cleanup;
        }

        {
            // Benchmark all three approaches
            __half *d_out1, *d_out2, *d_out3;
            HIP_CHECK(hipMalloc(&d_out1, out_size * 2));
            HIP_CHECK(hipMalloc(&d_out2, out_size * 2));
            HIP_CHECK(hipMalloc(&d_out3, out_size * 2));

            hipEvent_t start, stop;
            HIP_CHECK(hipEventCreate(&start));
            HIP_CHECK(hipEventCreate(&stop));
            int repeats = 20;

            // Warmup
            forward_original(ctx, d_input, d_out1, cfg.batch, cfg.H, cfg.W, cfg.pad);
            forward_tiled_rocblas(ctx, d_input, d_out2, cfg.batch, cfg.H, cfg.W, cfg.pad, bp_tile);
            forward_tiled_mfma(ctx, d_input, d_out3, cfg.batch, cfg.H, cfg.W, cfg.pad, bp_tile);

            // Original
            HIP_CHECK(hipEventRecord(start));
            for (int r = 0; r < repeats; r++)
                forward_original(ctx, d_input, d_out1, cfg.batch, cfg.H, cfg.W, cfg.pad);
            HIP_CHECK(hipEventRecord(stop));
            HIP_CHECK(hipEventSynchronize(stop));
            float ms_orig = 0;
            HIP_CHECK(hipEventElapsedTime(&ms_orig, start, stop));
            ms_orig /= repeats;

            // Tiled rocBLAS
            HIP_CHECK(hipEventRecord(start));
            for (int r = 0; r < repeats; r++)
                forward_tiled_rocblas(ctx, d_input, d_out2, cfg.batch, cfg.H, cfg.W, cfg.pad, bp_tile);
            HIP_CHECK(hipEventRecord(stop));
            HIP_CHECK(hipEventSynchronize(stop));
            float ms_tiled_rb = 0;
            HIP_CHECK(hipEventElapsedTime(&ms_tiled_rb, start, stop));
            ms_tiled_rb /= repeats;

            // Tiled MFMA
            HIP_CHECK(hipEventRecord(start));
            for (int r = 0; r < repeats; r++)
                forward_tiled_mfma(ctx, d_input, d_out3, cfg.batch, cfg.H, cfg.W, cfg.pad, bp_tile);
            HIP_CHECK(hipEventRecord(stop));
            HIP_CHECK(hipEventSynchronize(stop));
            float ms_tiled_mfma = 0;
            HIP_CHECK(hipEventElapsedTime(&ms_tiled_mfma, start, stop));
            ms_tiled_mfma /= repeats;

            printf("%-14s | %6.3fms %6.3fms %6.3fms | %5.1fMB | %4d  %s\n",
                   cfg.name, ms_orig, ms_tiled_rb, ms_tiled_mfma,
                   V_M_bytes / 1048576.0, bp_tile, pass ? "PASS" : "FAIL");

            HIP_CHECK(hipFree(d_out1));
            HIP_CHECK(hipFree(d_out2));
            HIP_CHECK(hipFree(d_out3));
            HIP_CHECK(hipEventDestroy(start));
            HIP_CHECK(hipEventDestroy(stop));
        }

cleanup:
        HIP_CHECK(hipFree(d_input));
        HIP_CHECK(hipFree(d_weight));
        HIP_CHECK(hipFree(d_ref));
        HIP_CHECK(hipFree(d_out_test));
        HIP_CHECK(hipFree(ctx.d_U_gemm));
        rocblas_destroy_handle(ctx.rb_handle);
    }

    return 0;
}
