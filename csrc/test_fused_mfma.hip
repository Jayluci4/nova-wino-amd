/**
 * NOVA Winograd F(6,3) — Fully Fused MFMA Kernel
 *
 * Architecture:
 *   Workgroup: 1024 threads = 16 wavefronts
 *   BP_PER_WG = 16 (each wavefront handles 1 bp value for transforms)
 *
 *   Phase 1 (Input Transform): 16 wavefronts do __shfl-based B^T·d·B
 *     → V[c_local][bp_local=wf_id] per position stored in LDS
 *
 *   Phase 2 (MFMA GEMM): 16 wavefronts handle 4 positions each (4 batches)
 *     → MFMA accumulate M[k][bp] = Σ_c U[k][c] × V[c][bp] per position
 *
 *   Phase 3 (Output Transform): Write M to LDS, wavefronts switch to 1-bp-each,
 *     → __shfl-based A·M·A^T, write to output
 *
 *   Zero intermediate global memory. Read input + U_gemm, write output.
 *
 * Build:
 *   hipcc -o test_fused_mfma test_fused_mfma.hip -std=c++17 -L/opt/rocm/lib
 *         -lrocblas -I/opt/rocm/include --offload-arch=gfx942 -O3
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <rocblas/rocblas.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>
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
// Filter Transform
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
// Fully Fused MFMA Winograd Kernel
//
// Workgroup: 1024 threads = 16 wavefronts
// Grid: ceil(BP / 16) workgroups
//
// LDS layout for V:
//   V_lds[64_pos][16_c_local][17_bp_padded]  (FP16)
//   Size: 64 × 16 × 17 × 2 = 34816 bytes ≈ 34KB
//   Padding of 17 (not 16) reduces LDS bank conflicts for MFMA reads
//
// LDS layout for M (reuses V space):
//   M_lds[16_k_local][16_bp][64_pos]  (FP16)
//   Size: 16 × 16 × 64 × 2 = 32768 bytes = 32KB
// ═══════════════════════════════════════════════════════════════════════════

#define BP_PER_WG 16
#define C_CHUNK   16   // channels per LDS staging round
#define POS_PER_WF 4   // positions per wavefront in Phase 2
#define K_SUPER   64   // max k values accumulated before output transform

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

            // Write M to LDS
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

            // Output transform per k
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
// Test
// ═══════════════════════════════════════════════════════════════════════════
void compute_tiling(int H, int W, int pad, int* nh, int* nw, int* H_out, int* W_out) {
    const int m = 6, n = 8;
    *H_out = H + 2 * pad - 2; *W_out = W + 2 * pad - 2;
    int Hp = *H_out + n - 1, Wp = *W_out + n - 1;
    Hp += (m - (Hp - n) % m) % m;
    Wp += (m - (Wp - n) % m) % m;
    *nh = (Hp - n) / m + 1; *nw = (Wp - n) / m + 1;
}

int main() {
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    printf("Device: %s (%s), %d CUs\n", prop.name, prop.gcnArchName, prop.multiProcessorCount);
    printf("Max LDS: %zu bytes\n\n", prop.sharedMemPerBlock);

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

    int lds_size = 64 * C_CHUNK * 17 * 2;  // V_lds size in bytes
    printf("LDS per workgroup: %d bytes (%.1f KB)\n\n", lds_size, lds_size / 1024.0);

    printf("%-14s | %7s %9s | %s\n", "Config", "Time", "GFLOPS", "Status");
    printf("──────────────────────────────────────────────────────\n");

    for (int ci = 0; ci < num_configs; ci++) {
        auto& cfg = configs[ci];
        int nh, nw, H_out, W_out;
        compute_tiling(cfg.H, cfg.W, cfg.pad, &nh, &nw, &H_out, &W_out);
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

        __half* d_in; float* d_wt; float* d_ref; __half* d_out;
        HIP_CHECK(hipMalloc(&d_in, in_sz * 2));
        HIP_CHECK(hipMalloc(&d_wt, wt_sz * 4));
        HIP_CHECK(hipMalloc(&d_ref, out_sz * 4));
        HIP_CHECK(hipMalloc(&d_out, out_sz * 2));
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

        // Fused MFMA kernel — 1D grid over BP tiles
        dim3 grid((BP + BP_PER_WG - 1) / BP_PER_WG);
        dim3 block(1024);
        int lds_bytes = 64 * C_CHUNK * 17 * sizeof(__half);
        // M_lds needs: 16 * 16 * 64 * 2 = 32768 bytes. V_lds needs 34816.
        // Use max of both
        lds_bytes = std::max(lds_bytes, 16 * 16 * 64 * (int)sizeof(__half));

        // Warmup
        hipLaunchKernelGGL(fused_winograd_mfma_kernel, grid, block, lds_bytes, 0,
                           d_in, d_U, d_out, cfg.batch, cfg.C, cfg.K, cfg.H, cfg.W,
                           cfg.pad, nh, nw, H_out, W_out);
        HIP_CHECK(hipDeviceSynchronize());

        // Verify
        std::vector<__half> h_out(out_sz);
        std::vector<float> h_ref(out_sz);
        HIP_CHECK(hipMemcpy(h_out.data(), d_out, out_sz * 2, hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(h_ref.data(), d_ref, out_sz * 4, hipMemcpyDeviceToHost));

        double sse = 0, sse_ref = 0;
        int nans = 0, infs = 0;
        float max_err = 0;
        for (int i = 0; i < out_sz; i++) {
            float v = __half2float(h_out[i]), r = h_ref[i];
            if (std::isnan(v)) { nans++; continue; }
            if (std::isinf(v)) { infs++; continue; }
            float err = fabsf(v - r);
            if (err > max_err) max_err = err;
            sse += (double)(v - r) * (v - r);
            sse_ref += (double)r * r;
        }
        float rel = (float)sqrt(sse / (sse_ref + 1e-12));
        bool pass = (nans == 0 && infs == 0 && rel < 0.06f);

        if (!pass) {
            printf("%-14s | FAIL rel=%.3e max=%.3e nan=%d inf=%d\n",
                   cfg.name, rel, max_err, nans, infs);
            goto next;
        }

        {
            // Benchmark
            hipEvent_t start, stop;
            HIP_CHECK(hipEventCreate(&start));
            HIP_CHECK(hipEventCreate(&stop));
            int repeats = 20;

            HIP_CHECK(hipEventRecord(start));
            for (int r = 0; r < repeats; r++)
                hipLaunchKernelGGL(fused_winograd_mfma_kernel, grid, block, lds_bytes, 0,
                                   d_in, d_U, d_out, cfg.batch, cfg.C, cfg.K, cfg.H, cfg.W,
                                   cfg.pad, nh, nw, H_out, W_out);
            HIP_CHECK(hipEventRecord(stop));
            HIP_CHECK(hipEventSynchronize(stop));
            float ms = 0;
            HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
            ms /= repeats;

            double flops = 2.0 * cfg.batch * cfg.K * cfg.C * H_out * W_out * 9;
            printf("%-14s | %5.3fms %7.1f G | %s  (rel=%.3e, WGs=%d)\n",
                   cfg.name, ms, flops / (ms * 1e6), pass ? "PASS" : "FAIL", rel, grid.x);

            HIP_CHECK(hipEventDestroy(start));
            HIP_CHECK(hipEventDestroy(stop));
        }

next:
        HIP_CHECK(hipFree(d_in));
        HIP_CHECK(hipFree(d_wt));
        HIP_CHECK(hipFree(d_ref));
        HIP_CHECK(hipFree(d_out));
        HIP_CHECK(hipFree(d_U));
    }
    return 0;
}
