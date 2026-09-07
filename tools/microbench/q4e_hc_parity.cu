// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// I-3 Hyper-Connections (Qwen4-Exp GatedResidual) standalone parity harness.
// Runs the GatedResidual forward on the GPU with the exact kernels that will be
// lifted into Qwen4ExpBackend, and checks them against the numpy reference dump
// from q4e_hc_ref.py. This proves the HC math (grouped RMSNorm, low-rank input
// mix, weighted stream mean, block injection) in ISOLATION before it is wired
// into the 4-stream forward — the (c) plan for 5.27 I-3.
//
// Build (on the box, in the builder-cuda container):
//   nvcc -O2 -arch=sm_121 tools/microbench/q4e_hc_parity.cu -o /tmp/q4e_hc_parity
// Run:
//   /tmp/q4e_hc_parity /opt/mimirmind/q4e_hc_dump
//
// Deliberately dependency-free (CUDA runtime only, naive GEMMs) so it is a
// disposable single-purpose harness (ncu/compute-sanitizer-safe per the box
// rules). All math in float32 to isolate correctness from bf16 rounding.

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

#define CK(call)                                                              \
    do {                                                                      \
        cudaError_t _e = (call);                                             \
        if (_e != cudaSuccess) {                                             \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n",                 \
                         cudaGetErrorString(_e), __FILE__, __LINE__);        \
            std::exit(2);                                                    \
        }                                                                     \
    } while (0)

std::vector<float> readBin(const std::string& path, std::size_t expectElems) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
        std::fprintf(stderr, "cannot open %s\n", path.c_str());
        std::exit(2);
    }
    std::vector<float> v(expectElems);
    const std::size_t n = std::fread(v.data(), sizeof(float), expectElems, f);
    std::fclose(f);
    if (n != expectElems) {
        std::fprintf(stderr, "%s: read %zu of %zu floats\n", path.c_str(), n, expectElems);
        std::exit(2);
    }
    return v;
}

// normed[t,c] = ( H[t,c] * rsqrt(mean_group(H^2) + eps) ) * wbaked[c]
// one block per (t, group); blockDim.x threads reduce the group_size chunk.
__global__ void groupedRmsNorm(const float* __restrict__ H,
                               const float* __restrict__ wbaked, float eps,
                               int hcd, int group_size, float* __restrict__ normed) {
    const int t = blockIdx.y;
    const int g = blockIdx.x;              // group index
    const int base = g * group_size;
    extern __shared__ float sm[];
    float acc = 0.0f;
    for (int j = threadIdx.x; j < group_size; j += blockDim.x) {
        const float v = H[t * hcd + base + j];
        acc += v * v;
    }
    sm[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sm[threadIdx.x] += sm[threadIdx.x + s];
        __syncthreads();
    }
    const float inv = rsqrtf(sm[0] / group_size + eps);
    for (int j = threadIdx.x; j < group_size; j += blockDim.x) {
        const int c = base + j;
        normed[t * hcd + c] = H[t * hcd + c] * inv * wbaked[c];
    }
}

// y[t,o] = sum_i x[t,i] * W[o,i]   (nn.Linear: W is [out,in])
__global__ void naiveGemm(const float* __restrict__ x, const float* __restrict__ W,
                          int T, int IN, int OUT, float* __restrict__ y) {
    const int o = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (o >= OUT || t >= T) return;
    double acc = 0.0;
    const float* xr = x + t * IN;
    const float* wr = W + static_cast<std::size_t>(o) * IN;
    for (int i = 0; i < IN; ++i) acc += static_cast<double>(xr[i]) * wr[i];
    y[t * OUT + o] = static_cast<float>(acc);
}

__global__ void siluScale(float* __restrict__ x, int n, float scale) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float z = x[i] * scale;
    x[i] = z / (1.0f + expf(-z));   // silu(z) = z*sigmoid(z)
}

__global__ void sigmoidInPlace(float* __restrict__ x, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] = 1.0f / (1.0f + expf(-x[i]));
}

// mixed[t,j] = (1/hc) * sum_g w2[t, g*d + j] * normed[t, g*d + j]
__global__ void weightedMeanStreams(const float* __restrict__ w2,
                                    const float* __restrict__ normed, int T, int hc,
                                    int d, float* __restrict__ mixed) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (j >= d || t >= T) return;
    const int hcd = hc * d;
    float acc = 0.0f;
    for (int g = 0; g < hc; ++g) {
        const int c = g * d + j;
        acc += w2[t * hcd + c] * normed[t * hcd + c];
    }
    mixed[t * d + j] = acc / hc;
}

// inj[t,k] = 2 * sigmoid(raw[t,k] * scale)
__global__ void injSigmoid(float* __restrict__ inj, int n, float scale) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    inj[i] = 2.0f / (1.0f + expf(-inj[i] * scale));
}

void compare(const char* tag, const std::vector<float>& got,
             const std::vector<float>& ref) {
    double maxAbs = 0.0, maxRel = 0.0, sumSq = 0.0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        const double a = std::fabs(static_cast<double>(got[i]) - ref[i]);
        const double denom = std::fabs(static_cast<double>(ref[i])) + 1e-6;
        maxAbs = a > maxAbs ? a : maxAbs;
        const double r = a / denom;
        maxRel = r > maxRel ? r : maxRel;
        sumSq += a * a;
    }
    const double rmse = std::sqrt(sumSq / ref.size());
    std::printf("  %-7s maxAbs=%.3e  maxRel=%.3e  rmse=%.3e  (n=%zu)\n", tag, maxAbs,
                maxRel, rmse, ref.size());
}

} // namespace

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : "/opt/mimirmind/q4e_hc_dump";
    // Parse manifest.
    int T = 0, d = 0, hc = 0, lowrank = 0;
    double eps = 1e-6;
    {
        FILE* f = std::fopen((dir + "/manifest.txt").c_str(), "r");
        if (!f) { std::fprintf(stderr, "no manifest in %s\n", dir.c_str()); return 2; }
        char key[64]; double val;
        while (std::fscanf(f, "%63s %lf", key, &val) == 2) {
            const std::string k(key);
            if (k == "T") T = int(val); else if (k == "d") d = int(val);
            else if (k == "hc") hc = int(val); else if (k == "lowrank") lowrank = int(val);
            else if (k == "eps") eps = val;
        }
        std::fclose(f);
    }
    const int hcd = hc * d;
    std::printf("q4e_hc_parity: T=%d d=%d hc=%d lowrank=%d eps=%.3g\n", T, d, hc, lowrank, eps);

    auto H     = readBin(dir + "/H.bin",     std::size_t(T) * hcd);
    auto wnorm = readBin(dir + "/wnorm.bin", std::size_t(hcd));
    auto wdown = readBin(dir + "/wdown.bin", std::size_t(lowrank) * hcd);
    auto wup   = readBin(dir + "/wup.bin",   std::size_t(hcd) * lowrank);
    auto wbi   = readBin(dir + "/wbi.bin",   std::size_t(hc) * hcd);
    auto mixedRef = readBin(dir + "/mixed.bin", std::size_t(T) * d);
    auto injRef   = readBin(dir + "/inj.bin",   std::size_t(T) * hc);

    float *dH, *dWnorm, *dWdown, *dWup, *dWbi;
    float *dNormed, *dW1, *dW2, *dMixed, *dInjRaw;
    CK(cudaMalloc(&dH, sizeof(float) * H.size()));
    CK(cudaMalloc(&dWnorm, sizeof(float) * wnorm.size()));
    CK(cudaMalloc(&dWdown, sizeof(float) * wdown.size()));
    CK(cudaMalloc(&dWup, sizeof(float) * wup.size()));
    CK(cudaMalloc(&dWbi, sizeof(float) * wbi.size()));
    CK(cudaMalloc(&dNormed, sizeof(float) * std::size_t(T) * hcd));
    CK(cudaMalloc(&dW1, sizeof(float) * std::size_t(T) * lowrank));
    CK(cudaMalloc(&dW2, sizeof(float) * std::size_t(T) * hcd));
    CK(cudaMalloc(&dMixed, sizeof(float) * std::size_t(T) * d));
    CK(cudaMalloc(&dInjRaw, sizeof(float) * std::size_t(T) * hc));
    CK(cudaMemcpy(dH, H.data(), sizeof(float) * H.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dWnorm, wnorm.data(), sizeof(float) * wnorm.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dWdown, wdown.data(), sizeof(float) * wdown.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dWup, wup.data(), sizeof(float) * wup.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dWbi, wbi.data(), sizeof(float) * wbi.size(), cudaMemcpyHostToDevice));

    const int TH = 256;
    // 1. grouped RMSNorm
    {
        dim3 grid(hc, T);
        groupedRmsNorm<<<grid, TH, TH * sizeof(float)>>>(dH, dWnorm, float(eps), hcd, d, dNormed);
    }
    // 2. down GEMM -> [T,lowrank]; silu(x/hc)
    {
        dim3 grid((lowrank + TH - 1) / TH, T);
        naiveGemm<<<grid, TH>>>(dNormed, dWdown, T, hcd, lowrank, dW1);
    }
    siluScale<<<(T * lowrank + TH - 1) / TH, TH>>>(dW1, T * lowrank, 1.0f / hc);
    // 3. up GEMM -> [T,hcd]; sigmoid
    {
        dim3 grid((hcd + TH - 1) / TH, T);
        naiveGemm<<<grid, TH>>>(dW1, dWup, T, lowrank, hcd, dW2);
    }
    sigmoidInPlace<<<(T * hcd + TH - 1) / TH, TH>>>(dW2, T * hcd);
    // 4. weighted mean over streams -> mixed [T,d]
    {
        dim3 grid((d + TH - 1) / TH, T);
        weightedMeanStreams<<<grid, TH>>>(dW2, dNormed, T, hc, d, dMixed);
    }
    // 5. block inject GEMM -> [T,hc]; 2*sigmoid(x/hc)
    {
        dim3 grid((hc + TH - 1) / TH, T);
        naiveGemm<<<grid, TH>>>(dNormed, dWbi, T, hcd, hc, dInjRaw);
    }
    injSigmoid<<<(T * hc + TH - 1) / TH, TH>>>(dInjRaw, T * hc, 1.0f / hc);
    CK(cudaDeviceSynchronize());

    std::vector<float> mixed(std::size_t(T) * d), inj(std::size_t(T) * hc);
    CK(cudaMemcpy(mixed.data(), dMixed, sizeof(float) * mixed.size(), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(inj.data(), dInjRaw, sizeof(float) * inj.size(), cudaMemcpyDeviceToHost));

    std::printf("parity vs numpy reference:\n");
    compare("mixed", mixed, mixedRef);
    compare("inj", inj, injRef);

    // Verdict: f32 naive GEMM vs numpy f64-accumulated — expect ~1e-5 or better.
    double maxAbs = 0.0;
    for (std::size_t i = 0; i < mixedRef.size(); ++i)
        maxAbs = std::max(maxAbs, std::fabs(double(mixed[i]) - mixedRef[i]));
    for (std::size_t i = 0; i < injRef.size(); ++i)
        maxAbs = std::max(maxAbs, std::fabs(double(inj[i]) - injRef[i]));
    const bool pass = maxAbs < 1e-3;
    std::printf("%s (max abs err %.3e, tol 1e-3)\n", pass ? "PARITY PASS" : "PARITY FAIL", maxAbs);
    return pass ? 0 : 1;
}
