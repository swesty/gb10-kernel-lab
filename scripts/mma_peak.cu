// Tensor-core peak on GB10 (sm_121a) with warp-level mma.sync, register operands only (no memory traffic).
// Each warp runs ILP independent accumulator chains for ITERS iterations.
// Build: nvcc -O3 -arch=sm_121a mma_peak.cu -o mma_peak     Run: ./mma_peak
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

constexpr int ILP = 8;
constexpr int ITERS = 4096;

__global__ void bf16_k(float* out, uint32_t seed) {
  uint32_t a[4], b[2];
  for (int i = 0; i < 4; i++) a[i] = seed * (i + 1) ^ threadIdx.x;
  for (int i = 0; i < 2; i++) b[i] = seed * (i + 7) ^ threadIdx.x;
  float c[ILP][4] = {};
  for (int it = 0; it < ITERS; it++) {
#pragma unroll
    for (int j = 0; j < ILP; j++)
      asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                   : "+f"(c[j][0]), "+f"(c[j][1]), "+f"(c[j][2]), "+f"(c[j][3])
                   : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
  }
  float s = 0; for (int j = 0; j < ILP; j++) s += c[j][0] + c[j][1] + c[j][2] + c[j][3];
  if (s == 12345.f) out[threadIdx.x] = s;
}

__global__ void fp8_k(float* out, uint32_t seed) {
  uint32_t a[4], b[2];
  for (int i = 0; i < 4; i++) a[i] = (seed * (i + 1) ^ threadIdx.x) & 0x3f3f3f3f;
  for (int i = 0; i < 2; i++) b[i] = (seed * (i + 7) ^ threadIdx.x) & 0x3f3f3f3f;
  float c[ILP][4] = {};
  for (int it = 0; it < ITERS; it++) {
#pragma unroll
    for (int j = 0; j < ILP; j++)
      asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                   : "+f"(c[j][0]), "+f"(c[j][1]), "+f"(c[j][2]), "+f"(c[j][3])
                   : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
  }
  float s = 0; for (int j = 0; j < ILP; j++) s += c[j][0] + c[j][1] + c[j][2] + c[j][3];
  if (s == 12345.f) out[threadIdx.x] = s;
}

__global__ void nvfp4_k(float* out, uint32_t seed) {
  uint32_t a[4], b[2];
  for (int i = 0; i < 4; i++) a[i] = (seed * (i + 1) ^ threadIdx.x) & 0x33333333;
  for (int i = 0; i < 2; i++) b[i] = (seed * (i + 7) ^ threadIdx.x) & 0x33333333;
  uint32_t sfa = 0x38383838u, sfb = 0x38383838u;  // ue4m3 scale 1.0 in every byte
  uint16_t bid = 0, tid = 0;
  float c[ILP][4] = {};
  for (int it = 0; it < ITERS; it++) {
#pragma unroll
    for (int j = 0; j < ILP; j++)
      asm volatile(
          "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};\n"
          : "+f"(c[j][0]), "+f"(c[j][1]), "+f"(c[j][2]), "+f"(c[j][3])
          : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
            "r"(sfa), "h"(bid), "h"(tid), "r"(sfb), "h"(bid), "h"(tid));
  }
  float s = 0; for (int j = 0; j < ILP; j++) s += c[j][0] + c[j][1] + c[j][2] + c[j][3];
  if (s == 12345.f) out[threadIdx.x] = s;
}

template <typename K>
void run(const char* name, K kern, double flop_per_mma) {
  float* out; cudaMalloc(&out, 4096 * sizeof(float));
  int sms; cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  for (int warps : {4, 8, 16}) {
    int blocks = sms * 4, threads = warps * 32 / 4 * 4 / 4 * 4;  // warps per block = warps/4... keep simple below
    threads = 32 * warps / 4;  // 4 blocks per SM -> `warps` warps per SM in total
    kern<<<blocks, threads>>>(out, 1u); cudaDeviceSynchronize();
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    for (int r = 0; r < 5; r++) kern<<<blocks, threads>>>(out, 1u + r);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms; cudaEventElapsedTime(&ms, s, e);
    double mmas = 5.0 * blocks * (threads / 32) * (double)ITERS * ILP;
    printf("%-34s warps/SM %2d : %7.1f TFLOPS\n", name, warps, mmas * flop_per_mma / (ms * 1e-3) / 1e12);
    cudaError_t err = cudaGetLastError(); if (err) printf("  error: %s\n", cudaGetErrorString(err));
  }
  cudaFree(out);
}

int main() {
  run("BF16 m16n8k16, f32 acc", bf16_k, 2.0 * 16 * 8 * 16);
  run("FP8 e4m3 m16n8k32, f32 acc", fp8_k, 2.0 * 16 * 8 * 32);
  run("NVFP4 m16n8k64 block-scaled, f32", nvfp4_k, 2.0 * 16 * 8 * 64);
  return 0;
}
