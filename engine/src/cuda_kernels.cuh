// M1 kernels for the CUDA backend: attention over the paged KV cache (a flash-style
// prefill kernel on tensor cores, a split-context decode kernel) and weight-only int8
// matrix-vector products. Included by cuda_backend.cu only.
//
// Cache layout (both K and V, per layer): [block][kv_head][slot][head_dim], fp16.
#pragma once

#include <cuda_fp16.h>
#include <mma.h>

#include <cstdint>

namespace relay {
namespace kernels {

struct TokenMeta {
  const int* pos;        // [T]
  const int* seq;        // [T] chunk index of each token
  const int* table_off;  // [S] offset of each chunk's block table in tables
  const int* tables;     // concatenated block tables
};

__device__ __forceinline__ long long kv_offset(const int* table, int pos, int kv_heads, int kvh, int block_size, int D) {
  return (static_cast<long long>(table[pos / block_size]) * kv_heads + kvh) * block_size * D +
         static_cast<long long>(pos % block_size) * D;
}

// ---- prefill: flash attention on tensor cores --------------------------------
//
// One block per (tile of up to 64 consecutive query tokens of one chunk, query head);
// four warps, 16 query rows each. The block walks the chunk's keys 32 at a time (causal:
// up to its last row's position), gathering them from the paged cache into shared memory.
// Each warp computes S = Q K^T for its rows with WMMA (fp16 in, fp32 out), keeps a running
// max m and sum l per row (online softmax), writes P = exp(S - m) as fp16, rescales its
// output rows and adds P V, again with WMMA. Nothing of size context x context exists.

constexpr int kFlashRows = 64;  // query rows per block
constexpr int kFlashKeys = 32;  // keys per step
constexpr int kFlashWarps = 4;

template <int D>
constexpr int flash_smem_bytes() {
  return kFlashRows * D * 2                     // Q (fp16)
         + 2 * kFlashKeys * D * 2               // K, V tiles (fp16)
         + kFlashWarps * 16 * kFlashKeys * 4    // S per warp (fp32)
         + kFlashWarps * 16 * kFlashKeys * 2    // P per warp (fp16)
         + kFlashWarps * 16 * D * 4             // O per warp (fp32)
         + 3 * kFlashRows * 4;                  // m, l, alpha per row
}

template <int D>
__global__ void __launch_bounds__(kFlashWarps * 32)
    flash_prefill_kernel(const float* qkv, TokenMeta m, const int2* tiles, const __half* kc, const __half* vc,
                         __half* out, int heads, int kv_heads, int block_size, float scale) {
  using namespace nvcuda;
  constexpr int BQ = kFlashRows, BK = kFlashKeys, W = kFlashWarps;
  extern __shared__ __align__(32) unsigned char smem[];
  __half* sQ = reinterpret_cast<__half*>(smem);
  __half* sK = sQ + BQ * D;
  __half* sV = sK + BK * D;
  float* sS = reinterpret_cast<float*>(sV + BK * D);
  __half* sP = reinterpret_cast<__half*>(sS + W * 16 * BK);
  float* sO = reinterpret_cast<float*>(sP + W * 16 * BK);
  float* sM = sO + W * 16 * D;
  float* sL = sM + BQ;
  float* sA = sL + BQ;

  const int h = blockIdx.y, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int t0 = tiles[blockIdx.x].x, nrows = tiles[blockIdx.x].y;
  const int kvh = h / (heads / kv_heads);
  const int Q = heads * D, QKV = Q + 2 * kv_heads * D;
  const int p0 = m.pos[t0];
  const int* table = m.tables + m.table_off[m.seq[t0]];
  const int nkeys = p0 + nrows;  // keys 0 .. last row's position

  for (int i = threadIdx.x; i < BQ * D; i += blockDim.x) {
    int r = i / D, d = i % D;
    sQ[i] = __float2half(r < nrows ? qkv[static_cast<long long>(t0 + r) * QKV + h * D + d] : 0.0f);
  }
  for (int i = threadIdx.x; i < W * 16 * D; i += blockDim.x) sO[i] = 0.0f;
  for (int i = threadIdx.x; i < BQ; i += blockDim.x) {
    sM[i] = -INFINITY;
    sL[i] = 0.0f;
  }
  __syncthreads();

  float* wS = sS + warp * 16 * BK;
  __half* wP = sP + warp * 16 * BK;
  float* wO = sO + warp * 16 * D;
  const __half* wQ = sQ + warp * 16 * D;

  for (int k0 = 0; k0 < nkeys; k0 += BK) {
    // Gather this step's keys and values from their cache blocks, 16 bytes at a time.
    for (int i = threadIdx.x; i < BK * D / 8; i += blockDim.x) {
      int r = i / (D / 8), c = i % (D / 8), kp = k0 + r;
      uint4 kv = make_uint4(0, 0, 0, 0), vv = make_uint4(0, 0, 0, 0);
      if (kp < nkeys) {
        long long off = kv_offset(table, kp, kv_heads, kvh, block_size, D) + c * 8;
        kv = *reinterpret_cast<const uint4*>(kc + off);
        vv = *reinterpret_cast<const uint4*>(vc + off);
      }
      reinterpret_cast<uint4*>(sK)[i] = kv;
      reinterpret_cast<uint4*>(sV)[i] = vv;
    }
    __syncthreads();

    // S = Q K^T for this warp's 16 rows and the step's 32 keys.
    for (int n = 0; n < BK / 16; ++n) {
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
      wmma::fill_fragment(acc, 0.0f);
      for (int k = 0; k < D / 16; ++k) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b;
        wmma::load_matrix_sync(a, wQ + k * 16, D);
        wmma::load_matrix_sync(b, sK + n * 16 * D + k * 16, D);  // B(k, n) = K[n][k]
        wmma::mma_sync(acc, a, b, acc);
      }
      wmma::store_matrix_sync(wS + n * 16, acc, BK, wmma::mem_row_major);
    }
    __syncwarp();

    // Online softmax: two lanes per row, 16 keys each.
    {
      const int r = lane >> 1, part = lane & 1, row = warp * 16 + r;
      const int qpos = p0 + row;
      const bool live = row < nrows;
      float* srow = wS + r * BK;
      __half* prow = wP + r * BK;
      float mx = -INFINITY;
      for (int j = part * 16; j < part * 16 + 16; ++j) {
        float s = srow[j] * scale;
        if (!live || k0 + j > qpos) s = -INFINITY;  // causal, and padding rows
        srow[j] = s;
        mx = fmaxf(mx, s);
      }
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
      const float m_old = sM[row];
      const float m_new = fmaxf(m_old, mx);
      float sum = 0.0f;
      for (int j = part * 16; j < part * 16 + 16; ++j) {
        float p = m_new == -INFINITY ? 0.0f : __expf(srow[j] - m_new);
        prow[j] = __float2half(p);
        sum += p;
      }
      sum += __shfl_xor_sync(0xffffffffu, sum, 1);
      const float alpha = m_old == -INFINITY ? 0.0f : __expf(m_old - m_new);
      __syncwarp();
      if (part == 0) {
        sM[row] = m_new;
        sL[row] = sL[row] * alpha + sum;
        sA[row] = alpha;
      }
    }
    __syncwarp();

    // O = alpha * O + P V.
    for (int i = lane; i < 16 * D; i += 32) wO[i] *= sA[warp * 16 + i / D];
    __syncwarp();
    for (int n = 0; n < D / 16; ++n) {
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> o;
      wmma::load_matrix_sync(o, wO + n * 16, D, wmma::mem_row_major);
      for (int k = 0; k < BK / 16; ++k) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> pa;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> vb;
        wmma::load_matrix_sync(pa, wP + k * 16, BK);
        wmma::load_matrix_sync(vb, sV + k * 16 * D + n * 16, D);  // B(k, n) = V[k][n]
        wmma::mma_sync(o, pa, vb, o);
      }
      wmma::store_matrix_sync(wO + n * 16, o, D, wmma::mem_row_major);
    }
    __syncthreads();  // the next step overwrites K and V
  }

  for (int i = threadIdx.x; i < BQ * D; i += blockDim.x) {
    int r = i / D, d = i % D;
    if (r < nrows) out[static_cast<long long>(t0 + r) * Q + h * D + d] = __float2half(sO[i] / sL[r]);
  }
}

// ---- decode: one new token per sequence, split over the context ---------------
//
// One block per (decode token, KV head, slice of the context): all G query heads that
// share the KV head are handled together, so each K and V row is read once for all of
// them. Each thread scores one key of a 128-key step; the block keeps a running max and
// sum per head; the step's V rows go to shared memory and each thread accumulates a few
// (head, dim) outputs. Every slice writes its partial (m, l, O); a second kernel merges
// the slices with log-sum-exp. Long contexts thus spread over many blocks (flash-decoding).

constexpr int kDecodeThreads = 128;
constexpr int kDecodeSlice = 512;  // keys per block

template <int G>
__device__ __forceinline__ void block_reduce_max(float (&v)[G], float* red) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
  for (int g = 0; g < G; ++g) {
    float x = v[g];
    for (int o = 16; o > 0; o >>= 1) x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, o));
    if (lane == 0) red[g * 4 + warp] = x;
  }
  __syncthreads();
#pragma unroll
  for (int g = 0; g < G; ++g)
    v[g] = fmaxf(fmaxf(red[g * 4], red[g * 4 + 1]), fmaxf(red[g * 4 + 2], red[g * 4 + 3]));
  __syncthreads();
}

template <int G>
__device__ __forceinline__ void block_reduce_sum(float (&v)[G], float* red) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
  for (int g = 0; g < G; ++g) {
    float x = v[g];
    for (int o = 16; o > 0; o >>= 1) x += __shfl_xor_sync(0xffffffffu, x, o);
    if (lane == 0) red[g * 4 + warp] = x;
  }
  __syncthreads();
#pragma unroll
  for (int g = 0; g < G; ++g) v[g] = (red[g * 4] + red[g * 4 + 1]) + (red[g * 4 + 2] + red[g * 4 + 3]);
  __syncthreads();
}

template <int D, int G>
__global__ void __launch_bounds__(kDecodeThreads)
    decode_attention_kernel(const float* qkv, TokenMeta m, const int* decode_tokens, const __half* kc,
                            const __half* vc, float* part_o, float* part_ml, int heads, int kv_heads, int block_size,
                            float scale, int max_parts) {
  constexpr int NT = kDecodeThreads;
  constexpr int PER = (G * D + NT - 1) / NT;  // outputs per thread
  __shared__ float sq[G][D];
  __shared__ float sp[G][NT];
  __shared__ __align__(16) __half sv[NT][D];
  __shared__ float red[G * 4];
  __shared__ float salpha[G];  // this step's rescale per head (an array indexed at run time)

  const int di = blockIdx.x, kvh = blockIdx.y, part = blockIdx.z;
  const int t = decode_tokens[di];
  const int Q = heads * D, QKV = Q + 2 * kv_heads * D;
  const int ctx = m.pos[t] + 1;
  const int begin = part * kDecodeSlice;
  const long long pbase = (static_cast<long long>(di) * heads + kvh * G) * max_parts + part;  // head g adds g * max_parts
  if (begin >= ctx) {  // nothing in this slice: an empty partial
    if (threadIdx.x < G) {
      part_ml[(pbase + threadIdx.x * max_parts) * 2] = -INFINITY;
      part_ml[(pbase + threadIdx.x * max_parts) * 2 + 1] = 0.0f;
    }
    return;
  }
  const int end = min(ctx, begin + kDecodeSlice);
  const int* table = m.tables + m.table_off[m.seq[t]];
  for (int i = threadIdx.x; i < G * D; i += NT)
    sq[i / D][i % D] = qkv[static_cast<long long>(t) * QKV + (kvh * G + i / D) * D + i % D] * scale;
  __syncthreads();

  float m_run[G], l_run[G], o[PER];
#pragma unroll
  for (int g = 0; g < G; ++g) {
    m_run[g] = -INFINITY;
    l_run[g] = 0.0f;
  }
#pragma unroll
  for (int i = 0; i < PER; ++i) o[i] = 0.0f;

  for (int k0 = begin; k0 < end; k0 += NT) {
    const int j = k0 + threadIdx.x;
    const bool live = j < end;
    float s[G];
#pragma unroll
    for (int g = 0; g < G; ++g) s[g] = 0.0f;
    if (live) {
      const __half* krow = kc + kv_offset(table, j, kv_heads, kvh, block_size, D);
#pragma unroll
      for (int c = 0; c < D / 8; ++c) {
        uint4 raw = *reinterpret_cast<const uint4*>(krow + c * 8);
        const __half2* k2 = reinterpret_cast<const __half2*>(&raw);
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          float2 kf = __half22float2(k2[e]);
#pragma unroll
          for (int g = 0; g < G; ++g) s[g] += sq[g][c * 8 + 2 * e] * kf.x + sq[g][c * 8 + 2 * e + 1] * kf.y;
        }
      }
    } else {
#pragma unroll
      for (int g = 0; g < G; ++g) s[g] = -INFINITY;
    }
    float mx[G];
#pragma unroll
    for (int g = 0; g < G; ++g) mx[g] = s[g];
    block_reduce_max<G>(mx, red);
    float alpha[G], sum[G];
#pragma unroll
    for (int g = 0; g < G; ++g) {
      const float m_new = fmaxf(m_run[g], mx[g]);  // finite: every step has a live key
      alpha[g] = m_run[g] == -INFINITY ? 0.0f : __expf(m_run[g] - m_new);
      const float p = live ? __expf(s[g] - m_new) : 0.0f;
      sp[g][threadIdx.x] = p;
      sum[g] = p;
      m_run[g] = m_new;
    }
    if (threadIdx.x == 0) {
#pragma unroll
      for (int g = 0; g < G; ++g) salpha[g] = alpha[g];
    }
    // The step's V rows (zero past the end).
    for (int i = threadIdx.x; i < NT * D / 8; i += NT) {
      int r = i / (D / 8), c = i % (D / 8), kp = k0 + r;
      uint4 vv = make_uint4(0, 0, 0, 0);
      if (kp < end) vv = *reinterpret_cast<const uint4*>(vc + kv_offset(table, kp, kv_heads, kvh, block_size, D) + c * 8);
      reinterpret_cast<uint4*>(&sv[0][0])[i] = vv;
    }
    block_reduce_sum<G>(sum, red);  // also the barrier for sp and sv
#pragma unroll
    for (int g = 0; g < G; ++g) l_run[g] = l_run[g] * alpha[g] + sum[g];
    const int n = min(NT, end - k0);
#pragma unroll
    for (int i = 0; i < PER; ++i) {
      const int idx = threadIdx.x + i * NT;
      if (idx >= G * D) break;
      const int g = idx / D, d = idx % D;
      float acc = 0.0f;
      for (int r = 0; r < n; ++r) acc += sp[g][r] * __half2float(sv[r][d]);
      o[i] = o[i] * salpha[g] + acc;
    }
    __syncthreads();  // sp and sv are rewritten next step
  }

#pragma unroll
  for (int i = 0; i < PER; ++i) {
    const int idx = threadIdx.x + i * NT;
    if (idx >= G * D) break;
    const int g = idx / D, d = idx % D;
    part_o[(pbase + static_cast<long long>(g) * max_parts) * D + d] = o[i];
  }
  if (threadIdx.x < G) {
    part_ml[(pbase + threadIdx.x * max_parts) * 2] = m_run[threadIdx.x];
    part_ml[(pbase + threadIdx.x * max_parts) * 2 + 1] = l_run[threadIdx.x];
  }
}

// Merges the slices of each (decode token, head): out = sum_p e^(m_p - M) O_p / sum_p e^(m_p - M) l_p.
__global__ void decode_combine_kernel(const float* part_o, const float* part_ml, const int* decode_tokens, __half* out,
                                      int heads, int D, int max_parts) {
  const int di = blockIdx.x, h = blockIdx.y;
  const long long base = (static_cast<long long>(di) * heads + h) * max_parts;
  float M = -INFINITY;
  for (int p = 0; p < max_parts; ++p) M = fmaxf(M, part_ml[(base + p) * 2]);
  float L = 0.0f;
  for (int p = 0; p < max_parts; ++p) {
    float mp = part_ml[(base + p) * 2];
    if (mp != -INFINITY) L += __expf(mp - M) * part_ml[(base + p) * 2 + 1];
  }
  const int t = decode_tokens[di];
  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    float acc = 0.0f;
    for (int p = 0; p < max_parts; ++p) {
      float mp = part_ml[(base + p) * 2];
      if (mp != -INFINITY) acc += __expf(mp - M) * part_o[(base + p) * D + d];
    }
    out[static_cast<long long>(t) * heads * D + h * D + d] = __float2half(acc / L);
  }
}

// ---- weight-only int8 ---------------------------------------------------------
//
// Weights as int8 with one scale per output row (symmetric, scale = max|w| / 127).
// Decode batches (a few tokens) are matrix-vector products bound by weight bandwidth:
// one warp per output row streams the row 16 bytes per lane per step and multiplies it
// with every token's activations. Larger batches dequantize into fp16 and use cuBLAS.

constexpr int kInt8MaxTokens = 8;
constexpr int kInt8Warps = 8;     // output rows per block
constexpr int kInt8Slice = 512;   // activations staged in shared memory per step (32 lanes x 16)

// Each block computes kInt8Warps output rows for every token. Per 512-wide slice of K, the
// tokens' activations are staged once in shared memory for all the block's warps; each lane
// then streams 16 int8 weights of its row and multiplies them with every token's slice.
template <int TMAX>
__global__ void __launch_bounds__(kInt8Warps * 32)
    gemv_int8_kernel(const int8_t* W, const float* row_scale, const __half* x, float* y, int T, int K, int N,
                     float beta) {
  __shared__ __align__(16) __half xs[TMAX][kInt8Slice];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int n = blockIdx.x * kInt8Warps + warp;
  const int8_t* w = W + static_cast<long long>(min(n, N - 1)) * K;
  float acc[TMAX];
#pragma unroll
  for (int t = 0; t < TMAX; ++t) acc[t] = 0.0f;
  for (int k0 = 0; k0 < K; k0 += kInt8Slice) {
    for (int i = threadIdx.x; i < T * (kInt8Slice / 8); i += blockDim.x) {
      const int t = i / (kInt8Slice / 8), c = (i % (kInt8Slice / 8)) * 8, k = k0 + c;
      uint4 v = make_uint4(0, 0, 0, 0);
      if (k < K) v = *reinterpret_cast<const uint4*>(x + static_cast<long long>(t) * K + k);
      *reinterpret_cast<uint4*>(&xs[t][c]) = v;
    }
    __syncthreads();
    const int k = k0 + lane * 16;
    if (n < N && k < K) {
      int4 wv = *reinterpret_cast<const int4*>(w + k);
      const int8_t* wb = reinterpret_cast<const int8_t*>(&wv);
#pragma unroll
      for (int t = 0; t < TMAX; ++t) {
        if (t >= T) break;
        uint4 x0 = *reinterpret_cast<const uint4*>(&xs[t][lane * 16]);
        uint4 x1 = *reinterpret_cast<const uint4*>(&xs[t][lane * 16 + 8]);
        const __half2* a = reinterpret_cast<const __half2*>(&x0);
        const __half2* b = reinterpret_cast<const __half2*>(&x1);
        float s = 0.0f;
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          float2 fa = __half22float2(a[e]), fb = __half22float2(b[e]);
          s += wb[2 * e] * fa.x + wb[2 * e + 1] * fa.y + wb[8 + 2 * e] * fb.x + wb[8 + 2 * e + 1] * fb.y;
        }
        acc[t] += s;
      }
    }
    __syncthreads();  // xs is refilled next slice
  }
  if (n >= N) return;
#pragma unroll
  for (int t = 0; t < TMAX; ++t) {
    if (t >= T) break;
    float v = acc[t];
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if (lane == 0) {
      float* dst = y + static_cast<long long>(t) * N + n;
      *dst = v * row_scale[n] + (beta != 0.0f ? beta * *dst : 0.0f);
    }
  }
}

__global__ void dequant_int8_kernel(const int8_t* W, const float* row_scale, __half* out, int K, long long total) {
  for (long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; i < total;
       i += static_cast<long long>(gridDim.x) * blockDim.x)
    out[i] = __float2half(static_cast<float>(W[i]) * row_scale[i / K]);
}

}  // namespace kernels
}  // namespace relay
