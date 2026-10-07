// The CUDA backend: fp16 weights (or int8, optionally) and KV cache, fp32 residual stream.
//
// Matrix multiplies go to cuBLAS (fp16 inputs, fp32 accumulation and output; the
// residual adds are folded into the GEMM with beta = 1). Everything else is a small
// kernel: embedding lookup, RMSNorm, bias + RoPE + KV-cache write, attention over the
// paged cache, SiLU-and-multiply.
//
// Attention (M1, cuda_kernels.cuh): chunks of several tokens (prefill) go to a
// flash-style kernel on tensor cores; single tokens (decode) to a kernel that splits the
// context across blocks and merges the slices. The M0 kernel (one block per token and
// head, every score in shared memory) remains as the reference the tests compare
// against, and for head sizes the M1 kernels do not cover.
//
// Streams: the forward pass runs on its own stream; KV block reads and writes (the
// transfer engine's traffic) run on a second one, so a transfer never waits for the
// forward pass to finish. An event recorded after each layer's KV write tells the
// transfer engine when that layer can be read (wait_kv_written).
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>

#include <cstdlib>

#include "cuda_kernels.cuh"
#include "relay/backend.h"
#include "relay/safetensors.h"

namespace relay {

CudaOptions CudaOptions::from_env() {
  CudaOptions o;
  if (const char* a = std::getenv("RELAY_CUDA_ATTENTION")) o.reference_attention = std::string(a) == "m0";
  if (const char* q = std::getenv("RELAY_CUDA_INT8")) o.int8_weights = std::string(q) == "1";
  return o;
}

namespace {

using kernels::TokenMeta;

#define CUDA_CHECK(x)                                                                                    \
  do {                                                                                                   \
    cudaError_t _e = (x);                                                                                \
    if (_e != cudaSuccess)                                                                               \
      throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(_e) + " at " #x);              \
  } while (0)

#define CUBLAS_CHECK(x)                                                                                  \
  do {                                                                                                   \
    cublasStatus_t _s = (x);                                                                             \
    if (_s != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cuBLAS error " + std::to_string(_s) + " at " #x); \
  } while (0)

template <typename T>
struct DeviceArray {
  T* p = nullptr;
  std::size_t n = 0;
  DeviceArray() = default;
  explicit DeviceArray(std::size_t count) : n(count) {
    if (count) CUDA_CHECK(cudaMalloc(&p, count * sizeof(T)));
  }
  ~DeviceArray() {
    if (p) cudaFree(p);
  }
  DeviceArray(const DeviceArray&) = delete;
  DeviceArray& operator=(const DeviceArray&) = delete;
  DeviceArray(DeviceArray&& o) noexcept : p(o.p), n(o.n) {
    o.p = nullptr;
    o.n = 0;
  }
  DeviceArray& operator=(DeviceArray&& o) noexcept {
    if (this != &o) {
      if (p) cudaFree(p);
      p = o.p;
      n = o.n;
      o.p = nullptr;
      o.n = 0;
    }
    return *this;
  }
  void upload(const T* host, std::size_t count) {
    if (count > n) throw std::logic_error("upload larger than the device array");
    CUDA_CHECK(cudaMemcpy(p, host, count * sizeof(T), cudaMemcpyHostToDevice));
  }
};

DeviceArray<__half> to_device_f16(const std::vector<float>& v) {
  std::vector<std::uint16_t> h(v.size());
  for (std::size_t i = 0; i < v.size(); ++i) h[i] = f32_to_f16(v[i]);
  DeviceArray<__half> d(v.size());
  CUDA_CHECK(cudaMemcpy(d.p, h.data(), h.size() * 2, cudaMemcpyHostToDevice));
  return d;
}

DeviceArray<float> to_device_f32(const std::vector<float>& v) {
  DeviceArray<float> d(v.size());
  if (!v.empty()) d.upload(v.data(), v.size());
  return d;
}

// ---- kernels ---------------------------------------------------------------

// fp32 -> fp16, round to nearest even: the same bits as the host's f32_to_f16, so a
// weight updated on the device equals one uploaded from the host.
__global__ void f32_to_f16_kernel(const float* in, __half* out, std::size_t n) {
  for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<std::size_t>(gridDim.x) * blockDim.x)
    out[i] = __float2half_rn(in[i]);
}

__device__ float block_sum(float v, float* scratch) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
  int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if (lane == 0) scratch[warp] = v;
  __syncthreads();
  int nwarps = (blockDim.x + 31) >> 5;
  v = threadIdx.x < nwarps ? scratch[threadIdx.x] : 0.0f;
  if (warp == 0)
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
  if (threadIdx.x == 0) scratch[0] = v;
  __syncthreads();
  float r = scratch[0];
  __syncthreads();
  return r;
}

__device__ float block_max(float v, float* scratch) {
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
  int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if (lane == 0) scratch[warp] = v;
  __syncthreads();
  int nwarps = (blockDim.x + 31) >> 5;
  v = threadIdx.x < nwarps ? scratch[threadIdx.x] : -INFINITY;
  if (warp == 0)
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
  if (threadIdx.x == 0) scratch[0] = v;
  __syncthreads();
  float r = scratch[0];
  __syncthreads();
  return r;
}

__global__ void embed_kernel(const int* tokens, const __half* table, float* x, int H) {
  int t = blockIdx.x;
  const __half* row = table + static_cast<long long>(tokens[t]) * H;
  for (int i = threadIdx.x; i < H; i += blockDim.x) x[static_cast<long long>(t) * H + i] = __half2float(row[i]);
}

// out[r] = w * (x[rows[r]] * rsqrt(mean(x^2) + eps)), rows == nullptr meaning r itself.
__global__ void rmsnorm_kernel(const float* x, const int* rows, const float* w, __half* out, int H, float eps) {
  __shared__ float scratch[32];
  int r = blockIdx.x;
  const float* xr = x + static_cast<long long>(rows ? rows[r] : r) * H;
  float ss = 0;
  for (int i = threadIdx.x; i < H; i += blockDim.x) ss += xr[i] * xr[i];
  ss = block_sum(ss, scratch);
  float inv = rsqrtf(ss / H + eps);
  for (int i = threadIdx.x; i < H; i += blockDim.x)
    out[static_cast<long long>(r) * H + i] = __float2half(w[i] * (xr[i] * inv));
}

// Adds the q/k/v bias, applies RoPE to q (in place) and k, and writes k and v into
// the paged cache at the token's position.
__global__ void qkv_post_kernel(float* qkv, const float* bias, const float* inv_freq, TokenMeta m, __half* kc,
                                __half* vc, int heads, int kv_heads, int D, int block_size) {
  int t = blockIdx.x;
  const int Q = heads * D, KV = kv_heads * D, QKV = Q + 2 * KV, half = D / 2;
  float* row = qkv + static_cast<long long>(t) * QKV;
  if (bias)
    for (int i = threadIdx.x; i < QKV; i += blockDim.x) row[i] += bias[i];
  __syncthreads();
  int p = m.pos[t];
  int block = m.tables[m.table_off[m.seq[t]] + p / block_size], slot = p % block_size;
  // RoPE pairs (i, i + half) of every q and k head.
  for (int idx = threadIdx.x; idx < (heads + kv_heads) * half; idx += blockDim.x) {
    int h = idx / half, i = idx % half;
    float* v = row + h * D;  // q heads, then k heads right after them
    float angle = static_cast<float>(p) * inv_freq[i];
    float cs = cosf(angle), sn = sinf(angle);
    float a = v[i], b = v[i + half];
    v[i] = a * cs - b * sn;
    v[i + half] = b * cs + a * sn;
  }
  __syncthreads();
  for (int idx = threadIdx.x; idx < KV; idx += blockDim.x) {
    int h = idx / D, d = idx % D;
    long long off = (static_cast<long long>(block) * kv_heads + h) * block_size * D + static_cast<long long>(slot) * D + d;
    kc[off] = __float2half(row[Q + idx]);
    vc[off] = __float2half(row[Q + KV + idx]);
  }
}

// One thread block per (token, query head). Scores for every visible position go to
// shared memory, then softmax, then the weighted sum of V.
__global__ void attention_kernel(const float* qkv, TokenMeta m, const __half* kc, const __half* vc, __half* out,
                                 int heads, int kv_heads, int D, int block_size, float scale) {
  extern __shared__ float smem[];
  __shared__ float scratch[32];
  int t = blockIdx.x, h = blockIdx.y;
  const int Q = heads * D, QKV = Q + 2 * kv_heads * D, kvh = h / (heads / kv_heads);
  float* q = smem;           // [D]
  float* score = smem + D;   // [ctx]
  const float* qsrc = qkv + static_cast<long long>(t) * QKV + h * D;
  for (int d = threadIdx.x; d < D; d += blockDim.x) q[d] = qsrc[d];
  __syncthreads();
  const int ctx = m.pos[t] + 1;
  const int* table = m.tables + m.table_off[m.seq[t]];
  float mx = -INFINITY;
  for (int j = threadIdx.x; j < ctx; j += blockDim.x) {
    long long off = (static_cast<long long>(table[j / block_size]) * kv_heads + kvh) * block_size * D +
                    static_cast<long long>(j % block_size) * D;
    float s = 0;
    for (int d = 0; d < D; ++d) s += q[d] * __half2float(kc[off + d]);
    s *= scale;
    score[j] = s;
    mx = fmaxf(mx, s);
  }
  mx = block_max(mx, scratch);
  float sum = 0;
  for (int j = threadIdx.x; j < ctx; j += blockDim.x) {
    float e = __expf(score[j] - mx);
    score[j] = e;
    sum += e;
  }
  sum = block_sum(sum, scratch);  // also a barrier: every score is written
  float inv = 1.0f / sum;
  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    float acc = 0;
    for (int j = 0; j < ctx; ++j) {
      long long off = (static_cast<long long>(table[j / block_size]) * kv_heads + kvh) * block_size * D +
                      static_cast<long long>(j % block_size) * D;
      acc += score[j] * __half2float(vc[off + d]);
    }
    out[static_cast<long long>(t) * Q + h * D + d] = __float2half(acc * inv);
  }
}

__global__ void silu_mul_kernel(const float* gu, __half* act, int I) {
  int t = blockIdx.x;
  const float* g = gu + static_cast<long long>(t) * 2 * I;
  const float* u = g + I;
  for (int i = threadIdx.x; i < I; i += blockDim.x) {
    float x = g[i];
    act[static_cast<long long>(t) * I + i] = __float2half(x / (1.0f + __expf(-x)) * u[i]);
  }
}

// Copies blocks of one layer's K and V into (gather) or out of (scatter) one contiguous
// buffer laid out as [K of each block][V of each block], 16 bytes per thread step.
__global__ void kv_gather_kernel(const uint4* k, const uint4* v, const int* blocks, int n, long long block_vec,
                                 uint4* out) {
  int i = blockIdx.x, which = blockIdx.y;
  const uint4* src = (which ? v : k) + static_cast<long long>(blocks[i]) * block_vec;
  uint4* dst = out + (static_cast<long long>(which) * n + i) * block_vec;
  for (long long e = threadIdx.x; e < block_vec; e += blockDim.x) dst[e] = src[e];
}

__global__ void kv_scatter_kernel(uint4* k, uint4* v, const int* blocks, int n, long long block_vec, const uint4* in) {
  int i = blockIdx.x, which = blockIdx.y;
  uint4* dst = (which ? v : k) + static_cast<long long>(blocks[i]) * block_vec;
  const uint4* src = in + (static_cast<long long>(which) * n + i) * block_vec;
  for (long long e = threadIdx.x; e < block_vec; e += blockDim.x) dst[e] = src[e];
}

// ---- backend ----------------------------------------------------------------

// A weight matrix [N, K]: fp16, or int8 with one scale per row.
struct DeviceMatrix {
  int N = 0, K = 0;
  DeviceArray<__half> f16;
  DeviceArray<std::int8_t> i8;
  DeviceArray<float> scale;
};

DeviceMatrix to_device_matrix(const std::vector<float>& w, int N, int K, bool int8) {
  DeviceMatrix m;
  m.N = N;
  m.K = K;
  if (!int8) {
    m.f16 = to_device_f16(w);
    return m;
  }
  std::vector<std::int8_t> q(w.size());
  std::vector<float> sc(N);
  for (int n = 0; n < N; ++n) {
    float mx = 0.0f;
    for (int k = 0; k < K; ++k) mx = std::max(mx, std::fabs(w[static_cast<std::size_t>(n) * K + k]));
    sc[n] = mx > 0.0f ? mx / 127.0f : 1.0f;
    for (int k = 0; k < K; ++k)
      q[static_cast<std::size_t>(n) * K + k] =
          static_cast<std::int8_t>(std::lrintf(w[static_cast<std::size_t>(n) * K + k] / sc[n]));
  }
  m.i8 = DeviceArray<std::int8_t>(q.size());
  m.i8.upload(q.data(), q.size());
  m.scale = to_device_f32(sc);
  return m;
}

struct DeviceLayer {
  DeviceArray<float> attn_norm, mlp_norm, bqkv;
  DeviceMatrix wqkv, wo, w_gate_up, w_down;
};

class CudaBackend final : public Backend {
 public:
  CudaBackend(const HostWeights& w, int num_blocks, int block_size, int device, int max_tokens, int max_context,
              CudaOptions options)
      : c_(w.config), opt_(options), max_tokens_(max_tokens), max_context_(max_context) {
    CUDA_CHECK(cudaSetDevice(device));
    device_ = device;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&copy_stream_, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasCreate(&cublas_));
    CUBLAS_CHECK(cublasSetStream(cublas_, stream_));
    layer_done_.resize(c_.layers);
    for (auto& e : layer_done_) CUDA_CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
    layout_ = {c_.layers, c_.kv_heads, c_.head_dim, block_size, num_blocks};
    if (kv_block_bytes() % 16 != 0) throw std::invalid_argument("a KV block must be a multiple of 16 bytes");

    const int D = c_.head_dim, G = c_.heads / c_.kv_heads;
    fast_attention_ = !opt_.reference_attention && (D == 16 || D == 32 || D == 64) && G >= 1 && G <= 8 &&
                      block_size * D * 2 % 16 == 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    // The M0 kernel keeps one float per visible position in shared memory.
    std::size_t smem = static_cast<std::size_t>(max_context_ + c_.head_dim) * sizeof(float);
    if (!fast_attention_ && smem > prop.sharedMemPerBlockOptin)
      throw std::invalid_argument("max_context too long for the reference attention kernel's shared memory");
    if (fast_attention_) smem = 0;
    // The limit belongs to the kernel, for the whole process, not to this backend: only
    // ever raise it, or a second backend with a shorter context would break the first.
    {
      static std::mutex mu;
      static std::size_t current = 0;
      std::lock_guard<std::mutex> lock(mu);
      if (smem > current && !fast_attention_) {
        CUDA_CHECK(cudaFuncSetAttribute(attention_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        static_cast<int>(smem)));
        current = smem;
      }
    }

    embed_ = to_device_f16(w.embed);
    lm_head_ = w.lm_head.empty() ? DeviceArray<__half>() : to_device_f16(w.lm_head);
    final_norm_ = to_device_f32(w.final_norm);
    inv_freq_ = to_device_f32(c_.rope_inv_freq());
    for (const LayerWeights& L : w.layers) {
      DeviceLayer d;
      d.attn_norm = to_device_f32(L.attn_norm);
      d.mlp_norm = to_device_f32(L.mlp_norm);
      d.bqkv = to_device_f32(L.bqkv);
      const int H = c_.hidden, QKV = c_.q_dim() + 2 * c_.kv_dim(), I = c_.intermediate;
      const bool q8 = opt_.int8_weights;
      d.wqkv = to_device_matrix(L.wqkv, QKV, H, q8);
      d.wo = to_device_matrix(L.wo, H, c_.q_dim(), q8);
      d.w_gate_up = to_device_matrix(L.w_gate_up, 2 * I, H, q8);
      d.w_down = to_device_matrix(L.w_down, H, I, q8);
      if (q8) dequant_elems_ = std::max({dequant_elems_, d.wqkv.i8.n, d.wo.i8.n, d.w_gate_up.i8.n, d.w_down.i8.n});
      layers_.push_back(std::move(d));
    }
    kc_ = DeviceArray<__half>(static_cast<std::size_t>(layout_.layer_elems()) * c_.layers);
    vc_ = DeviceArray<__half>(kc_.n);
    CUDA_CHECK(cudaMemset(kc_.p, 0, kc_.n * 2));
    CUDA_CHECK(cudaMemset(vc_.p, 0, vc_.n * 2));

    const std::size_t T = max_tokens_;
    const int H = c_.hidden, QKV = c_.q_dim() + 2 * c_.kv_dim(), I = c_.intermediate;
    x_ = DeviceArray<float>(T * H);
    xn_ = DeviceArray<__half>(T * H);
    qkv_ = DeviceArray<float>(T * QKV);
    attn_ = DeviceArray<__half>(T * c_.q_dim());
    gu_ = DeviceArray<float>(T * 2 * I);
    act_ = DeviceArray<__half>(T * I);
    logits_ = DeviceArray<float>(T * c_.vocab);
    tokens_ = DeviceArray<int>(T);
    pos_ = DeviceArray<int>(T);
    seq_ = DeviceArray<int>(T);
    rows_ = DeviceArray<int>(T);
    table_off_ = DeviceArray<int>(T);
    tables_ = DeviceArray<int>(T + static_cast<std::size_t>(num_blocks) * 4 + 1024);
    decode_tokens_ = DeviceArray<int>(T);
    prefill_tiles_ = DeviceArray<int2>(T);
    if (opt_.int8_weights) dequant_ = DeviceArray<__half>(dequant_elems_);
  }

  // Feeds `src` (host memory, or device memory on any GPU) to `apply` as float32 chunks
  // on this device: in place when it already lives here, else through a staging buffer
  // of at most kStageFloats (a peer copy between GPUs, or an upload from the host).
  template <class F>
  void copy_converted(const float* src, std::size_t n, F&& apply) {
    cudaPointerAttributes a{};
    CUDA_CHECK(cudaPointerGetAttributes(&a, src));
    if (a.type == cudaMemoryTypeDevice && a.device == device_) {
      apply(src, 0, n);
      return;
    }
    constexpr std::size_t kStageFloats = std::size_t(16) << 20;  // 64 MB
    if (stage_.n < std::min(n, kStageFloats)) stage_ = DeviceArray<float>(std::min(n, kStageFloats));
    for (std::size_t off = 0; off < n; off += kStageFloats) {
      std::size_t len = std::min(kStageFloats, n - off);
      if (a.type == cudaMemoryTypeDevice)
        CUDA_CHECK(cudaMemcpyPeerAsync(stage_.p, device_, src + off, a.device, len * 4, stream_));
      else
        CUDA_CHECK(cudaMemcpyAsync(stage_.p, src + off, len * 4, cudaMemcpyHostToDevice, stream_));
      apply(stage_.p, off, len);
    }
  }

  ~CudaBackend() override {
    cudaSetDevice(device_);
    if (pinned_) cudaFreeHost(pinned_);
    if (cublas_) cublasDestroy(cublas_);
    for (auto e : layer_done_) cudaEventDestroy(e);
    if (stream_) cudaStreamDestroy(stream_);
    if (copy_stream_) cudaStreamDestroy(copy_stream_);
  }

  const ModelConfig& config() const override { return c_; }
  const KVLayout& kv_layout() const override { return layout_; }
  std::string name() const override { return "cuda"; }

  void update_weight(const std::string& name, const float* src, std::size_t n) override {
    CUDA_CHECK(cudaSetDevice(device_));
    auto check = [&](std::size_t want) {
      if (n != want) throw std::invalid_argument("update_weight " + name + ": wrong size");
    };
    auto to_f16 = [&](DeviceArray<__half>& dst) {
      check(dst.n);
      copy_converted(src, n, [&](const float* chunk, std::size_t off, std::size_t len) {
        f32_to_f16_kernel<<<static_cast<unsigned>(std::min<std::size_t>((len + 255) / 256, 4096)), 256, 0, stream_>>>(
            chunk, dst.p + off, len);
      });
    };
    auto to_f32 = [&](DeviceArray<float>& dst) {
      check(dst.n);
      copy_converted(src, n, [&](const float* chunk, std::size_t off, std::size_t len) {
        CUDA_CHECK(cudaMemcpyAsync(dst.p + off, chunk, len * 4, cudaMemcpyDeviceToDevice, stream_));
      });
    };
    auto matrix = [&](DeviceMatrix& m) {
      if (m.i8.p) throw std::logic_error("update_weight: int8 weights must be requantized; rebuild the backend");
      to_f16(m.f16);
    };
    if (name == "embed") {
      to_f16(embed_);
    } else if (name == "lm_head") {
      if (!lm_head_.n) throw std::invalid_argument("update_weight: tied embeddings have no lm_head");
      to_f16(lm_head_);
    } else if (name == "final_norm") {
      to_f32(final_norm_);
    } else if (name.rfind("layers.", 0) == 0) {
      std::size_t dot = name.find('.', 7);
      int i = dot == std::string::npos ? -1 : std::stoi(name.substr(7, dot - 7));
      if (i < 0 || i >= static_cast<int>(layers_.size())) throw std::invalid_argument("update_weight: no " + name);
      DeviceLayer& L = layers_[i];
      std::string f = name.substr(dot + 1);
      if (f == "attn_norm") to_f32(L.attn_norm);
      else if (f == "mlp_norm") to_f32(L.mlp_norm);
      else if (f == "bqkv" && L.bqkv.n) to_f32(L.bqkv);
      else if (f == "wqkv") matrix(L.wqkv);
      else if (f == "wo") matrix(L.wo);
      else if (f == "w_gate_up") matrix(L.w_gate_up);
      else if (f == "w_down") matrix(L.w_down);
      else throw std::invalid_argument("update_weight: no " + name);
    } else {
      throw std::invalid_argument("update_weight: no " + name);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream_));
  }
  std::size_t kv_block_bytes() const override { return static_cast<std::size_t>(layout_.block_elems()) * 2; }

  void read_kv_block(int layer, int block, void* k_out, void* v_out) override {
    CUDA_CHECK(cudaSetDevice(device_));
    read_kv_layer(layer, {block}, nullptr, k_out, v_out);
  }

  void write_kv_block(int layer, int block, const void* k_in, const void* v_in) override {
    CUDA_CHECK(cudaSetDevice(device_));
    write_kv_layer(layer, {block}, k_in, v_in);
  }

  void read_kv_layer(int layer, const std::vector<int>& blocks, void* out) override {
    CUDA_CHECK(cudaSetDevice(device_));
    auto* k = static_cast<unsigned char*>(out);
    read_kv_layer(layer, blocks, nullptr, k, k + blocks.size() * kv_block_bytes());
  }

  void write_kv_layer(int layer, const std::vector<int>& blocks, const void* in) override {
    CUDA_CHECK(cudaSetDevice(device_));
    const auto* k = static_cast<const unsigned char*>(in);
    write_kv_layer(layer, blocks, k, k + blocks.size() * kv_block_bytes());
  }

  void wait_kv_written(int layer) override { CUDA_CHECK(cudaEventSynchronize(layer_done_.at(layer))); }

  std::vector<float> forward(const ForwardBatch& batch, LayerObserver* observer) override {
    CUDA_CHECK(cudaSetDevice(device_));  // the caller's current device may be another GPU (a trainer's)
    const int H = c_.hidden, Q = c_.q_dim(), I = c_.intermediate, D = c_.head_dim, V = c_.vocab;

    std::vector<int> tok, pos, seq, table_off, tables, rows;
    for (int s = 0; s < static_cast<int>(batch.seqs.size()); ++s) {
      const SeqChunk& ch = batch.seqs[s];
      int end = ch.start_pos + static_cast<int>(ch.tokens.size());
      if (static_cast<int>(ch.block_table.size()) < layout_.blocks_for(end))
        throw std::invalid_argument("block table too short");
      if (end > max_context_) throw std::invalid_argument("sequence longer than max_context");
      table_off.push_back(static_cast<int>(tables.size()));
      tables.insert(tables.end(), ch.block_table.begin(), ch.block_table.end());
      for (std::size_t i = 0; i < ch.tokens.size(); ++i) {
        if (ch.tokens[i] < 0 || ch.tokens[i] >= V) throw std::out_of_range("token id out of range");
        if (batch.all_logits) rows.push_back(static_cast<int>(tok.size()));
        tok.push_back(ch.tokens[i]);
        pos.push_back(ch.start_pos + static_cast<int>(i));
        seq.push_back(s);
      }
      if (!batch.all_logits && ch.want_logits) rows.push_back(static_cast<int>(tok.size()) - 1);
    }
    const int T = static_cast<int>(tok.size()), R = static_cast<int>(rows.size());
    if (T == 0) return {};
    // Which attention kernel each token goes through: single-token chunks to the decode
    // kernel, longer chunks to the prefill kernel in tiles of up to 64 rows.
    std::vector<int> decode;
    std::vector<int2> tiles;
    int max_ctx = 0;
    if (fast_attention_) {
      int t = 0;
      for (const SeqChunk& ch : batch.seqs) {
        const int n = static_cast<int>(ch.tokens.size());
        if (n == 1) {
          decode.push_back(t);
          max_ctx = std::max(max_ctx, ch.start_pos + 1);
        } else {
          for (int r = 0; r < n; r += kernels::kFlashRows) tiles.push_back(make_int2(t + r, std::min(kernels::kFlashRows, n - r)));
        }
        t += n;
      }
    }
    if (T > max_tokens_) throw std::invalid_argument("batch has more tokens than max_batch_tokens");
    if (tables.size() > tables_.n) tables_ = DeviceArray<int>(tables.size() * 2);
    if (table_off.size() > table_off_.n) table_off_ = DeviceArray<int>(table_off.size() * 2);
    upload(tokens_, tok);
    upload(pos_, pos);
    upload(seq_, seq);
    upload(table_off_, table_off);
    upload(tables_, tables);
    if (R) upload(rows_, rows);
    if (!decode.empty()) upload(decode_tokens_, decode);
    if (!tiles.empty()) {
      if (tiles.size() > prefill_tiles_.n) throw std::logic_error("more prefill tiles than tokens");
      CUDA_CHECK(cudaMemcpyAsync(prefill_tiles_.p, tiles.data(), tiles.size() * sizeof(int2), cudaMemcpyHostToDevice, stream_));
    }
    const int parts = decode.empty() ? 0 : (max_ctx + kernels::kDecodeSlice - 1) / kernels::kDecodeSlice;
    if (!decode.empty()) {
      const std::size_t need = decode.size() * static_cast<std::size_t>(c_.heads) * parts;
      if (need * D > part_o_.n) part_o_ = DeviceArray<float>(need * D * 2);
      if (need * 2 > part_ml_.n) part_ml_ = DeviceArray<float>(need * 4);
    }
    TokenMeta meta{pos_.p, seq_.p, table_off_.p, tables_.p};

    const float eps = static_cast<float>(c_.rms_eps), scale = 1.0f / std::sqrt(static_cast<float>(D));
    const std::size_t smem = fast_attention_ ? 0 : static_cast<std::size_t>(max_context_ + D) * sizeof(float);

    embed_kernel<<<T, 256, 0, stream_>>>(tokens_.p, embed_.p, x_.p, H);
    for (int l = 0; l < c_.layers; ++l) {
      DeviceLayer& L = layers_[l];
      __half* kc = kc_.p + static_cast<std::size_t>(l) * layout_.layer_elems();
      __half* vc = vc_.p + static_cast<std::size_t>(l) * layout_.layer_elems();
      rmsnorm_kernel<<<T, 256, 0, stream_>>>(x_.p, nullptr, L.attn_norm.p, xn_.p, H, eps);
      gemm(L.wqkv, xn_.p, qkv_.p, T, 0.0f);
      qkv_post_kernel<<<T, 256, 0, stream_>>>(qkv_.p, L.bqkv.n ? L.bqkv.p : nullptr, inv_freq_.p, meta, kc, vc, c_.heads,
                                  c_.kv_heads, D, layout_.block_size);
      CUDA_CHECK(cudaEventRecord(layer_done_[l], stream_));
      if (observer) observer->kv_written(l);
      if (fast_attention_) {
        attention_m1(meta, decode, tiles, parts, kc, vc, scale);
      } else {
        attention_kernel<<<dim3(T, c_.heads), 128, smem, stream_>>>(qkv_.p, meta, kc, vc, attn_.p, c_.heads,
                                                                    c_.kv_heads, D, layout_.block_size, scale);
      }
      gemm(L.wo, attn_.p, x_.p, T, 1.0f);  // x += attn @ wo^T
      rmsnorm_kernel<<<T, 256, 0, stream_>>>(x_.p, nullptr, L.mlp_norm.p, xn_.p, H, eps);
      gemm(L.w_gate_up, xn_.p, gu_.p, T, 0.0f);
      silu_mul_kernel<<<T, 256, 0, stream_>>>(gu_.p, act_.p, I);
      gemm(L.w_down, act_.p, x_.p, T, 1.0f);  // x += act @ w_down^T
    }
    std::vector<float> out(static_cast<std::size_t>(R) * V);
    if (R) {
      rmsnorm_kernel<<<R, 256, 0, stream_>>>(x_.p, rows_.p, final_norm_.p, xn_.p, H, eps);
      gemm(lm_head_.n ? lm_head_.p : embed_.p, xn_.p, logits_.p, R, H, V, 0.0f);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaMemcpyAsync(out.data(), logits_.p, out.size() * sizeof(float), cudaMemcpyDeviceToHost, stream_));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    return out;
  }

 private:
  void upload(DeviceArray<int>& d, const std::vector<int>& h) {
    if (h.size() > d.n) throw std::logic_error("metadata larger than its device buffer");
    CUDA_CHECK(cudaMemcpyAsync(d.p, h.data(), h.size() * sizeof(int), cudaMemcpyHostToDevice, stream_));
  }

  // K blocks to k_out and V blocks to v_out, on the copy stream.
  // KV traffic for the transfer engine. One layer's blocks are gathered on the GPU into a
  // contiguous buffer and moved in a single copy to page-locked host memory (only
  // page-locked memory lets a copy run while kernels run); writes go the other way and
  // are scattered by a kernel. All on the copy stream, never waiting for the forward pass.
  void ensure_staging(std::size_t n_blocks) {
    std::size_t bytes = 2 * n_blocks * kv_block_bytes();
    if (bytes > staging_dev_.n * sizeof(uint4)) {
      staging_dev_ = DeviceArray<uint4>((bytes + 15) / 16);
    }
    std::size_t need = bytes + n_blocks * sizeof(int);
    if (need > pinned_bytes_) {
      if (pinned_) CUDA_CHECK(cudaFreeHost(pinned_));
      CUDA_CHECK(cudaMallocHost(&pinned_, need));
      pinned_bytes_ = need;
    }
    if (n_blocks > staging_ids_.n) staging_ids_ = DeviceArray<int>(n_blocks);
  }

  void read_kv_layer(int layer, const std::vector<int>& blocks, void*, void* k_out, void* v_out) {
    std::lock_guard<std::mutex> lock(copy_mu_);
    const std::size_t bb = kv_block_bytes(), n = blocks.size();
    ensure_staging(n);
    auto* ids = reinterpret_cast<int*>(static_cast<char*>(pinned_) + 2 * n * bb);
    std::memcpy(ids, blocks.data(), n * sizeof(int));
    CUDA_CHECK(cudaMemcpyAsync(staging_ids_.p, ids, n * sizeof(int), cudaMemcpyHostToDevice, copy_stream_));
    const std::size_t off = static_cast<std::size_t>(layer) * layout_.layer_elems();
    kv_gather_kernel<<<dim3(static_cast<unsigned>(n), 2), 256, 0, copy_stream_>>>(
        reinterpret_cast<const uint4*>(kc_.p + off), reinterpret_cast<const uint4*>(vc_.p + off), staging_ids_.p,
        static_cast<int>(n), static_cast<long long>(bb / 16), staging_dev_.p);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpyAsync(pinned_, staging_dev_.p, 2 * n * bb, cudaMemcpyDeviceToHost, copy_stream_));
    CUDA_CHECK(cudaStreamSynchronize(copy_stream_));
    std::memcpy(k_out, pinned_, n * bb);
    std::memcpy(v_out, static_cast<char*>(pinned_) + n * bb, n * bb);
  }

  void write_kv_layer(int layer, const std::vector<int>& blocks, const void* k_in, const void* v_in) {
    std::lock_guard<std::mutex> lock(copy_mu_);
    const std::size_t bb = kv_block_bytes(), n = blocks.size();
    ensure_staging(n);
    std::memcpy(pinned_, k_in, n * bb);
    std::memcpy(static_cast<char*>(pinned_) + n * bb, v_in, n * bb);
    auto* ids = reinterpret_cast<int*>(static_cast<char*>(pinned_) + 2 * n * bb);
    std::memcpy(ids, blocks.data(), n * sizeof(int));
    CUDA_CHECK(cudaMemcpyAsync(staging_ids_.p, ids, n * sizeof(int), cudaMemcpyHostToDevice, copy_stream_));
    CUDA_CHECK(cudaMemcpyAsync(staging_dev_.p, pinned_, 2 * n * bb, cudaMemcpyHostToDevice, copy_stream_));
    const std::size_t off = static_cast<std::size_t>(layer) * layout_.layer_elems();
    kv_scatter_kernel<<<dim3(static_cast<unsigned>(n), 2), 256, 0, copy_stream_>>>(
        reinterpret_cast<uint4*>(kc_.p + off), reinterpret_cast<uint4*>(vc_.p + off), staging_ids_.p,
        static_cast<int>(n), static_cast<long long>(bb / 16), staging_dev_.p);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(copy_stream_));
  }

  void attention_m1(const TokenMeta& meta, const std::vector<int>& decode, const std::vector<int2>& tiles, int parts,
                    const __half* kc, const __half* vc, float scale) {
    const int D = c_.head_dim, G = c_.heads / c_.kv_heads, bs = layout_.block_size;
    if (!tiles.empty()) {
      dim3 grid(static_cast<unsigned>(tiles.size()), c_.heads);
      switch (D) {
        case 16: launch_flash<16>(grid, meta, kc, vc, bs, scale); break;
        case 32: launch_flash<32>(grid, meta, kc, vc, bs, scale); break;
        case 64: launch_flash<64>(grid, meta, kc, vc, bs, scale); break;
      }
    }
    if (!decode.empty()) {
      dim3 grid(static_cast<unsigned>(decode.size()), c_.kv_heads, parts);
      switch (D) {
        case 16: launch_decode<16>(G, grid, meta, kc, vc, bs, scale, parts); break;
        case 32: launch_decode<32>(G, grid, meta, kc, vc, bs, scale, parts); break;
        case 64: launch_decode<64>(G, grid, meta, kc, vc, bs, scale, parts); break;
      }
      kernels::decode_combine_kernel<<<dim3(static_cast<unsigned>(decode.size()), c_.heads), D, 0, stream_>>>(
          part_o_.p, part_ml_.p, decode_tokens_.p, attn_.p, c_.heads, D, parts);
    }
  }

  template <int D>
  void launch_flash(dim3 grid, const TokenMeta& meta, const __half* kc, const __half* vc, int bs, float scale) {
    kernels::flash_prefill_kernel<D><<<grid, kernels::kFlashWarps * 32, kernels::flash_smem_bytes<D>(), stream_>>>(
        qkv_.p, meta, prefill_tiles_.p, kc, vc, attn_.p, c_.heads, c_.kv_heads, bs, scale);
  }

  template <int D>
  void launch_decode(int G, dim3 grid, const TokenMeta& meta, const __half* kc, const __half* vc, int bs, float scale,
                     int parts) {
#define RELAY_DECODE(g)                                                                                         \
  case g:                                                                                                       \
    kernels::decode_attention_kernel<D, g><<<grid, kernels::kDecodeThreads, 0, stream_>>>(                     \
        qkv_.p, meta, decode_tokens_.p, kc, vc, part_o_.p, part_ml_.p, c_.heads, c_.kv_heads, bs, scale, parts); \
    break;
    switch (G) {
      RELAY_DECODE(1) RELAY_DECODE(2) RELAY_DECODE(3) RELAY_DECODE(4)
      RELAY_DECODE(5) RELAY_DECODE(6) RELAY_DECODE(7) RELAY_DECODE(8)
    }
#undef RELAY_DECODE
  }

  // y[T, N] = x[T, K] @ W[N, K]^T + beta * y, with W fp16 or int8.
  void gemm(const DeviceMatrix& W, const __half* x, float* y, int T, float beta) {
    if (!W.i8.p) return gemm(W.f16.p, x, y, T, W.K, W.N, beta);
    if (T <= kernels::kInt8MaxTokens) {
      const int warps = kernels::kInt8Warps;
      kernels::gemv_int8_kernel<kernels::kInt8MaxTokens><<<(W.N + warps - 1) / warps, warps * 32, 0, stream_>>>(
          W.i8.p, W.scale.p, x, y, T, W.K, W.N, beta);
      return;
    }
    const long long total = static_cast<long long>(W.N) * W.K;
    kernels::dequant_int8_kernel<<<1024, 256, 0, stream_>>>(W.i8.p, W.scale.p, dequant_.p, W.K, total);
    gemm(dequant_.p, x, y, T, W.K, W.N, beta);
  }

  // y[T, N] = x[T, K] @ W[N, K]^T + beta * y. Row-major y is column-major y^T (N x T),
  // which is W (column-major K x N, transposed) times x (column-major K x T).
  void gemm(const __half* W, const __half* x, float* y, int T, int K, int N, float beta) {
    const float alpha = 1.0f;
    CUBLAS_CHECK(cublasGemmEx(cublas_, CUBLAS_OP_T, CUBLAS_OP_N, N, T, K, &alpha, W, CUDA_R_16F, K, x, CUDA_R_16F, K,
                              &beta, y, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
  }

  ModelConfig c_;
  CudaOptions opt_;
  bool fast_attention_ = false;
  std::size_t dequant_elems_ = 0;
  KVLayout layout_;
  int max_tokens_, max_context_;
  cublasHandle_t cublas_ = nullptr;
  cudaStream_t stream_ = nullptr, copy_stream_ = nullptr;
  int device_ = 0;
  DeviceArray<float> stage_;  // update_weight's staging buffer
  std::vector<cudaEvent_t> layer_done_;
  std::mutex copy_mu_;
  DeviceArray<uint4> staging_dev_;
  DeviceArray<int> staging_ids_;
  void* pinned_ = nullptr;
  std::size_t pinned_bytes_ = 0;
  DeviceArray<__half> embed_, lm_head_;
  DeviceArray<float> final_norm_, inv_freq_;
  std::vector<DeviceLayer> layers_;
  DeviceArray<__half> kc_, vc_;
  DeviceArray<float> x_, qkv_, gu_, logits_;
  DeviceArray<__half> xn_, attn_, act_;
  DeviceArray<int> tokens_, pos_, seq_, rows_, table_off_, tables_;
  DeviceArray<int> decode_tokens_;
  DeviceArray<int2> prefill_tiles_;
  DeviceArray<float> part_o_, part_ml_;  // decode slices' partial outputs and (max, sum)
  DeviceArray<__half> dequant_;          // int8 weights dequantized for a large batch
};

}  // namespace

std::unique_ptr<Backend> make_cuda_backend(const HostWeights& weights, int num_blocks, int block_size, int device,
                                           int max_batch_tokens, int max_context, CudaOptions options) {
  return std::make_unique<CudaBackend>(weights, num_blocks, block_size, device, max_batch_tokens, max_context, options);
}

}  // namespace relay
