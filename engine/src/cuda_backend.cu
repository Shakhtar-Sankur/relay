// The CUDA backend (M0 version): fp16 weights and KV cache, fp32 residual stream.
//
// Matrix multiplies go to cuBLAS (fp16 inputs, fp32 accumulation and output; the
// residual adds are folded into the GEMM with beta = 1). Everything else is a small
// kernel here: embedding lookup, RMSNorm, bias + RoPE + KV-cache write, attention over
// the paged cache, SiLU-and-multiply. The attention kernel is deliberately simple
// (one thread block per token and head, scores in shared memory); M1 replaces it with
// a paged decode kernel and a flash-style prefill kernel.
//
// Streams: the forward pass runs on its own stream; KV block reads and writes (the
// transfer engine's traffic) run on a second one, so a transfer never waits for the
// forward pass to finish. An event recorded after each layer's KV write tells the
// transfer engine when that layer can be read (wait_kv_written).
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

#include "relay/backend.h"
#include "relay/safetensors.h"

namespace relay {
namespace {

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

struct TokenMeta {
  const int* pos;         // [T]
  const int* seq;         // [T] chunk index of each token
  const int* table_off;   // [S] offset of each chunk's block table in tables
  const int* tables;      // concatenated block tables
};

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

// ---- backend ----------------------------------------------------------------

struct DeviceLayer {
  DeviceArray<float> attn_norm, mlp_norm, bqkv;
  DeviceArray<__half> wqkv, wo, w_gate_up, w_down;
};

class CudaBackend final : public Backend {
 public:
  CudaBackend(const HostWeights& w, int num_blocks, int block_size, int device, int max_tokens, int max_context)
      : c_(w.config), max_tokens_(max_tokens), max_context_(max_context) {
    CUDA_CHECK(cudaSetDevice(device));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&copy_stream_, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasCreate(&cublas_));
    CUBLAS_CHECK(cublasSetStream(cublas_, stream_));
    layer_done_.resize(c_.layers);
    for (auto& e : layer_done_) CUDA_CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
    layout_ = {c_.layers, c_.kv_heads, c_.head_dim, block_size, num_blocks};

    // Attention keeps one float per visible position in shared memory.
    std::size_t smem = static_cast<std::size_t>(max_context_ + c_.head_dim) * sizeof(float);
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    if (smem > prop.sharedMemPerBlockOptin)
      throw std::invalid_argument("max_context too long for this GPU's shared memory (M1's kernel lifts this)");
    CUDA_CHECK(cudaFuncSetAttribute(attention_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    static_cast<int>(smem)));

    embed_ = to_device_f16(w.embed);
    lm_head_ = w.lm_head.empty() ? DeviceArray<__half>() : to_device_f16(w.lm_head);
    final_norm_ = to_device_f32(w.final_norm);
    inv_freq_ = to_device_f32(c_.rope_inv_freq());
    for (const LayerWeights& L : w.layers) {
      DeviceLayer d;
      d.attn_norm = to_device_f32(L.attn_norm);
      d.mlp_norm = to_device_f32(L.mlp_norm);
      d.bqkv = to_device_f32(L.bqkv);
      d.wqkv = to_device_f16(L.wqkv);
      d.wo = to_device_f16(L.wo);
      d.w_gate_up = to_device_f16(L.w_gate_up);
      d.w_down = to_device_f16(L.w_down);
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
  }

  ~CudaBackend() override {
    if (cublas_) cublasDestroy(cublas_);
    for (auto e : layer_done_) cudaEventDestroy(e);
    if (stream_) cudaStreamDestroy(stream_);
    if (copy_stream_) cudaStreamDestroy(copy_stream_);
  }

  const ModelConfig& config() const override { return c_; }
  const KVLayout& kv_layout() const override { return layout_; }
  std::string name() const override { return "cuda"; }
  std::size_t kv_block_bytes() const override { return static_cast<std::size_t>(layout_.block_elems()) * 2; }

  void read_kv_block(int layer, int block, void* k_out, void* v_out) override {
    read_kv_layer(layer, {block}, nullptr, k_out, v_out);
  }

  void write_kv_block(int layer, int block, const void* k_in, const void* v_in) override {
    write_kv_layer(layer, {block}, k_in, v_in);
  }

  void read_kv_layer(int layer, const std::vector<int>& blocks, void* out) override {
    auto* k = static_cast<unsigned char*>(out);
    read_kv_layer(layer, blocks, nullptr, k, k + blocks.size() * kv_block_bytes());
  }

  void write_kv_layer(int layer, const std::vector<int>& blocks, const void* in) override {
    const auto* k = static_cast<const unsigned char*>(in);
    write_kv_layer(layer, blocks, k, k + blocks.size() * kv_block_bytes());
  }

  void wait_kv_written(int layer) override { CUDA_CHECK(cudaEventSynchronize(layer_done_.at(layer))); }

  std::vector<float> forward(const ForwardBatch& batch, LayerObserver* observer) override {
    const int H = c_.hidden, Q = c_.q_dim(), KV = c_.kv_dim(), I = c_.intermediate, D = c_.head_dim, V = c_.vocab;
    const int QKV = Q + 2 * KV;

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
    if (T > max_tokens_) throw std::invalid_argument("batch has more tokens than max_batch_tokens");
    if (tables.size() > tables_.n) tables_ = DeviceArray<int>(tables.size() * 2);
    if (table_off.size() > table_off_.n) table_off_ = DeviceArray<int>(table_off.size() * 2);
    upload(tokens_, tok);
    upload(pos_, pos);
    upload(seq_, seq);
    upload(table_off_, table_off);
    upload(tables_, tables);
    if (R) upload(rows_, rows);
    TokenMeta meta{pos_.p, seq_.p, table_off_.p, tables_.p};

    const float eps = static_cast<float>(c_.rms_eps), scale = 1.0f / std::sqrt(static_cast<float>(D));
    const std::size_t smem = static_cast<std::size_t>(max_context_ + D) * sizeof(float);

    embed_kernel<<<T, 256, 0, stream_>>>(tokens_.p, embed_.p, x_.p, H);
    for (int l = 0; l < c_.layers; ++l) {
      DeviceLayer& L = layers_[l];
      __half* kc = kc_.p + static_cast<std::size_t>(l) * layout_.layer_elems();
      __half* vc = vc_.p + static_cast<std::size_t>(l) * layout_.layer_elems();
      rmsnorm_kernel<<<T, 256, 0, stream_>>>(x_.p, nullptr, L.attn_norm.p, xn_.p, H, eps);
      gemm(L.wqkv.p, xn_.p, qkv_.p, T, H, QKV, 0.0f);
      qkv_post_kernel<<<T, 256, 0, stream_>>>(qkv_.p, L.bqkv.n ? L.bqkv.p : nullptr, inv_freq_.p, meta, kc, vc, c_.heads,
                                  c_.kv_heads, D, layout_.block_size);
      CUDA_CHECK(cudaEventRecord(layer_done_[l], stream_));
      if (observer) observer->kv_written(l);
      attention_kernel<<<dim3(T, c_.heads), 128, smem, stream_>>>(qkv_.p, meta, kc, vc, attn_.p, c_.heads, c_.kv_heads, D,
                                                         layout_.block_size, scale);
      gemm(L.wo.p, attn_.p, x_.p, T, Q, H, 1.0f);  // x += attn @ wo^T
      rmsnorm_kernel<<<T, 256, 0, stream_>>>(x_.p, nullptr, L.mlp_norm.p, xn_.p, H, eps);
      gemm(L.w_gate_up.p, xn_.p, gu_.p, T, H, 2 * I, 0.0f);
      silu_mul_kernel<<<T, 256, 0, stream_>>>(gu_.p, act_.p, I);
      gemm(L.w_down.p, act_.p, x_.p, T, I, H, 1.0f);  // x += act @ w_down^T
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
  void read_kv_layer(int layer, const std::vector<int>& blocks, void*, void* k_out, void* v_out) {
    const std::size_t bb = kv_block_bytes();
    for (std::size_t i = 0; i < blocks.size(); ++i) {
      std::size_t off = static_cast<std::size_t>(layer * layout_.layer_elems() + blocks[i] * layout_.block_elems());
      CUDA_CHECK(cudaMemcpyAsync(static_cast<char*>(k_out) + i * bb, kc_.p + off, bb, cudaMemcpyDeviceToHost, copy_stream_));
      CUDA_CHECK(cudaMemcpyAsync(static_cast<char*>(v_out) + i * bb, vc_.p + off, bb, cudaMemcpyDeviceToHost, copy_stream_));
    }
    CUDA_CHECK(cudaStreamSynchronize(copy_stream_));
  }

  void write_kv_layer(int layer, const std::vector<int>& blocks, const void* k_in, const void* v_in) {
    const std::size_t bb = kv_block_bytes();
    for (std::size_t i = 0; i < blocks.size(); ++i) {
      std::size_t off = static_cast<std::size_t>(layer * layout_.layer_elems() + blocks[i] * layout_.block_elems());
      CUDA_CHECK(cudaMemcpyAsync(kc_.p + off, static_cast<const char*>(k_in) + i * bb, bb, cudaMemcpyHostToDevice, copy_stream_));
      CUDA_CHECK(cudaMemcpyAsync(vc_.p + off, static_cast<const char*>(v_in) + i * bb, bb, cudaMemcpyHostToDevice, copy_stream_));
    }
    CUDA_CHECK(cudaStreamSynchronize(copy_stream_));
  }

  // y[T, N] = x[T, K] @ W[N, K]^T + beta * y. Row-major y is column-major y^T (N x T),
  // which is W (column-major K x N, transposed) times x (column-major K x T).
  void gemm(const __half* W, const __half* x, float* y, int T, int K, int N, float beta) {
    const float alpha = 1.0f;
    CUBLAS_CHECK(cublasGemmEx(cublas_, CUBLAS_OP_T, CUBLAS_OP_N, N, T, K, &alpha, W, CUDA_R_16F, K, x, CUDA_R_16F, K,
                              &beta, y, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
  }

  ModelConfig c_;
  KVLayout layout_;
  int max_tokens_, max_context_;
  cublasHandle_t cublas_ = nullptr;
  cudaStream_t stream_ = nullptr, copy_stream_ = nullptr;
  std::vector<cudaEvent_t> layer_done_;
  DeviceArray<__half> embed_, lm_head_;
  DeviceArray<float> final_norm_, inv_freq_;
  std::vector<DeviceLayer> layers_;
  DeviceArray<__half> kc_, vc_;
  DeviceArray<float> x_, qkv_, gu_, logits_;
  DeviceArray<__half> xn_, attn_, act_;
  DeviceArray<int> tokens_, pos_, seq_, rows_, table_off_, tables_;
};

}  // namespace

std::unique_ptr<Backend> make_cuda_backend(const HostWeights& weights, int num_blocks, int block_size, int device,
                                           int max_batch_tokens, int max_context) {
  return std::make_unique<CudaBackend>(weights, num_blocks, block_size, device, max_batch_tokens, max_context);
}

}  // namespace relay
