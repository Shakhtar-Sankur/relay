// The CPU backend: float32 throughout. It is the reference every other backend is
// checked against, so it favors being obviously right over being fast, with one
// rule that matters for testing: each output element is computed by the same
// sequence of floating-point operations no matter how many other tokens are in the
// batch. Running a sequence alone or inside any batch therefore gives bit-identical
// logits, which the batching tests rely on.

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include "relay/backend.h"

namespace relay {
namespace {

// Dot product with 8 independent partial sums, added in a fixed order at the end.
// Fixed order means fixed result; 8 lanes lets the compiler use one AVX register.
inline float dot(const float* a, const float* b, int n) {
  float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  int i = 0;
  for (; i + 8 <= n; i += 8)
    for (int l = 0; l < 8; ++l) acc[l] += a[i + l] * b[i + l];
  float tail = 0;
  for (; i < n; ++i) tail += a[i] * b[i];
  return ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7])) + tail;
}

// y[t, n] = sum_k x[t, k] * W[n, k] (+ y[t, n] if accumulate). x: [T, K], W: [N, K].
void matmul(const float* x, const float* W, float* y, int T, int K, int N, bool accumulate) {
  constexpr int kRows = 16;
#pragma omp parallel for schedule(static)
  for (int n0 = 0; n0 < N; n0 += kRows) {
    int n1 = std::min(N, n0 + kRows);
    for (int t = 0; t < T; ++t) {
      const float* xt = x + static_cast<std::int64_t>(t) * K;
      float* yt = y + static_cast<std::int64_t>(t) * N;
      for (int n = n0; n < n1; ++n) {
        float v = dot(xt, W + static_cast<std::int64_t>(n) * K, K);
        yt[n] = accumulate ? yt[n] + v : v;
      }
    }
  }
}

void rmsnorm(const float* x, const float* w, float* out, int n, float eps) {
  double sum = 0;  // double so the mean does not depend on summation order details
  for (int i = 0; i < n; ++i) sum += static_cast<double>(x[i]) * x[i];
  float inv = 1.0f / std::sqrt(static_cast<float>(sum / n) + eps);
  for (int i = 0; i < n; ++i) out[i] = w[i] * (x[i] * inv);
}

class CpuBackend final : public Backend {
 public:
  CpuBackend(const HostWeights& w, int num_blocks, int block_size) : w_(w), c_(w.config) {
    layout_.layers = c_.layers;
    layout_.kv_heads = c_.kv_heads;
    layout_.head_dim = c_.head_dim;
    layout_.block_size = block_size;
    layout_.num_blocks = num_blocks;
    k_.assign(static_cast<std::size_t>(layout_.layer_elems() * c_.layers), 0.0f);
    v_.assign(k_.size(), 0.0f);
    inv_freq_ = c_.rope_inv_freq();
  }

  const ModelConfig& config() const override { return c_; }
  const KVLayout& kv_layout() const override { return layout_; }
  std::string name() const override { return "cpu"; }

  std::size_t kv_block_bytes() const override { return static_cast<std::size_t>(layout_.block_elems()) * 4; }

  void read_kv_block(int layer, int block, void* k_out, void* v_out) override {
    std::int64_t off = layer * layout_.layer_elems() + block * layout_.block_elems();
    std::memcpy(k_out, k_.data() + off, kv_block_bytes());
    std::memcpy(v_out, v_.data() + off, kv_block_bytes());
  }

  void write_kv_block(int layer, int block, const void* k_in, const void* v_in) override {
    std::int64_t off = layer * layout_.layer_elems() + block * layout_.block_elems();
    std::memcpy(k_.data() + off, k_in, kv_block_bytes());
    std::memcpy(v_.data() + off, v_in, kv_block_bytes());
  }

  void* kv_host_ptr(int layer, int block, bool value) override {
    std::int64_t off = layer * layout_.layer_elems() + block * layout_.block_elems();
    return (value ? v_.data() : k_.data()) + off;
  }

  std::vector<float> forward(const ForwardBatch& batch, LayerObserver* observer) override {
    const int H = c_.hidden, Q = c_.q_dim(), KV = c_.kv_dim(), I = c_.intermediate, D = c_.head_dim;
    const int QKV = Q + 2 * KV;

    // Flatten the chunks: token t belongs to sequence seq_of[t] at position pos[t].
    std::vector<int> tok, pos, seq_of;
    for (int s = 0; s < static_cast<int>(batch.seqs.size()); ++s) {
      const SeqChunk& ch = batch.seqs[s];
      int need = layout_.blocks_for(ch.start_pos + static_cast<int>(ch.tokens.size()));
      if (static_cast<int>(ch.block_table.size()) < need) throw std::invalid_argument("block table too short");
      for (std::size_t i = 0; i < ch.tokens.size(); ++i) {
        tok.push_back(ch.tokens[i]);
        pos.push_back(ch.start_pos + static_cast<int>(i));
        seq_of.push_back(s);
      }
    }
    const int T = static_cast<int>(tok.size());
    if (T == 0) return {};

    std::vector<float> x(static_cast<std::size_t>(T) * H), xn(x.size());
    std::vector<float> qkv(static_cast<std::size_t>(T) * QKV), attn(static_cast<std::size_t>(T) * Q);
    std::vector<float> gu(static_cast<std::size_t>(T) * 2 * I), act(static_cast<std::size_t>(T) * I);

    for (int t = 0; t < T; ++t) {
      if (tok[t] < 0 || tok[t] >= c_.vocab) throw std::out_of_range("token id out of range");
      std::memcpy(&x[static_cast<std::size_t>(t) * H], &w_.embed[static_cast<std::size_t>(tok[t]) * H], H * 4);
    }

    const float scale = 1.0f / std::sqrt(static_cast<float>(D));
    const int group = c_.heads / c_.kv_heads;

    for (int l = 0; l < c_.layers; ++l) {
      const LayerWeights& L = w_.layers[l];
      float* Kc = k_.data() + l * layout_.layer_elems();
      float* Vc = v_.data() + l * layout_.layer_elems();

#pragma omp parallel for
      for (int t = 0; t < T; ++t)
        rmsnorm(&x[static_cast<std::size_t>(t) * H], L.attn_norm.data(), &xn[static_cast<std::size_t>(t) * H], H,
                static_cast<float>(c_.rms_eps));
      matmul(xn.data(), L.wqkv.data(), qkv.data(), T, H, QKV, false);

      // Bias, rotary embedding on q and k, then write k and v into the cache.
#pragma omp parallel for
      for (int t = 0; t < T; ++t) {
        float* row = &qkv[static_cast<std::size_t>(t) * QKV];
        if (!L.bqkv.empty())
          for (int i = 0; i < QKV; ++i) row[i] += L.bqkv[i];
        rope(row, c_.heads, pos[t]);           // q
        rope(row + Q, c_.kv_heads, pos[t]);    // k
        const SeqChunk& ch = batch.seqs[seq_of[t]];
        int block = ch.block_table[pos[t] / layout_.block_size], slot = pos[t] % layout_.block_size;
        for (int h = 0; h < c_.kv_heads; ++h) {
          std::int64_t off = layout_.offset(block, h, slot);
          std::memcpy(Kc + off, row + Q + h * D, D * 4);
          std::memcpy(Vc + off, row + Q + KV + h * D, D * 4);
        }
      }
      // This layer's K and V for the whole batch are in the cache; nothing later in
      // this forward pass writes them, so they can be sent while the next layers run.
      if (observer) observer->kv_written(l);

      // Causal attention: the token at position p sees positions 0..p of its own sequence.
#pragma omp parallel for collapse(2) schedule(dynamic)
      for (int t = 0; t < T; ++t) {
        for (int h = 0; h < c_.heads; ++h) {
          const SeqChunk& ch = batch.seqs[seq_of[t]];
          const float* q = &qkv[static_cast<std::size_t>(t) * QKV + h * D];
          const int kvh = h / group, n = pos[t] + 1;
          std::vector<float> score(n);
          float mx = -INFINITY;
          for (int j = 0; j < n; ++j) {
            int b = ch.block_table[j / layout_.block_size], s = j % layout_.block_size;
            score[j] = dot(q, Kc + layout_.offset(b, kvh, s), D) * scale;
            mx = std::max(mx, score[j]);
          }
          float sum = 0;
          for (int j = 0; j < n; ++j) {
            score[j] = std::exp(score[j] - mx);
            sum += score[j];
          }
          float* out = &attn[static_cast<std::size_t>(t) * Q + h * D];
          std::fill(out, out + D, 0.0f);
          for (int j = 0; j < n; ++j) {
            int b = ch.block_table[j / layout_.block_size], s = j % layout_.block_size;
            const float* v = Vc + layout_.offset(b, kvh, s);
            float p = score[j] / sum;
            for (int d = 0; d < D; ++d) out[d] += p * v[d];
          }
        }
      }
      matmul(attn.data(), L.wo.data(), x.data(), T, Q, H, true);  // residual add

#pragma omp parallel for
      for (int t = 0; t < T; ++t)
        rmsnorm(&x[static_cast<std::size_t>(t) * H], L.mlp_norm.data(), &xn[static_cast<std::size_t>(t) * H], H,
                static_cast<float>(c_.rms_eps));
      matmul(xn.data(), L.w_gate_up.data(), gu.data(), T, H, 2 * I, false);
#pragma omp parallel for
      for (int t = 0; t < T; ++t) {
        const float* g = &gu[static_cast<std::size_t>(t) * 2 * I];
        const float* u = g + I;
        float* a = &act[static_cast<std::size_t>(t) * I];
        for (int i = 0; i < I; ++i) a[i] = g[i] / (1.0f + std::exp(-g[i])) * u[i];
      }
      matmul(act.data(), L.w_down.data(), x.data(), T, I, H, true);  // residual add
    }

    // Final norm and LM head, only on the rows that need logits.
    std::vector<int> rows;
    int t0 = 0;
    for (const SeqChunk& ch : batch.seqs) {
      int n = static_cast<int>(ch.tokens.size());
      if (batch.all_logits) {
        for (int i = 0; i < n; ++i) rows.push_back(t0 + i);
      } else if (ch.want_logits) {
        rows.push_back(t0 + n - 1);
      }
      t0 += n;
    }
    const int R = static_cast<int>(rows.size());
    std::vector<float> xr(static_cast<std::size_t>(R) * H);
    for (int r = 0; r < R; ++r)
      rmsnorm(&x[static_cast<std::size_t>(rows[r]) * H], w_.final_norm.data(), &xr[static_cast<std::size_t>(r) * H], H,
              static_cast<float>(c_.rms_eps));
    std::vector<float> logits(static_cast<std::size_t>(R) * c_.vocab);
    matmul(xr.data(), w_.lm_head_or_embed().data(), logits.data(), R, H, c_.vocab, false);
    return logits;
  }

 private:
  // Rotates pairs (i, i + D/2) of each head by angle pos * inv_freq[i]: the
  // "rotate_half" convention of Hugging Face Llama.
  void rope(float* v, int nheads, int p) const {
    const int D = c_.head_dim, half = D / 2;
    for (int i = 0; i < half; ++i) {
      float angle = static_cast<float>(p) * inv_freq_[i];
      float cs = std::cos(angle), sn = std::sin(angle);
      for (int h = 0; h < nheads; ++h) {
        float* x = v + h * D;
        float a = x[i], b = x[i + half];
        x[i] = a * cs - b * sn;
        x[i + half] = b * cs + a * sn;
      }
    }
  }

  const HostWeights& w_;
  ModelConfig c_;
  KVLayout layout_;
  std::vector<float> k_, v_;
  std::vector<float> inv_freq_;
};

}  // namespace

std::unique_ptr<Backend> make_cpu_backend(const HostWeights& weights, int num_blocks, int block_size) {
  return std::make_unique<CpuBackend>(weights, num_blocks, block_size);
}

}  // namespace relay
