// M1's attention kernels against M0's, on the GPU: the same schedule of forward passes on
// two CUDA backends, one with the reference kernel (RELAY_CUDA_ATTENTION=m0) and one with
// the flash-style prefill and split-context decode kernels, must give the same logits up
// to floating-point rounding. The schedule covers prompts split into chunks (queries at an
// offset in the cache, tiles crossing chunk boundaries), contexts past one decode slice,
// decode and prefill in the same batch, and every model's head layout (multi-head and
// grouped-query). Also measures weight-only int8 against fp16. Skips without CUDA.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "check.h"
#include "relay/backend.h"

using namespace relay;

namespace {

double rel_diff(const std::vector<float>& a, const std::vector<float>& b) {
  double num = 0, den = 1e-12;
  for (std::size_t i = 0; i < a.size(); ++i) {
    num += (a[i] - b[i]) * static_cast<double>(a[i] - b[i]);
    den += static_cast<double>(b[i]) * b[i];
  }
  return std::sqrt(num / den);
}

struct Result {
  double worst = 0;         // largest relative difference of any forward pass's logits
  int argmax_agree = 0, argmax_total = 0;
};

// Drives two backends through the same forward passes; the first one's greedy tokens feed
// both, so they see identical inputs throughout.
using Factory = std::function<std::unique_ptr<Backend>(const HostWeights&, int blocks, int block_size)>;

Result compare(const HostWeights& w, const Factory& make_a, const Factory& make_b, const std::vector<int>& prompt_lens,
               int chunk, int decode_steps) {
  const int bs = 16, vocab = w.config.vocab;
  int total = 0;
  for (int len : prompt_lens) total += (len + decode_steps + 1 + bs - 1) / bs;
  auto a = make_a(w, total + 4, bs);
  auto b = make_b(w, total + 4, bs);
  std::mt19937 rng(7);
  struct Seq {
    std::vector<int> tokens, table;
    int prompt = 0;  // prompt length (tokens after it were generated)
    int done = 0;    // tokens already in the cache
  };
  std::vector<Seq> seqs(prompt_lens.size());
  int next_block = 0;
  for (std::size_t i = 0; i < seqs.size(); ++i) {
    seqs[i].prompt = prompt_lens[i];
    for (int t = 0; t < prompt_lens[i]; ++t) seqs[i].tokens.push_back(static_cast<int>(rng() % vocab));
    for (int k = 0; k < (prompt_lens[i] + decode_steps + 1 + bs - 1) / bs; ++k) seqs[i].table.push_back(next_block++);
  }
  Result res;
  auto step = [&](ForwardBatch& batch, std::vector<int>& owners) {
    std::vector<float> la = a->forward(batch), lb = b->forward(batch);
    CHECK_EQ(la.size(), lb.size());
    res.worst = std::max(res.worst, rel_diff(lb, la));
    for (std::size_t r = 0; r < owners.size(); ++r) {
      const float* ra = la.data() + r * vocab;
      const float* rb = lb.data() + r * vocab;
      int ia = static_cast<int>(std::max_element(ra, ra + vocab) - ra);
      int ib = static_cast<int>(std::max_element(rb, rb + vocab) - rb);
      res.argmax_agree += ia == ib;
      ++res.argmax_total;
      Seq& s = seqs[owners[r]];
      if (s.done == s.prompt) s.tokens.push_back(ia);  // the first generated token, chosen by A
    }
  };
  // Prefill, several sequences' chunks per forward pass.
  bool more = true;
  while (more) {
    more = false;
    ForwardBatch batch;
    std::vector<int> owners;
    int budget = 2048;
    for (std::size_t i = 0; i < seqs.size(); ++i) {
      Seq& s = seqs[i];
      int left = s.prompt - s.done;  // only the prompt is prefilled
      if (left <= 0 || budget <= 0) continue;
      int n = std::min({left, chunk, budget});
      SeqChunk ch;
      ch.tokens.assign(s.tokens.begin() + s.done, s.tokens.begin() + s.done + n);
      ch.start_pos = s.done;
      ch.block_table = s.table;
      ch.want_logits = true;
      batch.seqs.push_back(ch);
      owners.push_back(static_cast<int>(i));
      s.done += n;
      budget -= n;
    }
    for (const Seq& s : seqs) more = more || s.done < s.prompt;  // also those the budget left out
    step(batch, owners);
  }
  // Decode: every sequence adds one token per forward pass; on odd steps a fresh prompt
  // chunk rides along, so decode and prefill share a batch.
  for (int d = 0; d < decode_steps; ++d) {
    ForwardBatch batch;
    std::vector<int> owners;
    for (std::size_t i = 0; i < seqs.size(); ++i) {
      Seq& s = seqs[i];
      SeqChunk ch;
      ch.tokens = {s.tokens[s.done]};
      ch.start_pos = s.done;
      ch.block_table = s.table;
      batch.seqs.push_back(ch);
      owners.push_back(static_cast<int>(i));
      s.done += 1;
    }
    if (d % 2 == 1) {
      SeqChunk extra;  // a throwaway prompt in blocks of its own
      for (int t = 0; t < 40 + d; ++t) extra.tokens.push_back(static_cast<int>(rng() % vocab));
      extra.start_pos = 0;
      for (int k = 0; k < 4; ++k) extra.block_table.push_back(next_block + k);
      batch.seqs.push_back(extra);
      owners.push_back(-1);
    }
    std::vector<float> la = a->forward(batch), lb = b->forward(batch);
    res.worst = std::max(res.worst, rel_diff(lb, la));
    for (std::size_t r = 0; r < owners.size(); ++r) {
      const float* ra = la.data() + r * vocab;
      const float* rb = lb.data() + r * vocab;
      res.argmax_agree += (std::max_element(ra, ra + vocab) - ra) == (std::max_element(rb, rb + vocab) - rb);
      ++res.argmax_total;
      if (owners[r] >= 0) seqs[owners[r]].tokens.push_back(static_cast<int>(std::max_element(ra, ra + vocab) - ra));
    }
  }
  return res;
}

std::vector<std::string> models() {
  std::vector<std::string> out = {RELAY_FIXTURES "/llama-mha", RELAY_FIXTURES "/chat-tiny", RELAY_FIXTURES "/qwen2-bias",
                                  RELAY_FIXTURES "/llama3-rope"};
  if (const char* env = std::getenv("RELAY_REFERENCE_MODELS")) {
    std::string s = env;
    for (std::size_t p = 0; p <= s.size();) {
      std::size_t q = s.find(':', p);
      if (q == std::string::npos) q = s.size();
      if (q > p) out.push_back(s.substr(p, q - p));
      p = q + 1;
    }
  }
  return out;
}

bool cuda() { return check::test_backend() == "cuda"; }

#ifdef RELAY_CUDA
Factory cuda_factory(CudaOptions o) {
  return [o](const HostWeights& w, int blocks, int bs) { return make_cuda_backend(w, blocks, bs, 0, 2048, blocks * bs, o); };
}
#endif

Factory cpu_factory() {
  return [](const HostWeights& w, int blocks, int bs) { return make_cpu_backend(w, blocks, bs); };
}

}  // namespace

// The harness itself, on the CPU backend against itself: must agree exactly.
TEST(the_comparison_schedule_runs_and_agrees_with_itself) {
  HostWeights w = HostWeights::load(RELAY_FIXTURES "/chat-tiny");
  Result r = compare(w, cpu_factory(), cpu_factory(), {1, 17, 64, 200, 700, 1300}, 192, 12);
  CHECK(r.worst == 0.0);
  CHECK_EQ(r.argmax_agree, r.argmax_total);
}

#ifdef RELAY_CUDA

TEST(m1_attention_equals_m0_attention) {
  if (!cuda()) {
    std::fprintf(stderr, "  skipped: RELAY_TEST_BACKEND=cuda runs it\n");
    return;
  }
  CudaOptions m0, m1;
  m0.reference_attention = true;
  for (const std::string& dir : models()) {
    HostWeights w = HostWeights::load(dir);
    const bool big = w.config.vocab > 10000;
    // Real models: fewer, shorter sequences (their logits are large); still past one decode slice.
    std::vector<int> lens = big ? std::vector<int>{1, 37, 300, 700} : std::vector<int>{1, 17, 64, 200, 700, 1300};
    Result r = compare(w, cuda_factory(m0), cuda_factory(m1), lens, big ? 256 : 192, big ? 8 : 12);
    std::fprintf(stderr, "  %-60s D=%d G=%d: worst relative logit difference %.2e, argmax agrees %d/%d\n", dir.c_str(),
                 w.config.head_dim, w.config.heads / w.config.kv_heads, r.worst, r.argmax_agree, r.argmax_total);
    CHECK(r.worst < 1e-2);
    CHECK(r.argmax_agree >= r.argmax_total - std::max(1, r.argmax_total / 50));  // near-ties may flip
  }
}

TEST(int8_weights_stay_close_to_fp16) {
  if (!cuda()) {
    std::fprintf(stderr, "  skipped: RELAY_TEST_BACKEND=cuda runs it\n");
    return;
  }
  CudaOptions f16, i8;
  i8.int8_weights = true;
  for (const std::string& dir : models()) {
    HostWeights w = HostWeights::load(dir);
    const bool big = w.config.vocab > 10000;
    std::vector<int> lens = big ? std::vector<int>{5, 120} : std::vector<int>{5, 60, 130};
    Result r = compare(w, cuda_factory(f16), cuda_factory(i8), lens, 64, 10);
    std::fprintf(stderr, "  %-60s int8 vs fp16: relative logit difference %.2e, argmax agrees %d/%d\n", dir.c_str(),
                 r.worst, r.argmax_agree, r.argmax_total);
    CHECK(r.worst < 0.25);  // a sanity bound; the printed numbers are the measurement
  }
}
#else
TEST(cuda_attention_tests_need_a_cuda_build) { std::fprintf(stderr, "  skipped: built without CUDA\n"); }
#endif

RUN_TESTS()
