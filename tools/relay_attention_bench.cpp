// relay-attention-bench: forward-pass time on the CUDA backend with M0's attention kernel,
// M1's kernels, and M1 with int8 weights.
//
//   relay-attention-bench --model DIR [--repeats 5]
//
// Prefill: one prompt of 512 or 2048 tokens, in chunks of 512 (as the engine schedules it).
// Decode: B sequences whose contexts are C tokens long each add one token per forward pass.
// Times are medians over the repeats, measured around Backend::forward (which ends with a
// stream synchronize), so they include everything a step costs: GEMMs, attention, the rest.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "relay/backend.h"

using namespace relay;
using Clock = std::chrono::steady_clock;

#ifdef RELAY_CUDA
namespace {

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

struct Bench {
  const HostWeights& w;
  CudaOptions opt;
  int repeats;

  // Milliseconds for one prompt of `len` tokens, chunked by 512.
  double prefill(int len) {
    const int bs = 16, blocks = (len + bs - 1) / bs + 1;
    auto be = make_cuda_backend(w, blocks, bs, 0, 512, blocks * bs, opt);
    std::mt19937 rng(1);
    std::vector<int> prompt(len);
    for (int& t : prompt) t = static_cast<int>(rng() % w.config.vocab);
    std::vector<int> table(blocks);
    for (int i = 0; i < blocks; ++i) table[i] = i;
    std::vector<double> ms;
    for (int r = 0; r <= repeats; ++r) {  // the first run warms up
      auto t0 = Clock::now();
      for (int s = 0; s < len; s += 512) {
        ForwardBatch b;
        SeqChunk ch;
        ch.tokens.assign(prompt.begin() + s, prompt.begin() + std::min(len, s + 512));
        ch.start_pos = s;
        ch.block_table = table;
        b.seqs.push_back(ch);
        be->forward(b);
      }
      if (r) ms.push_back(std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
    }
    return median(ms);
  }

  // Milliseconds per decode step for `batch` sequences at context `ctx`.
  double decode(int batch, int ctx) {
    const int bs = 16, per = (ctx + 64 + bs - 1) / bs;
    auto be = make_cuda_backend(w, batch * per + 1, bs, 0, 512, per * bs, opt);
    std::vector<std::vector<int>> tables(batch);
    for (int i = 0; i < batch; ++i)
      for (int k = 0; k < per; ++k) tables[i].push_back(i * per + k);
    // Fill each sequence's cache with a prompt of ctx tokens (untimed).
    std::mt19937 rng(2);
    for (int i = 0; i < batch; ++i)
      for (int s = 0; s < ctx; s += 512) {
        ForwardBatch b;
        SeqChunk ch;
        for (int t = s; t < std::min(ctx, s + 512); ++t) ch.tokens.push_back(static_cast<int>(rng() % w.config.vocab));
        ch.start_pos = s;
        ch.block_table = tables[i];
        b.seqs.push_back(ch);
        be->forward(b);
      }
    std::vector<double> ms;
    for (int r = 0; r <= repeats * 4; ++r) {
      ForwardBatch b;
      for (int i = 0; i < batch; ++i) {
        SeqChunk ch;
        ch.tokens = {static_cast<int>(rng() % w.config.vocab)};
        ch.start_pos = ctx + r;
        ch.block_table = tables[i];
        b.seqs.push_back(ch);
      }
      auto t0 = Clock::now();
      be->forward(b);
      if (r) ms.push_back(std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
    }
    return median(ms);
  }
};

}  // namespace

int main(int argc, char** argv) {
  std::string model;
  int repeats = 5;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--model") && i + 1 < argc) model = argv[++i];
    else if (!std::strcmp(argv[i], "--repeats") && i + 1 < argc) repeats = std::stoi(argv[++i]);
  }
  if (model.empty()) {
    std::fprintf(stderr, "usage: relay-attention-bench --model DIR [--repeats 5]\n");
    return 2;
  }
  HostWeights w = HostWeights::load(model);
  CudaOptions m0, m1, m1_int8;
  m0.reference_attention = true;
  m1_int8.int8_weights = true;
  Bench b0{w, m0, repeats}, b1{w, m1, repeats}, b8{w, m1_int8, repeats};
  std::printf("model %s, CUDA backend, median of %d runs\n", model.c_str(), repeats);
  std::printf("%-28s %12s %12s %12s %10s\n", "", "M0 kernel", "M1 kernels", "M1 + int8", "M1 vs M0");
  for (int len : {512, 2048}) {
    double a = b0.prefill(len), c = b1.prefill(len), d = b8.prefill(len);
    std::printf("prefill %4d tokens (ms)    %12.1f %12.1f %12.1f %9.2fx\n", len, a, c, d, a / c);
  }
  for (int ctx : {512, 2048})
    for (int batch : {1, 8, 32}) {
      double a = b0.decode(batch, ctx), c = b1.decode(batch, ctx), d = b8.decode(batch, ctx);
      std::printf("decode b=%-2d ctx=%-4d (ms/step) %10.2f %12.2f %12.2f %9.2fx   (%.0f tok/s with M1, %.0f with int8)\n",
                  batch, ctx, a, c, d, a / c, batch * 1000.0 / c, batch * 1000.0 / d);
    }
  return 0;
}
#else
int main() {
  std::fprintf(stderr, "relay-attention-bench needs a CUDA build (-DRELAY_CUDA=ON)\n");
  return 1;
}
#endif
