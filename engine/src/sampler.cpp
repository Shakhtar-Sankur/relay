#include "relay/sampler.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace relay {

double uniform01(std::uint64_t seed, std::uint64_t index) {
  std::uint64_t z = seed + 0x9E3779B97F4A7C15ULL * (index + 1);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  z ^= z >> 31;
  return static_cast<double>(z >> 11) * (1.0 / 9007199254740992.0);  // 53 random bits
}

int argmax(const float* logits, int vocab) {
  int best = 0;
  for (int i = 1; i < vocab; ++i)
    if (logits[i] > logits[best]) best = i;  // ties: the lowest id wins
  return best;
}

double log_prob(const float* logits, int vocab, float temperature, int token) {
  const double t = temperature > 0.0f ? temperature : 1.0;
  double mx = logits[0];
  for (int i = 1; i < vocab; ++i) mx = std::max(mx, static_cast<double>(logits[i]));
  double sum = 0;
  for (int i = 0; i < vocab; ++i) sum += std::exp((logits[i] - mx) / t);
  return (logits[token] - mx) / t - std::log(sum);
}

int sample_with_log_prob(const float* logits, int vocab, const SamplingParams& p, std::uint64_t index,
                         double* log_prob_out) {
  if (p.temperature > 0.0f && (p.top_k <= 0 || p.top_k >= vocab) && p.top_p >= 1.0f) {
    const float inv_t = 1.0f / p.temperature;
    float mx = logits[0];
    for (int i = 1; i < vocab; ++i) mx = std::max(mx, logits[i]);
    thread_local std::vector<float> e;
    e.resize(vocab);
    double total = 0;
    for (int i = 0; i < vocab; ++i) total += e[i] = std::exp((logits[i] - mx) * inv_t);
    double r = uniform01(p.seed, index) * total;
    int tok = vocab - 1;
    for (int i = 0; i < vocab; ++i) {
      r -= e[i];
      if (r < 0) {
        tok = i;
        break;
      }
    }
    if (r >= 0)  // rounding: the last token with any mass
      while (tok > 0 && e[tok] == 0.0f) --tok;
    *log_prob_out = (static_cast<double>(logits[tok]) - mx) * inv_t - std::log(total);
    return tok;
  }
  int tok = sample(logits, vocab, p, index);
  *log_prob_out = log_prob(logits, vocab, p.temperature, tok);
  return tok;
}

int sample(const float* logits, int vocab, const SamplingParams& p, std::uint64_t index) {
  if (p.temperature <= 0.0f) return argmax(logits, vocab);

  // No truncation (no top-k, no top-p, as RL sampling uses): the inverse CDF in id order,
  // with no sort (sorting a 150k-entry vocabulary for every token cost more than the GPU's
  // forward pass). Shared with sample_with_log_prob, so both pick the same token.
  if ((p.top_k <= 0 || p.top_k >= vocab) && p.top_p >= 1.0f) {
    double lp;
    return sample_with_log_prob(logits, vocab, p, index, &lp);
  }

  // Candidates sorted by logit (ties by id, so the order is fully determined).
  std::vector<int> ids(vocab);
  std::iota(ids.begin(), ids.end(), 0);
  int k = (p.top_k > 0 && p.top_k < vocab) ? p.top_k : vocab;
  auto by_logit = [&](int a, int b) { return logits[a] > logits[b] || (logits[a] == logits[b] && a < b); };
  std::partial_sort(ids.begin(), ids.begin() + k, ids.end(), by_logit);
  ids.resize(k);

  std::vector<double> prob(k);
  double mx = logits[ids[0]] / p.temperature, total = 0;
  for (int i = 0; i < k; ++i) total += prob[i] = std::exp(logits[ids[i]] / p.temperature - mx);

  // Nucleus: the smallest prefix whose probability reaches top_p.
  int keep = k;
  if (p.top_p < 1.0f) {
    double cum = 0;
    for (int i = 0; i < k; ++i) {
      cum += prob[i] / total;
      if (cum >= p.top_p) {
        keep = i + 1;
        break;
      }
    }
  }
  double kept = 0;
  for (int i = 0; i < keep; ++i) kept += prob[i];
  double r = uniform01(p.seed, index) * kept;
  for (int i = 0; i < keep; ++i) {
    r -= prob[i];
    if (r < 0) return ids[i];
  }
  return ids[keep - 1];
}

}  // namespace relay
