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

int sample(const float* logits, int vocab, const SamplingParams& p, std::uint64_t index) {
  if (p.temperature <= 0.0f) return argmax(logits, vocab);

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
