// Picks the next token from a row of logits.
//
// Sampling is a pure function of (logits, params, seed, index): the random number
// for a request's n-th generated token comes from hashing (seed, n), not from a
// generator that advances as the request runs. So a request gives the same tokens
// whether it ran alone, in a batch, was preempted and recomputed, or (in later
// milestones) had its prefill done on another worker.
#pragma once

#include <cstdint>
#include <vector>

namespace relay {

struct SamplingParams {
  float temperature = 0.0f;  // 0: greedy
  float top_p = 1.0f;
  int top_k = 0;  // 0: no limit
  int max_new_tokens = 16;
  std::uint64_t seed = 0;
  bool ignore_eos = false;
};

// A uniform number in [0, 1) from (seed, index): splitmix64 of their mix.
double uniform01(std::uint64_t seed, std::uint64_t index);

int sample(const float* logits, int vocab, const SamplingParams& p, std::uint64_t index);
int argmax(const float* logits, int vocab);
// log softmax(logits / temperature)[token], in double precision (temperature <= 0: 1).
double log_prob(const float* logits, int vocab, float temperature, int token);

}  // namespace relay
