#include <cmath>
#include <vector>

#include "check.h"
#include "relay/sampler.h"

using namespace relay;

TEST(greedy_is_argmax_with_lowest_id_on_ties) {
  std::vector<float> lg = {0.1f, 3.0f, -1.0f, 3.0f};
  SamplingParams p;
  CHECK_EQ(sample(lg.data(), 4, p, 0), 1);
}

TEST(top_k_1_and_tiny_top_p_are_greedy) {
  std::vector<float> lg = {0.5f, 0.2f, 2.0f, 1.9f};
  SamplingParams p;
  p.temperature = 1.0f;
  p.top_k = 1;
  for (std::uint64_t i = 0; i < 50; ++i) CHECK_EQ(sample(lg.data(), 4, p, i), 2);
  p.top_k = 0;
  p.top_p = 1e-6f;
  for (std::uint64_t i = 0; i < 50; ++i) CHECK_EQ(sample(lg.data(), 4, p, i), 2);
}

TEST(same_seed_and_index_same_token) {
  std::vector<float> lg(100);
  for (int i = 0; i < 100; ++i) lg[i] = std::sin(i * 0.37f);
  SamplingParams p;
  p.temperature = 1.0f;
  p.seed = 42;
  for (std::uint64_t i = 0; i < 20; ++i) CHECK_EQ(sample(lg.data(), 100, p, i), sample(lg.data(), 100, p, i));
  int differ = 0;
  for (std::uint64_t i = 0; i < 20; ++i) differ += sample(lg.data(), 100, p, i) != sample(lg.data(), 100, p, i + 20);
  CHECK(differ > 5);
}

TEST(samples_follow_the_softmax) {
  // logits log(0.5), log(0.3), log(0.2): over 20000 draws each frequency is within 2%.
  std::vector<float> lg = {std::log(0.5f), std::log(0.3f), std::log(0.2f)};
  SamplingParams p;
  p.temperature = 1.0f;
  p.seed = 9;
  int count[3] = {0, 0, 0};
  const int n = 20000;
  for (int i = 0; i < n; ++i) ++count[sample(lg.data(), 3, p, static_cast<std::uint64_t>(i))];
  CHECK(std::fabs(count[0] / double(n) - 0.5) < 0.02);
  CHECK(std::fabs(count[1] / double(n) - 0.3) < 0.02);
  CHECK(std::fabs(count[2] / double(n) - 0.2) < 0.02);
}

TEST(log_prob_is_the_tempered_log_softmax) {
  std::vector<float> lg = {2.0f, -1.0f, 0.5f, 3.0f};
  for (float t : {0.0f, 1.0f, 0.7f}) {
    double tt = t > 0 ? t : 1.0, z = 0;
    for (float x : lg) z += std::exp(x / tt);
    for (int k = 0; k < 4; ++k) CHECK(std::fabs(log_prob(lg.data(), 4, t, k) - (lg[k] / tt - std::log(z))) < 1e-12);
  }
  // Large logits do not overflow.
  std::vector<float> big = {1000.0f, 999.0f};
  CHECK(std::fabs(log_prob(big.data(), 2, 1.0f, 0) - (-std::log(1 + std::exp(-1.0)))) < 1e-12);
}

RUN_TESTS()
