// Continuous batching must not change results: a request run alone, in a batch with
// others, with its prompt split into chunks, and after being preempted and recomputed
// must produce the same tokens from the same logits.
#include <algorithm>
#include <cstring>
#include <map>
#include <random>

#include "check.h"
#include "relay/engine.h"

using namespace relay;

namespace {

using LogitLog = std::map<std::pair<std::uint64_t, int>, std::vector<float>>;

std::vector<Request> make_requests(int vocab) {
  std::mt19937 rng(7);
  std::vector<Request> rs;
  int lengths[] = {1, 5, 17, 33, 2, 48, 9, 16};
  for (int i = 0; i < 8; ++i) {
    Request r;
    r.id = 100 + i;
    for (int t = 0; t < lengths[i]; ++t) r.prompt.push_back(static_cast<int>(rng() % vocab));
    r.params.max_new_tokens = 3 + (i * 5) % 11;
    r.params.ignore_eos = true;
    if (i % 3 == 1) {  // some requests sample instead of taking the argmax
      r.params.temperature = 0.9f;
      r.params.top_p = 0.9f;
      r.params.seed = 1000 + i;
    }
    rs.push_back(r);
  }
  return rs;
}

std::map<std::uint64_t, std::vector<int>> run(const HostWeights& w, const std::vector<Request>& rs, int blocks,
                                              int block_size, EngineOptions opt, bool one_at_a_time, LogitLog& log,
                                              EngineStats* stats = nullptr) {
  std::map<std::uint64_t, std::vector<int>> out;
  auto go = [&](const std::vector<Request>& batch) {
    auto be = check::make_backend(w, blocks, block_size);
    Engine e(*be, opt);
    e.on_logits = [&](std::uint64_t id, int index, const float* lg) {
      log[{id, index}] = std::vector<float>(lg, lg + w.config.vocab);
    };
    for (const auto& r : batch) e.add(r);
    for (auto& [id, toks] : e.run_all()) out[id] = toks;
    CHECK_EQ(e.free_blocks(), blocks);  // every block returned
    if (stats) *stats = e.stats();
  };
  if (one_at_a_time) {
    for (const auto& r : rs) go({r});
  } else {
    go(rs);
  }
  return out;
}

void compare(const LogitLog& a, const LogitLog& b) {
  CHECK_EQ(a.size(), b.size());
  int differing = 0;
  for (const auto& [key, row] : a) {
    auto it = b.find(key);
    if (it == b.end() || std::memcmp(row.data(), it->second.data(), row.size() * 4) != 0) ++differing;
  }
  CHECK_EQ(differing, 0);
}

const char* kModel = RELAY_FIXTURES "/llama-gqa-tied-bf16";

}  // namespace

TEST(batched_equals_alone) {
  HostWeights w = HostWeights::load(kModel);
  auto rs = make_requests(w.config.vocab);
  LogitLog alone, batched;
  auto a = run(w, rs, 64, 16, {512, 64}, true, alone);
  auto b = run(w, rs, 64, 16, {512, 64}, false, batched);
  CHECK(a == b);
  if (check::exact_backend()) compare(alone, batched);
}

TEST(chunked_prefill_equals_whole_prompt) {
  HostWeights w = HostWeights::load(kModel);
  auto rs = make_requests(w.config.vocab);
  LogitLog whole, chunked;
  EngineStats st;
  auto a = run(w, rs, 64, 16, {512, 64}, false, whole);
  auto b = run(w, rs, 64, 16, {7, 64}, false, chunked, &st);  // at most 7 tokens per forward pass
  CHECK(a == b);
  if (check::exact_backend()) compare(whole, chunked);
  CHECK(st.steps > 20);
}

TEST(preemption_and_recompute_change_nothing) {
  HostWeights w = HostWeights::load(kModel);
  auto rs = make_requests(w.config.vocab);
  LogitLog roomy, tight;
  EngineStats st;
  auto a = run(w, rs, 64, 4, {64, 64}, false, roomy);
  // 18 blocks of 4 tokens: enough for the longest request alone, far too few for all
  // of them at once, so the engine has to preempt and recompute.
  auto b = run(w, rs, 18, 4, {64, 64}, false, tight, &st);
  CHECK(a == b);
  if (check::exact_backend()) compare(roomy, tight);
  std::fprintf(stderr, "  %llu preemptions\n", static_cast<unsigned long long>(st.preemptions));
  CHECK(st.preemptions > 0);
}

TEST(block_size_does_not_matter) {
  HostWeights w = HostWeights::load(kModel);
  auto rs = make_requests(w.config.vocab);
  LogitLog a_log, b_log;
  auto a = run(w, rs, 128, 1, {512, 64}, false, a_log);
  auto b = run(w, rs, 8, 32, {512, 64}, false, b_log);
  CHECK(a == b);
  if (check::exact_backend()) compare(a_log, b_log);
}

TEST(rejects_requests_that_can_never_fit) {
  HostWeights w = HostWeights::load(kModel);
  auto be = check::make_backend(w, 4, 4);
  Engine e(*be, {64, 8});
  Request r;
  r.prompt.assign(15, 1);
  r.params.max_new_tokens = 2;  // 17 tokens need 5 blocks of 4; the cache has 4
  bool threw = false;
  try {
    e.add(r);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  CHECK(threw);
}

RUN_TESTS()
