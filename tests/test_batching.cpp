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

// Requests that share a long prefix (a system prompt) with different endings.
std::vector<Request> shared_prefix_requests(int vocab) {
  std::mt19937 rng(21);
  std::vector<int> system;
  for (int i = 0; i < 37; ++i) system.push_back(static_cast<int>(rng() % vocab));
  std::vector<Request> rs;
  for (int i = 0; i < 6; ++i) {
    Request r;
    r.id = 700 + i;
    r.prompt = system;
    for (int t = 0; t < 3 + i * 4; ++t) r.prompt.push_back(static_cast<int>(rng() % vocab));
    r.params.max_new_tokens = 5 + i;
    r.params.ignore_eos = true;
    if (i % 2) {
      r.params.temperature = 0.7f;
      r.params.seed = 9 + i;
    }
    rs.push_back(r);
  }
  return rs;
}

// Runs the requests one after another through one engine (so later ones can find the
// earlier ones' blocks in the cache) and returns tokens, logits and stats.
std::map<std::uint64_t, std::vector<int>> one_engine_in_turn(const HostWeights& w, const std::vector<Request>& rs,
                                                             int blocks, int block_size, bool caching, LogitLog& log,
                                                             EngineStats& st) {
  auto be = check::make_backend(w, blocks, block_size);
  Engine e(*be, {512, 64, caching});
  e.on_logits = [&](std::uint64_t id, int index, const float* lg) {
    log[{id, index}] = std::vector<float>(lg, lg + w.config.vocab);
  };
  std::map<std::uint64_t, std::vector<int>> out;
  for (const auto& r : rs) {
    e.add(r);
    for (auto& [id, toks] : e.run_all()) out[id] = toks;
  }
  CHECK_EQ(e.free_blocks(), blocks);
  st = e.stats();
  return out;
}

TEST(prefix_caching_reuses_blocks_and_changes_nothing) {
  HostWeights w = HostWeights::load(kModel);
  auto rs = shared_prefix_requests(w.config.vocab);
  LogitLog off_log, on_log;
  EngineStats off, on;
  auto a = one_engine_in_turn(w, rs, 64, 8, false, off_log, off);
  auto b = one_engine_in_turn(w, rs, 64, 8, true, on_log, on);
  CHECK(a == b);
  if (check::exact_backend()) compare(off_log, on_log);
  CHECK_EQ(off.prefix_hit_tokens, 0u);
  // Every request after the first finds the 37-token system prompt's 4 full blocks.
  CHECK(on.prefix_hit_tokens >= 5u * 32u);
  CHECK(on.forward_tokens < off.forward_tokens);
  std::fprintf(stderr, "  prompt tokens %llu, from the prefix cache %llu; forward tokens %llu -> %llu\n",
               (unsigned long long)on.prompt_tokens, (unsigned long long)on.prefix_hit_tokens,
               (unsigned long long)off.forward_tokens, (unsigned long long)on.forward_tokens);
}

TEST(prefix_caching_under_eviction_and_preemption_changes_nothing) {
  HostWeights w = HostWeights::load(kModel);
  auto rs = shared_prefix_requests(w.config.vocab);
  LogitLog off_log, on_log;
  EngineStats off, on;
  auto a = run(w, rs, 64, 4, {512, 64}, false, off_log);
  // A cache small enough that cached blocks are evicted and requests are preempted.
  auto be = check::make_backend(w, 22, 4);
  Engine e(*be, {16, 64, true});
  e.on_logits = [&](std::uint64_t id, int index, const float* lg) {
    on_log[{id, index}] = std::vector<float>(lg, lg + w.config.vocab);
  };
  for (const auto& r : rs) e.add(r);
  auto b = e.run_all();
  CHECK(a == b);
  if (check::exact_backend()) compare(off_log, on_log);
  CHECK_EQ(e.free_blocks(), 22);
  std::fprintf(stderr, "  %llu preemptions, %llu prompt tokens from the cache\n",
               (unsigned long long)e.stats().preemptions, (unsigned long long)e.stats().prefix_hit_tokens);
}

TEST(block_manager_evicts_least_recently_used_first) {
  BlockManager m(4, 2, true);
  std::vector<int> t = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<int> a, b;
  std::vector<std::uint64_t> ha, hb;
  CHECK(m.allocate(2, a));
  m.register_full(t, 4, a, ha);  // blocks for tokens [1,2] and [3,4]
  CHECK_EQ(m.num_cached(), 2);
  m.release_all(a);               // unused but cached: still available
  CHECK_EQ(m.num_available(), 4);
  std::vector<int> c;
  std::vector<std::uint64_t> hc;
  CHECK_EQ(m.match_prefix(t, 3, c, hc), 2);  // only whole blocks below 3 tokens
  m.release_all(c);
  CHECK(m.allocate(3, b));        // 2 free, then evicts the least recently used cached block
  CHECK_EQ(m.num_cached(), 1);
  std::vector<int> d;
  std::vector<std::uint64_t> hd;
  CHECK(m.match_prefix(t, 8, d, hd) <= 2);
}

TEST(cancel_frees_the_request_and_spares_the_others) {
  HostWeights w = HostWeights::load(kModel);
  auto rs = make_requests(w.config.vocab);
  LogitLog full_log, cut_log;
  auto full = run(w, rs, 64, 16, {512, 64}, false, full_log);
  auto be = check::make_backend(w, 64, 16);
  Engine e(*be, {512, 64});
  for (const auto& r : rs) e.add(r);
  std::map<std::uint64_t, std::vector<int>> out;
  for (const TokenEvent& ev : e.step()) out[ev.id].push_back(ev.token);
  CHECK(e.cancel(rs[2].id));   // running
  CHECK(!e.cancel(rs[2].id));  // already gone
  CHECK(!e.cancel(424242));
  while (e.has_work())
    for (const TokenEvent& ev : e.step()) out[ev.id].push_back(ev.token);
  CHECK_EQ(e.free_blocks(), 64);
  for (const auto& r : rs) {
    if (r.id == rs[2].id) {
      CHECK(out[r.id].size() <= 1);
    } else {
      CHECK(out[r.id] == full[r.id]);
    }
  }
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
