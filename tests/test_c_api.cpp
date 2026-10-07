// The C interface (what Python and an RL trainer use): generation with per-token
// log-probabilities equals the C++ engine's, the reported log-probabilities are the
// log-softmax of the logits each token was sampled from, weights are writable by name,
// and a reload makes an engine use the new weights.
#include <cmath>
#include <cstring>
#include <map>
#include <vector>

#include "check.h"
#include "relay/c_api.h"
#include "relay/engine.h"
#include "relay/weights.h"

using namespace relay;

namespace {

const char* kModel = RELAY_FIXTURES "/chat-tiny";

relay_sampling sampling(std::uint64_t seed) {
  relay_sampling sp{};
  sp.temperature = 0.9f;
  sp.top_p = 1.0f;
  sp.max_new_tokens = 12;
  sp.seed = seed;
  sp.ignore_eos = 1;
  return sp;
}

struct Out {
  std::map<std::uint64_t, std::vector<int>> tokens;
  std::map<std::uint64_t, std::vector<float>> logprobs;
};

Out run_c(relay_engine* e, int requests) {
  for (int i = 0; i < requests; ++i) {
    std::vector<int> prompt;
    for (int t = 0; t < 3 + 5 * i; ++t) prompt.push_back(1 + (t * 7 + i) % 50);
    relay_sampling sp = sampling(100 + i);
    CHECK_EQ(relay_engine_add(e, 10 + i, prompt.data(), static_cast<int>(prompt.size()), &sp), 0);
  }
  Out o;
  std::vector<relay_event> ev(64);
  while (relay_engine_has_work(e)) {
    int n = relay_engine_step(e, ev.data(), static_cast<int>(ev.size()));
    CHECK(n >= 0);
    for (int i = 0; i < n; ++i) {
      CHECK_EQ(ev[i].index, static_cast<int>(o.tokens[ev[i].id].size()));
      o.tokens[ev[i].id].push_back(ev[i].token);
      o.logprobs[ev[i].id].push_back(ev[i].logprob);
    }
  }
  return o;
}

}  // namespace

TEST(generation_and_logprobs_match_the_engine) {
  relay_model* m = relay_model_load(kModel);
  CHECK(m != nullptr);
  relay_engine* e = relay_engine_new(m, "cpu", 0, 64, 8, 64, 8, 1);
  CHECK(e != nullptr);
  Out c = run_c(e, 4);

  // The same requests through the C++ engine, recording each token's logits.
  HostWeights w = HostWeights::load(kModel);
  auto be = make_cpu_backend(w, 64, 8);
  Engine eng(*be, {64, 8, true});
  std::map<std::pair<std::uint64_t, int>, std::vector<float>> logits;
  eng.on_logits = [&](std::uint64_t id, int idx, const float* lg) {
    logits[{id, idx}] = std::vector<float>(lg, lg + w.config.vocab);
  };
  for (int i = 0; i < 4; ++i) {
    Request r;
    r.id = 10 + i;
    for (int t = 0; t < 3 + 5 * i; ++t) r.prompt.push_back(1 + (t * 7 + i) % 50);
    r.params.temperature = 0.9f;
    r.params.max_new_tokens = 12;
    r.params.seed = 100 + i;
    r.params.ignore_eos = true;
    eng.add(r);
  }
  auto ref = eng.run_all();
  CHECK(c.tokens == ref);
  double worst = 0;
  for (auto& [id, toks] : ref)
    for (std::size_t k = 0; k < toks.size(); ++k) {
      const std::vector<float>& lg = logits[{id, static_cast<int>(k)}];
      double want = log_prob(lg.data(), w.config.vocab, 0.9f, toks[k]);
      worst = std::max(worst, std::fabs(want - c.logprobs[id][k]));
      CHECK(c.logprobs[id][k] <= 0.0f);
    }
  CHECK(worst < 1e-5);  // float storage of a double computation
  relay_engine_free(e);
  relay_model_free(m);
}

TEST(weights_are_writable_by_name_and_reload_changes_the_output) {
  relay_model* m = relay_model_load(kModel);
  relay_config cfg;
  CHECK_EQ(relay_model_config(m, &cfg), 0);
  int64_t n = 0;
  float* wqkv = relay_model_tensor(m, "layers.0.wqkv", &n);
  CHECK(wqkv != nullptr);
  CHECK_EQ(n, static_cast<int64_t>((cfg.heads + 2 * cfg.kv_heads) * cfg.head_dim) * cfg.hidden);
  CHECK(relay_model_tensor(m, "layers.99.wqkv", &n) == nullptr);
  CHECK(relay_model_tensor(m, "nonsense", &n) == nullptr);
  CHECK(std::strlen(relay_last_error()) > 0);
  std::vector<float> inv(cfg.head_dim / 2);
  CHECK_EQ(relay_model_rope_inv_freq(m, inv.data()), 0);
  CHECK(inv[0] == 1.0f);

  relay_engine* e = relay_engine_new(m, "cpu", 0, 64, 8, 64, 8, 1);
  Out before = run_c(e, 2);
  // A write, then a reload: the engine uses the new weights.
  float* fn = relay_model_tensor(m, "final_norm", &n);
  for (int64_t i = 0; i < n; ++i) fn[i] *= -1.0f;
  CHECK_EQ(relay_engine_reload_weights(e), 0);
  Out after = run_c(e, 2);
  CHECK(after.tokens != before.tokens);
  // Undo the write: the original output comes back (and the prefix cache, dropped on
  // reload, does not serve stale KV entries).
  for (int64_t i = 0; i < n; ++i) fn[i] *= -1.0f;
  CHECK_EQ(relay_engine_reload_weights(e), 0);
  CHECK(run_c(e, 2).tokens == before.tokens);
  relay_engine_free(e);
  relay_model_free(m);
}

TEST(reload_is_refused_while_requests_are_in_flight) {
  relay_model* m = relay_model_load(kModel);
  relay_engine* e = relay_engine_new(m, "cpu", 0, 64, 8, 64, 8, 1);
  int prompt[] = {1, 2, 3};
  relay_sampling sp = sampling(1);
  CHECK_EQ(relay_engine_add(e, 1, prompt, 3, &sp), 0);
  CHECK_EQ(relay_engine_reload_weights(e), -1);
  relay_engine_cancel_all(e);
  CHECK_EQ(relay_engine_reload_weights(e), 0);
  relay_engine_free(e);
  relay_model_free(m);
}

TEST(resume_continues_exactly) {
  // A rollout stopped after 5 tokens and resumed gives the same tokens and
  // log-probabilities as one that ran through (the partial-rollout case).
  relay_model* m = relay_model_load(kModel);
  relay_engine* e = relay_engine_new(m, "cpu", 0, 64, 8, 64, 8, 1);
  int prompt[] = {5, 9, 2, 7, 1, 3};
  relay_sampling sp = sampling(7);
  CHECK_EQ(relay_engine_add(e, 1, prompt, 6, &sp), 0);
  std::vector<int> full;
  std::vector<float> full_lp;
  std::vector<relay_event> ev(8);
  while (relay_engine_has_work(e)) {
    int k = relay_engine_step(e, ev.data(), 8);
    for (int i = 0; i < k; ++i) full.push_back(ev[i].token), full_lp.push_back(ev[i].logprob);
  }
  std::vector<int> head(full.begin(), full.begin() + 5);
  CHECK_EQ(relay_engine_add_resume(e, 2, prompt, 6, &sp, head.data(), 5), 0);
  std::vector<int> rest;
  std::vector<float> rest_lp;
  while (relay_engine_has_work(e)) {
    int k = relay_engine_step(e, ev.data(), 8);
    for (int i = 0; i < k; ++i) {
      CHECK_EQ(ev[i].index, 5 + static_cast<int>(rest.size()));
      rest.push_back(ev[i].token), rest_lp.push_back(ev[i].logprob);
    }
  }
  CHECK(std::vector<int>(full.begin() + 5, full.end()) == rest);
  for (std::size_t i = 0; i < rest.size(); ++i) CHECK(rest_lp[i] == full_lp[5 + i]);
  relay_engine_free(e);
  relay_model_free(m);
}

TEST(update_tensor_equals_a_host_write_and_drops_the_prefix_cache) {
  relay_model* a = relay_model_load(kModel);
  relay_model* b = relay_model_load(kModel);
  relay_engine* ea = relay_engine_new(a, "cpu", 0, 64, 8, 64, 8, 1);
  relay_engine* eb = relay_engine_new(b, "cpu", 0, 64, 8, 64, 8, 1);
  run_c(ea, 3);  // fills the prefix cache
  relay_stats st;
  // New weights for every tensor of layer 1 and the final norm: a host write + reload
  // on one engine, update_tensor + finish_update on the other.
  for (const char* name : {"layers.1.wqkv", "layers.1.wo", "layers.1.w_gate_up", "layers.1.w_down",
                           "layers.1.attn_norm", "layers.1.mlp_norm", "final_norm"}) {
    int64_t n = 0;
    float* w = relay_model_tensor(b, name, &n);
    std::vector<float> nw(w, w + n);
    for (int64_t i = 0; i < n; ++i) nw[i] = nw[i] * 1.5f + 0.01f * static_cast<float>(i % 7);
    std::memcpy(w, nw.data(), n * sizeof(float));
    CHECK_EQ(relay_engine_update_tensor(ea, name, nw.data(), n), 0);
  }
  CHECK_EQ(relay_engine_finish_update(ea), 0);
  CHECK_EQ(relay_engine_reload_weights(eb), 0);
  CHECK_EQ(relay_engine_stats(ea, &st), 0);
  uint64_t hits_before = st.prefix_hit_tokens;
  Out x = run_c(ea, 3), y = run_c(eb, 3);
  CHECK(x.tokens == y.tokens);
  CHECK(x.logprobs == y.logprobs);
  CHECK_EQ(relay_engine_stats(ea, &st), 0);
  CHECK_EQ(st.prefix_hit_tokens, hits_before);  // nothing served from the old cache
  float dummy[1] = {0};
  CHECK_EQ(relay_engine_update_tensor(ea, "final_norm", dummy, 1), -1);  // wrong size
  CHECK_EQ(relay_engine_update_tensor(ea, "layers.9.wo", dummy, 1), -1);
  relay_engine_free(ea);
  relay_engine_free(eb);
  relay_model_free(a);
  relay_model_free(b);
}

RUN_TESTS()
