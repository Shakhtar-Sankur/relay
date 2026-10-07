// relay's C interface: thin wrappers that turn exceptions into error strings.
#include "relay/c_api.h"

#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

#include "relay/backend.h"
#include "relay/engine.h"
#include "relay/weights.h"

using namespace relay;

struct relay_model {
  HostWeights w;
};

struct relay_engine {
  relay_model* model;
  std::string backend_name;
  int device, num_blocks, block_size;
  EngineOptions opt;
  std::unique_ptr<Backend> backend;
  std::unique_ptr<Engine> engine;
  bool host_stale = false;  // the backend's weights were updated past the model's host copy
};

namespace {

thread_local std::string last_error;

int fail(const std::exception& e) {
  last_error = e.what();
  return -1;
}

template <class F>
int guarded(F&& f) {
  try {
    f();
    return 0;
  } catch (const std::exception& e) {
    return fail(e);
  }
}

void build(relay_engine* e) {
  e->engine.reset();
  e->backend.reset();
  if (e->backend_name == "cpu") {
    e->backend = make_cpu_backend(e->model->w, e->num_blocks, e->block_size);
  } else if (e->backend_name == "cuda") {
#ifdef RELAY_CUDA
    e->backend = make_cuda_backend(e->model->w, e->num_blocks, e->block_size, e->device, e->opt.max_batch_tokens,
                                   e->num_blocks * e->block_size);
#else
    throw std::runtime_error("relay was built without CUDA");
#endif
  } else {
    throw std::invalid_argument("unknown backend '" + e->backend_name + "' (cpu or cuda)");
  }
  e->engine = std::make_unique<Engine>(*e->backend, e->opt);
}

Request make_request(uint64_t id, const int* prompt, int n, const relay_sampling* sp) {
  if (!prompt || n <= 0) throw std::invalid_argument("empty prompt");
  if (!sp) throw std::invalid_argument("no sampling parameters");
  Request r;
  r.id = id;
  r.prompt.assign(prompt, prompt + n);
  r.params.temperature = sp->temperature;
  r.params.top_p = sp->top_p;
  r.params.top_k = sp->top_k;
  r.params.max_new_tokens = sp->max_new_tokens;
  r.params.seed = sp->seed;
  r.params.ignore_eos = sp->ignore_eos != 0;
  return r;
}

std::vector<float>* find_tensor(HostWeights& w, const std::string& name) {
  if (name == "embed") return &w.embed;
  if (name == "final_norm") return &w.final_norm;
  if (name == "lm_head") return w.lm_head.empty() ? nullptr : &w.lm_head;
  if (name.rfind("layers.", 0) != 0) return nullptr;
  std::size_t dot = name.find('.', 7);
  if (dot == std::string::npos) return nullptr;
  int i = std::stoi(name.substr(7, dot - 7));
  if (i < 0 || i >= static_cast<int>(w.layers.size())) return nullptr;
  LayerWeights& L = w.layers[i];
  std::string f = name.substr(dot + 1);
  if (f == "attn_norm") return &L.attn_norm;
  if (f == "wqkv") return &L.wqkv;
  if (f == "bqkv") return L.bqkv.empty() ? nullptr : &L.bqkv;
  if (f == "wo") return &L.wo;
  if (f == "mlp_norm") return &L.mlp_norm;
  if (f == "w_gate_up") return &L.w_gate_up;
  if (f == "w_down") return &L.w_down;
  return nullptr;
}

}  // namespace

extern "C" {

const char* relay_last_error(void) { return last_error.c_str(); }

int relay_cuda_available(void) {
#ifdef RELAY_CUDA
  return 1;
#else
  return 0;
#endif
}

relay_model* relay_model_load(const char* dir) {
  try {
    auto m = std::make_unique<relay_model>();
    m->w = HostWeights::load(dir);
    return m.release();
  } catch (const std::exception& e) {
    fail(e);
    return nullptr;
  }
}

void relay_model_free(relay_model* m) { delete m; }

int relay_model_config(const relay_model* m, relay_config* out) {
  return guarded([&] {
    const ModelConfig& c = m->w.config;
    *out = relay_config{};
    out->hidden = c.hidden;
    out->intermediate = c.intermediate;
    out->layers = c.layers;
    out->heads = c.heads;
    out->kv_heads = c.kv_heads;
    out->head_dim = c.head_dim;
    out->vocab = c.vocab;
    out->max_position = c.max_position;
    out->rms_eps = c.rms_eps;
    out->rope_theta = c.rope_theta;
    out->tie_embeddings = m->w.lm_head.empty() ? 1 : 0;
    out->qkv_bias = c.qkv_bias ? 1 : 0;
    out->num_eos = static_cast<int>(std::min<std::size_t>(c.eos_ids.size(), RELAY_MAX_EOS));
    for (int i = 0; i < out->num_eos; ++i) out->eos_ids[i] = c.eos_ids[i];
  });
}

int relay_model_rope_inv_freq(const relay_model* m, float* out) {
  return guarded([&] {
    std::vector<float> f = m->w.config.rope_inv_freq();
    std::memcpy(out, f.data(), f.size() * sizeof(float));
  });
}

float* relay_model_tensor(relay_model* m, const char* name, int64_t* numel) {
  try {
    std::vector<float>* t = find_tensor(m->w, name);
    if (!t) {
      last_error = std::string("no tensor '") + name + "'";
      return nullptr;
    }
    if (numel) *numel = static_cast<int64_t>(t->size());
    return t->data();
  } catch (const std::exception& e) {
    fail(e);
    return nullptr;
  }
}

relay_engine* relay_engine_new(relay_model* m, const char* backend, int device, int num_blocks, int block_size,
                               int max_batch_tokens, int max_seqs, int prefix_caching) {
  try {
    auto e = std::make_unique<relay_engine>();
    e->model = m;
    e->backend_name = backend;
    e->device = device;
    e->num_blocks = num_blocks;
    e->block_size = block_size;
    e->opt.max_batch_tokens = max_batch_tokens;
    e->opt.max_seqs = max_seqs;
    e->opt.prefix_caching = prefix_caching != 0;
    e->opt.logprobs = true;
    build(e.get());
    return e.release();
  } catch (const std::exception& ex) {
    fail(ex);
    return nullptr;
  }
}

void relay_engine_free(relay_engine* e) { delete e; }

int relay_engine_add(relay_engine* e, uint64_t id, const int* prompt, int n, const relay_sampling* sp) {
  return guarded([&] { e->engine->add(make_request(id, prompt, n, sp)); });
}

int relay_engine_add_resume(relay_engine* e, uint64_t id, const int* prompt, int n, const relay_sampling* sp,
                            const int* generated, int ng) {
  return guarded([&] {
    if (ng < 0 || (ng > 0 && !generated)) throw std::invalid_argument("bad generated tokens");
    std::vector<int> g(generated, generated + ng);
    e->engine->add_resume(make_request(id, prompt, n, sp), g);
  });
}

int relay_engine_step(relay_engine* e, relay_event* out, int cap) {
  try {
    std::vector<TokenEvent> ev = e->engine->step();
    if (static_cast<int>(ev.size()) > cap) throw std::length_error("event buffer too small");
    for (std::size_t i = 0; i < ev.size(); ++i)
      out[i] = relay_event{ev[i].id, ev[i].token, ev[i].index, static_cast<int>(ev[i].finish), ev[i].logprob};
    return static_cast<int>(ev.size());
  } catch (const std::exception& ex) {
    return fail(ex);
  }
}

int relay_engine_has_work(relay_engine* e) { return e->engine->has_work() ? 1 : 0; }

int relay_engine_cancel(relay_engine* e, uint64_t id) { return e->engine->cancel(id) ? 1 : 0; }

void relay_engine_cancel_all(relay_engine* e) { e->engine->cancel_all(); }

int relay_engine_reload_weights(relay_engine* e) {
  return guarded([&] {
    if (e->engine->has_work()) throw std::logic_error("requests in flight: reload weights between rollouts");
    if (e->host_stale)
      throw std::logic_error("the backend's weights are newer than the host copy (relay_engine_update_tensor): "
                             "a reload would bring the old weights back");
    build(e);
  });
}

int relay_engine_update_tensor(relay_engine* e, const char* name, const float* src, int64_t numel) {
  return guarded([&] {
    if (e->engine->has_work()) throw std::logic_error("requests in flight: update weights between rollouts");
    if (!src || numel < 0) throw std::invalid_argument("bad source");
    if (e->backend_name == "cpu") {
      std::vector<float>* t = find_tensor(e->model->w, name);
      if (!t) throw std::invalid_argument(std::string("no tensor '") + name + "'");
      if (static_cast<int64_t>(t->size()) != numel) throw std::invalid_argument(std::string(name) + ": wrong size");
      std::memcpy(t->data(), src, static_cast<std::size_t>(numel) * sizeof(float));
    } else {
      e->backend->update_weight(name, src, static_cast<std::size_t>(numel));
      e->host_stale = true;
    }
  });
}

int relay_engine_finish_update(relay_engine* e) {
  return guarded([&] { e->engine->drop_prefix_cache(); });
}

int relay_engine_stats(relay_engine* e, relay_stats* out) {
  return guarded([&] {
    EngineStats s = e->engine->stats();
    *out = relay_stats{s.steps, s.forward_tokens, s.preemptions, s.prompt_tokens, s.prefix_hit_tokens};
  });
}

}  // extern "C"
