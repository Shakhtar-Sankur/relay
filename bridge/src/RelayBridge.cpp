#include "RelayBridge.h"

#include <memory>

#include "relay/backend.h"
#include "relay/engine.h"
#include "relay/weights.h"

namespace relaybridge {

namespace {
thread_local std::string g_last_error;
}

struct LocalWorker::Impl {
  relay::HostWeights weights;
  std::unique_ptr<relay::Backend> backend;
  std::unique_ptr<relay::Engine> engine;
};

LocalWorker::LocalWorker(Impl* impl) : impl_(impl) {}
LocalWorker::~LocalWorker() { delete impl_; }


LocalWorker* LocalWorker::create(const WorkerOptions& o) {
  try {
    auto impl = std::make_unique<Impl>();
    impl->weights = relay::HostWeights::load(o.modelDir);
    impl->backend = relay::make_cpu_backend(impl->weights, o.blocks, o.blockSize);
    impl->engine = std::make_unique<relay::Engine>(*impl->backend, relay::EngineOptions{o.maxBatchTokens, o.maxSeqs});
    return new LocalWorker(impl.release());
  } catch (const std::exception& e) {
    g_last_error = e.what();
    return nullptr;
  }
}

std::string LocalWorker::lastError() { return g_last_error; }

std::string LocalWorker::add(uint64_t request, const TokenVector& prompt, const SamplingOptions& s) {
  try {
    relay::Request r;
    r.id = request;
    r.prompt.assign(prompt.begin(), prompt.end());
    r.params.temperature = s.temperature;
    r.params.top_p = s.topP;
    r.params.top_k = s.topK;
    r.params.max_new_tokens = s.maxNewTokens;
    r.params.seed = s.seed;
    r.params.ignore_eos = s.ignoreEos;
    for (int t : r.prompt)
      if (t < 0 || t >= impl_->weights.config.vocab) return "token id out of range";
    impl_->engine->add(std::move(r));
    return "";
  } catch (const std::exception& e) {
    return e.what();
  }
}

StepResult LocalWorker::step() {
  StepResult out;
  try {
    for (const relay::TokenEvent& e : impl_->engine->step())
      out.events.push_back({e.id, e.token, e.index, e.finish == relay::Finish::None ? 0 : e.finish == relay::Finish::Length ? 1 : 2});
  } catch (const std::exception& e) {
    out.error = e.what();
  }
  return out;
}

bool LocalWorker::cancel(uint64_t request) { return impl_->engine->cancel(request); }

bool LocalWorker::hasWork() const { return impl_->engine->has_work(); }

ModelInfo LocalWorker::info() const {
  const relay::ModelConfig& c = impl_->weights.config;
  ModelInfo m;
  m.modelType = c.model_type;
  m.vocab = c.vocab;
  m.layers = c.layers;
  m.hidden = c.hidden;
  m.maxPosition = c.max_position;
  m.eosIds.assign(c.eos_ids.begin(), c.eos_ids.end());
  return m;
}

int32_t LocalWorker::freeBlocks() const { return impl_->engine->free_blocks(); }

}  // namespace relaybridge

void relaybridge_retain(relaybridge::LocalWorker* w) { w->refs_.fetch_add(1, std::memory_order_relaxed); }

void relaybridge_release(relaybridge::LocalWorker* w) {
  if (w->refs_.fetch_sub(1, std::memory_order_acq_rel) == 1) delete w;
}
