#include "relay/engine.h"

#include <algorithm>
#include <stdexcept>

namespace relay {

struct Engine::Seq {
  Request req;
  std::vector<int> tokens;  // prompt, then generated tokens
  int generated = 0;
  int computed = 0;         // tokens whose K and V are in the cache
  std::vector<int> blocks;
  std::vector<std::uint64_t> hashes;  // prefix-cache hashes of the full blocks indexed so far
  std::uint64_t admit_order = 0;
  bool done = false;
};

Engine::Engine(Backend& backend, EngineOptions options)
    : backend_(backend),
      opt_(options),
      alloc_(backend.kv_layout().num_blocks, backend.kv_layout().block_size, options.prefix_caching) {
  if (opt_.max_batch_tokens <= 0 || opt_.max_seqs <= 0) throw std::invalid_argument("bad EngineOptions");
}

Engine::~Engine() = default;

bool Engine::has_work() const {
  std::lock_guard<std::mutex> lock(mu_);
  return has_work_locked();
}

EngineStats Engine::stats() const {
  std::lock_guard<std::mutex> lock(mu_);
  EngineStats s = stats_;
  s.cached_blocks = alloc_.num_cached();
  return s;
}

int Engine::free_blocks() const {
  std::lock_guard<std::mutex> lock(mu_);
  return alloc_.num_available();
}

bool Engine::reserve_blocks(int n, std::vector<int>& out) {
  std::lock_guard<std::mutex> lock(mu_);
  return alloc_.allocate(n, out);
}

bool Engine::try_reserve_for_transfer(int n, std::vector<int>& out) {
  std::lock_guard<std::mutex> lock(mu_);
  if (!waiting_.empty()) return false;
  if (alloc_.num_available() < n + static_cast<int>(running_.size())) return false;
  return alloc_.allocate(n, out);
}

void Engine::add_recompute(Request request, int first_token) {
  std::lock_guard<std::mutex> lock(mu_);
  const KVLayout& kv = backend_.kv_layout();
  if (kv.blocks_for(static_cast<int>(request.prompt.size()) + request.params.max_new_tokens) > kv.num_blocks)
    throw std::invalid_argument("request needs more KV blocks than the cache has");
  auto s = std::make_unique<Seq>();
  s->tokens = request.prompt;
  s->tokens.push_back(first_token);
  s->generated = 1;
  s->req = std::move(request);
  waiting_.push_back(std::move(s));
}

bool Engine::cancel(std::uint64_t id) {
  std::lock_guard<std::mutex> lock(mu_);
  for (auto it = waiting_.begin(); it != waiting_.end(); ++it)
    if ((*it)->req.id == id) {
      alloc_.release_all((*it)->blocks);
      waiting_.erase(it);
      return true;
    }
  for (auto it = running_.begin(); it != running_.end(); ++it)
    if ((*it)->req.id == id) {
      alloc_.release_all((*it)->blocks);
      running_.erase(it);
      return true;
    }
  return false;
}

void Engine::release_blocks(std::vector<int>& blocks) {
  std::lock_guard<std::mutex> lock(mu_);
  alloc_.release_all(blocks);
}

void Engine::add_prefilled(Request request, std::vector<int> blocks, int first_token) {
  std::lock_guard<std::mutex> lock(mu_);
  const KVLayout& kv = backend_.kv_layout();
  if (static_cast<int>(blocks.size()) < kv.blocks_for(static_cast<int>(request.prompt.size())))
    throw std::invalid_argument("add_prefilled: blocks do not cover the prompt");
  auto s = std::make_unique<Seq>();
  s->tokens = request.prompt;
  s->tokens.push_back(first_token);
  s->computed = static_cast<int>(request.prompt.size());
  s->generated = 1;
  s->blocks = std::move(blocks);
  s->req = std::move(request);
  s->admit_order = ++admitted_;
  running_.push_back(std::move(s));
}

void Engine::add(Request request) {
  std::lock_guard<std::mutex> lock(mu_);
  const KVLayout& kv = backend_.kv_layout();
  int longest = static_cast<int>(request.prompt.size()) + request.params.max_new_tokens;
  if (request.prompt.empty()) throw std::invalid_argument("empty prompt");
  if (kv.blocks_for(longest) > kv.num_blocks)
    throw std::invalid_argument("request needs more KV blocks than the cache has");
  auto s = std::make_unique<Seq>();
  s->tokens = request.prompt;
  s->req = std::move(request);
  waiting_.push_back(std::move(s));
}

bool Engine::ensure_blocks(Seq& s, int tokens) {
  int need = backend_.kv_layout().blocks_for(tokens) - static_cast<int>(s.blocks.size());
  return need <= 0 || alloc_.allocate(need, s.blocks);
}

void Engine::preempt(Seq& s) {
  // Its full blocks stay in the prefix cache (unless evicted), so the recompute is
  // usually just a lookup.
  alloc_.release_all(s.blocks);
  s.hashes.clear();
  s.computed = 0;
  ++stats_.preemptions;
  for (auto it = running_.begin(); it != running_.end(); ++it) {
    if (it->get() == &s) {
      waiting_.push_front(std::move(*it));
      running_.erase(it);
      return;
    }
  }
}

Engine::Seq* Engine::youngest_running() {
  Seq* y = nullptr;
  for (auto& s : running_)
    if (!y || s->admit_order > y->admit_order) y = s.get();
  return y;
}

std::vector<TokenEvent> Engine::step() {
  std::lock_guard<std::mutex> lock(mu_);
  return step_locked();
}

std::vector<TokenEvent> Engine::step_locked() {
  ForwardBatch batch;
  std::vector<Seq*> members;
  int budget = opt_.max_batch_tokens;

  // 1. Requests that are generating: one token each, oldest first. If the cache is
  //    full, free room by preempting the youngest request (possibly this one).
  std::vector<Seq*> order;
  for (auto& s : running_) order.push_back(s.get());
  std::sort(order.begin(), order.end(), [](Seq* a, Seq* b) { return a->admit_order < b->admit_order; });
  std::vector<Seq*> preempted;
  for (Seq* s : order) {
    if (std::find(preempted.begin(), preempted.end(), s) != preempted.end()) continue;
    int pending = static_cast<int>(s->tokens.size()) - s->computed;
    if (pending != 1 || budget == 0) continue;
    bool ok = true;
    while (!ensure_blocks(*s, static_cast<int>(s->tokens.size()))) {
      Seq* victim = youngest_running();
      preempted.push_back(victim);
      preempt(*victim);
      if (victim == s) {
        ok = false;
        break;
      }
    }
    if (!ok) continue;
    batch.seqs.push_back({{s->tokens.back()}, s->computed, s->blocks, true});
    members.push_back(s);
    --budget;
  }

  // 2. Prompt chunks: first requests already part-way through their prompt, then
  //    new ones from the queue, while there is budget and cache room.
  auto add_chunk = [&](Seq* s) -> bool {
    bool matched_now = false;
    if (s->computed == 0 && s->blocks.empty()) {
      // Reuse cached prefix blocks. At least one token is left to compute: its logits
      // are what the next token is sampled from.
      s->computed = alloc_.match_prefix(s->tokens, static_cast<int>(s->tokens.size()) - 1, s->blocks, s->hashes);
      matched_now = true;
    }
    int pending = static_cast<int>(s->tokens.size()) - s->computed;
    int n = std::min(pending, budget);
    if (n <= 0 || !ensure_blocks(*s, s->computed + n)) {
      if (matched_now) {  // not admitted after all: a waiting request holds no blocks
        alloc_.release_all(s->blocks);
        s->hashes.clear();
        s->computed = 0;
      }
      return false;
    }
    if (matched_now) {
      stats_.prompt_tokens += s->tokens.size();
      stats_.prefix_hit_tokens += static_cast<std::uint64_t>(s->computed);
    }
    std::vector<int> chunk(s->tokens.begin() + s->computed, s->tokens.begin() + s->computed + n);
    batch.seqs.push_back({std::move(chunk), s->computed, s->blocks, s->computed + n == static_cast<int>(s->tokens.size())});
    members.push_back(s);
    budget -= n;
    return true;
  };
  for (Seq* s : order) {
    if (budget == 0) break;
    if (std::find(preempted.begin(), preempted.end(), s) != preempted.end()) continue;
    if (static_cast<int>(s->tokens.size()) - s->computed > 1) add_chunk(s);
  }
  // No new admissions in a step that had to preempt: the cache is under pressure,
  // and re-admitting what was just preempted would only thrash.
  while (preempted.empty() && budget > 0 && !waiting_.empty() && static_cast<int>(running_.size()) < opt_.max_seqs) {
    Seq* s = waiting_.front().get();
    if (!add_chunk(s)) break;
    s->admit_order = ++admitted_;
    running_.push_back(std::move(waiting_.front()));
    waiting_.pop_front();
  }

  std::vector<TokenEvent> events;
  if (batch.seqs.empty()) {
    if (has_work_locked() && running_.empty() && preempted.empty())
      throw std::runtime_error("engine stalled: the next request does not fit in the KV cache");
    return events;
  }

  std::vector<float> logits = backend_.forward(batch);
  ++stats_.steps;
  const int V = backend_.config().vocab;
  int row = 0;
  for (std::size_t i = 0; i < members.size(); ++i) {
    Seq* s = members[i];
    s->computed += static_cast<int>(batch.seqs[i].tokens.size());
    stats_.forward_tokens += batch.seqs[i].tokens.size();
    alloc_.register_full(s->tokens, s->computed, s->blocks, s->hashes);
    if (!batch.seqs[i].want_logits) continue;
    const float* lg = logits.data() + static_cast<std::size_t>(row++) * V;
    if (on_logits) on_logits(s->req.id, s->generated, lg);
    int tok = sample(lg, V, s->req.params, static_cast<std::uint64_t>(s->generated));
    int index = s->generated++;
    s->tokens.push_back(tok);
    Finish f = Finish::None;
    const auto& eos = backend_.config().eos_ids;
    if (!s->req.params.ignore_eos && std::find(eos.begin(), eos.end(), tok) != eos.end()) f = Finish::Stop;
    else if (s->generated >= s->req.params.max_new_tokens) f = Finish::Length;
    events.push_back({s->req.id, tok, index, f});
    if (f != Finish::None) s->done = true;
  }

  for (auto it = running_.begin(); it != running_.end();) {
    if ((*it)->done) {
      alloc_.release_all((*it)->blocks);
      it = running_.erase(it);
    } else {
      ++it;
    }
  }
  return events;
}

std::map<std::uint64_t, std::vector<int>> Engine::run_all() {
  std::map<std::uint64_t, std::vector<int>> out;
  while (has_work())
    for (const TokenEvent& e : step()) out[e.id].push_back(e.token);
  return out;
}

}  // namespace relay
