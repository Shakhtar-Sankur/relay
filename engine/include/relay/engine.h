// The worker-local engine: continuous batching over one backend.
//
// Every step() builds one batch from the requests it holds and runs one forward
// pass. Requests that are generating contribute their newest token; requests that
// still have prompt to process contribute a chunk of it (chunked prefill), up to
// a token budget per step. Requests join as soon as there is room and leave the
// moment they finish, so the batch changes every step.
//
// When the KV cache runs out of blocks, the most recently admitted request is
// preempted: its blocks are freed and it goes back to the front of the queue, to be
// recomputed later from its prompt and the tokens it has generated so far. Because
// sampling depends only on (seed, token index), it then produces the same tokens.
//
// In the cluster (M3-M5) a worker runs one of these; the Swift control plane decides
// which requests it gets.
#pragma once

#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "relay/backend.h"
#include "relay/kv_cache.h"
#include "relay/sampler.h"

namespace relay {

struct EngineOptions {
  int max_batch_tokens = 512;  // tokens per forward pass (decode tokens + prefill chunks)
  int max_seqs = 64;           // requests running at once
};

struct Request {
  std::uint64_t id = 0;
  std::vector<int> prompt;
  SamplingParams params;
};

enum class Finish { None, Length, Stop };

struct TokenEvent {
  std::uint64_t id;
  int token;
  int index;  // 0 for the first generated token
  Finish finish;
};

struct EngineStats {
  std::uint64_t steps = 0;
  std::uint64_t forward_tokens = 0;
  std::uint64_t preemptions = 0;
};

class Engine {
 public:
  Engine(Backend& backend, EngineOptions options);
  ~Engine();

  // Throws std::invalid_argument if the prompt could never fit in the cache.
  void add(Request request);
  bool has_work() const { return !waiting_.empty() || !running_.empty(); }
  std::vector<TokenEvent> step();
  // Runs until every request has finished; returns the generated tokens per request.
  std::map<std::uint64_t, std::vector<int>> run_all();

  const EngineStats& stats() const { return stats_; }
  int free_blocks() const { return alloc_.num_free(); }

  // Called with the logits each generated token was sampled from (tests use it to
  // compare runs bit for bit).
  std::function<void(std::uint64_t id, int index, const float* logits)> on_logits;

 private:
  struct Seq;
  bool ensure_blocks(Seq& s, int tokens);
  void preempt(Seq& s);
  Seq* youngest_running();

  Backend& backend_;
  EngineOptions opt_;
  BlockAllocator alloc_;
  std::deque<std::unique_ptr<Seq>> waiting_;
  std::vector<std::unique_ptr<Seq>> running_;
  std::uint64_t admitted_ = 0;
  EngineStats stats_;
};

}  // namespace relay
