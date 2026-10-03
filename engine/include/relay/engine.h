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
// In the cluster a worker runs one of these; the Swift control plane decides which
// requests it gets. A decode worker also receives requests whose prompt another worker
// already processed (add_prefilled), with their KV cache arriving through the KV
// transfer engine. The public methods are thread-safe: the transfer engine's receiver
// thread reserves blocks and adds requests while another thread calls step().
#pragma once

#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "relay/backend.h"
#include "relay/kv_cache.h"
#include "relay/sampler.h"

namespace relay {

struct EngineOptions {
  int max_batch_tokens = 512;  // tokens per forward pass (decode tokens + prefill chunks)
  int max_seqs = 64;           // requests running at once
  bool prefix_caching = true;  // reuse the KV blocks of prompt prefixes seen before
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
  std::uint64_t prompt_tokens = 0;      // prompt tokens of admitted requests
  std::uint64_t prefix_hit_tokens = 0;  // of those, found in the prefix cache
  int cached_blocks = 0;
};

class Engine {
 public:
  Engine(Backend& backend, EngineOptions options);
  ~Engine();

  // Throws std::invalid_argument if the prompt could never fit in the cache.
  void add(Request request);
  bool has_work() const;
  std::vector<TokenEvent> step();
  // Runs until every request has finished; returns the generated tokens per request.
  std::map<std::uint64_t, std::vector<int>> run_all();

  // Takes n free blocks out of the pool for a request that is being transferred in;
  // false (and nothing taken) if fewer than n are free.
  bool reserve_blocks(int n, std::vector<int>& out);
  // The transfer engine's reservation: succeeds only if nothing is queued and, after it,
  // every running request still has a free block to grow into. Under memory pressure
  // the receiver therefore falls back to recompute instead of starving running requests.
  bool try_reserve_for_transfer(int n, std::vector<int>& out);
  void release_blocks(std::vector<int>& blocks);
  // Adds a request whose prompt was processed elsewhere: `blocks` (from reserve_blocks)
  // hold the KV cache of every prompt token, and `first_token` is the token the prefill
  // worker sampled (index 0). The request continues with token index 1.
  void add_prefilled(Request request, std::vector<int> blocks, int first_token);
  // Adds a request whose first token was sampled elsewhere but whose KV cache did not
  // come with it: the prompt is recomputed here (as after a preemption), then decoding
  // continues with token index 1. Same tokens either way: sampling is deterministic.
  void add_recompute(Request request, int first_token);
  // Drops a request wherever it is (queued or running) and frees its blocks; false if
  // the engine does not have it (already finished, or never added).
  bool cancel(std::uint64_t id);

  EngineStats stats() const;
  int free_blocks() const;

  // Called with the logits each generated token was sampled from (tests use it to
  // compare runs bit for bit).
  std::function<void(std::uint64_t id, int index, const float* logits)> on_logits;

 private:
  struct Seq;
  std::vector<TokenEvent> step_locked();
  bool has_work_locked() const { return !waiting_.empty() || !running_.empty(); }
  bool ensure_blocks(Seq& s, int tokens);
  void preempt(Seq& s);
  Seq* youngest_running();

  mutable std::mutex mu_;
  Backend& backend_;
  EngineOptions opt_;
  BlockManager alloc_;
  std::deque<std::unique_ptr<Seq>> waiting_;
  std::vector<std::unique_ptr<Seq>> running_;
  std::uint64_t admitted_ = 0;
  EngineStats stats_;
};

}  // namespace relay
