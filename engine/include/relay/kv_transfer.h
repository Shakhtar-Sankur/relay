// The KV transfer engine: moves a request's KV cache from the worker that processed its
// prompt (prefill) to the worker that will generate its tokens (decode).
//
// The prefill side streams the cache layer by layer. The backend reports each layer as
// soon as its K and V are written (LayerObserver), and a sender thread puts that layer's
// blocks on the wire while the backend computes the next layer. By the time the forward
// pass ends, most of the cache is already on its way; only the last layers' transfer is
// left on the critical path.
//
// Wire format, on any Connection: frames of a 32-byte header and a payload.
//   Hello  (once per connection)  payload: KVFingerprint; the receiver checks it matches
//                                  its own cache layout and dtype.
//   Begin  request                 payload: BeginInfo + prompt token ids.
//   Layer  request, a = layer, b = index in the request's block list of the first block
//                                  payload: K of each block, then V of each block.
//   End    request, a = first generated token, b = 1 if the request already finished
//   Abort  request                 the receiver drops the request.
// Block ids never cross the wire: each side maps "the request's i-th block" to its own.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <exception>
#include <functional>
#include <map>
#include <mutex>
#include <thread>
#include <vector>

#include "relay/backend.h"
#include "relay/engine.h"
#include "relay/transport.h"

namespace relay {

enum class FrameType : std::uint16_t { Hello = 1, Begin = 2, Layer = 3, End = 4, Abort = 5 };

struct FrameHeader {
  std::uint32_t magic;
  std::uint16_t type;
  std::uint16_t flags;
  std::uint64_t request;
  std::uint32_t a;
  std::uint32_t b;
  std::uint64_t payload;
};
static_assert(sizeof(FrameHeader) == 32);
constexpr std::uint32_t kFrameMagic = 0x4b56524cu;  // "LRVK"

struct KVFingerprint {
  std::int32_t layers, kv_heads, head_dim, block_size, block_bytes, vocab;
  static KVFingerprint of(const Backend& b);
  bool operator==(const KVFingerprint&) const = default;
};

struct BeginInfo {
  std::uint32_t prompt_len;
  std::uint32_t num_blocks;
  std::int32_t max_new_tokens;
  float temperature;
  float top_p;
  std::int32_t top_k;
  std::uint64_t seed;
  std::uint32_t ignore_eos;
  std::uint32_t reserved;
};
static_assert(sizeof(BeginInfo) == 40);

struct TransferStats {
  std::uint64_t frames = 0;
  std::uint64_t kv_bytes = 0;
  std::uint64_t requests = 0;
  std::uint64_t recomputed = 0;  // receiver had no room: KV discarded, prompt recomputed
};

// Prefill side: queues frames and sends them from its own thread.
class KVSender {
 public:
  KVSender(Backend& backend, Connection& conn);
  ~KVSender();  // sends what is queued, then stops

  void begin(const Request& r, int num_blocks);
  // Sends layer `layer` of `blocks` (backend block ids), which are the request's blocks
  // number first_index, first_index + 1, ... The blocks must stay allocated until the
  // request's end() callback has run.
  void layer(std::uint64_t request, int layer, int first_index, std::vector<int> blocks);
  // on_sent runs on the sender thread once every earlier frame of the request is out.
  void end(std::uint64_t request, int first_token, bool finished, std::function<void()> on_sent);
  void abort(std::uint64_t request);
  // Waits until everything queued so far has been sent; rethrows a send error.
  void flush();

  TransferStats stats() const;
  // Called on the sender thread after each Layer frame is written: (request, layer).
  std::function<void(std::uint64_t, int)> on_layer_sent;

 private:
  struct Job {
    FrameType type;
    std::uint64_t request = 0;
    int a = 0, b = 0;
    std::vector<int> blocks;
    std::vector<unsigned char> payload;
    std::function<void()> on_sent;
  };
  void push(Job j);
  void run();
  void send_layer(Job& j);

  Backend& be_;
  Connection& conn_;
  mutable std::mutex mu_;
  std::condition_variable cv_, idle_cv_;
  std::deque<Job> queue_;
  bool busy_ = false, stop_ = false;
  std::exception_ptr error_;
  TransferStats stats_;
  std::vector<unsigned char> staging_;
  std::thread thread_;
};

// Decode side: reads frames from one connection on its own thread, reserves blocks in
// the engine for each incoming request, writes the arriving layers straight into them,
// and hands the request to the engine (add_prefilled) when its End frame arrives.
// If the connection breaks, every request still in flight gives its blocks back.
//
// The receiver never waits for blocks. Frames of several requests share one ordered
// stream, so waiting for room for request B would also hold up the End of request A,
// whose blocks are what B is waiting for. When there is no room, the request's KV frames
// are read and dropped and the engine recomputes its prompt (add_recompute).
class KVReceiver {
 public:
  KVReceiver(Engine& engine, Backend& backend, Connection& conn);
  ~KVReceiver();  // closes the connection and joins
  void join();    // waits for the peer to close the connection

  TransferStats stats() const;
  std::exception_ptr error() const;
  int in_flight() const;
  // Called on the receiver thread when a request is complete and handed to the engine.
  std::function<void(std::uint64_t request, int first_token)> on_ready;

 private:
  struct Incoming {
    Request req;
    std::vector<int> blocks;  // empty: no room, the KV cache is being discarded
  };
  void run();
  void handle(const FrameHeader& h);

  Engine& engine_;
  Backend& be_;
  Connection& conn_;
  mutable std::mutex mu_;
  std::map<std::uint64_t, Incoming> incoming_;
  TransferStats stats_;
  std::exception_ptr error_;
  std::vector<unsigned char> staging_;
  std::thread thread_;
};

// A worker that only processes prompts. Each step runs one forward pass over prompt
// chunks (up to max_batch_tokens), streams every layer's KV blocks to the decode side as
// the layer completes, samples each finished prompt's first token, and sends End.
// The request's blocks are freed once its last frame is on the wire.
class PrefillWorker {
 public:
  PrefillWorker(Backend& backend, KVSender& sender, int max_batch_tokens);
  void add(Request r);
  bool has_work() const;
  std::vector<TokenEvent> step();
  int free_blocks() const;
  // Called with the logits each request's first token was sampled from.
  std::function<void(std::uint64_t id, const float* logits)> on_logits;
  // false: send every layer after the forward pass instead of during it (the baseline
  // the benchmark compares streaming against).
  bool stream_layers = true;

 private:
  struct Pending {
    Request req;
    std::vector<int> blocks;
    int computed = 0;
    bool begun = false;
    bool done = false;
  };
  Backend& be_;
  KVSender& sender_;
  int budget_;
  mutable std::mutex alloc_mu_;  // the sender thread frees blocks
  BlockAllocator alloc_;
  std::deque<Pending> queue_;
};

}  // namespace relay
