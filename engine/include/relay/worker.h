// A relay worker: one model on one backend, driven by the control plane over a TCP
// connection. Roles:
//   both     prefill and decode in one engine (colocated serving)
//   prefill  prompts only: samples the first token and streams the KV cache to the decode
//            worker the control plane chose for the request
//   decode   accepts KV transfers on its KV port and generates the remaining tokens
//
// Control protocol: frames with the KV transfer's 32-byte header layout (magic "RLYC").
//   worker -> plane  Hello   payload WorkerHello, once per connection
//                    Token   request, a = token, b = index, flags = 0 / 1 length / 2 stop
//                    Error   request, payload: message (the request is dropped here);
//                            flags = kErrorRetryable when another worker can still serve
//                            it (its KV cache did not reach the decode worker)
//                    Load    payload WorkerLoad, about every 20 ms (the plane's heartbeat)
//   plane -> worker  Submit  request, a = decode peer (prefill role) or kNoPeer,
//                            payload BeginInfo + prompt
//                    Resume  request, b = generated count, payload BeginInfo + prompt + generated
//                    Cancel  request
//                    Peer    a = peer index, payload "host:port" of a decode worker's KV port
//                            (replacing any earlier one); an empty payload forgets the peer
// Request ids are chosen by the control plane and unique across the cluster.
//
// A worker serves one control connection at a time. When a new one starts (the control
// plane reconnected after losing this worker, or restarted), the worker drops every
// request it holds: the plane has already moved them elsewhere.
#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "relay/backend.h"
#include "relay/engine.h"
#include "relay/kv_transfer.h"
#include "relay/transport.h"

namespace relay {

enum class Ctl : std::uint16_t { Hello = 1, Token = 2, Error = 3, Load = 4, Submit = 5, Resume = 6, Cancel = 7, Peer = 8 };
constexpr std::uint32_t kCtlMagic = 0x43594c52u;  // "RLYC"
constexpr std::uint32_t kNoPeer = 0xffffffffu;
constexpr std::uint16_t kErrorRetryable = 1;

enum class WorkerRole : std::uint32_t { Both = 0, Prefill = 1, Decode = 2 };

struct WorkerHello {
  std::uint32_t role;
  std::int32_t num_blocks, block_size, vocab, layers, kv_port;
};
static_assert(sizeof(WorkerHello) == 24);

struct WorkerLoad {
  std::int32_t free_blocks, total_blocks, running, waiting;
  std::uint64_t prompt_tokens, prefix_hit_tokens, forward_tokens, steps;
};
static_assert(sizeof(WorkerLoad) == 48);

struct WorkerOptions {
  WorkerRole role = WorkerRole::Both;
  std::string host = "127.0.0.1";
  int control_port = 0;  // 0: any free port
  int kv_port = 0;
  int max_batch_tokens = 512;
  int max_seqs = 64;
  bool prefix_caching = true;
  // Testing aid: sleep this long before every step, so a test's fault lands mid-request.
  int step_delay_ms = 0;
};

class Worker {
 public:
  Worker(Backend& backend, WorkerOptions options);
  ~Worker();
  int control_port() const { return control_->port(); }
  int kv_port() const { return kv_ ? kv_->port() : 0; }
  // Serves control connections one at a time until stop().
  void serve();
  void stop();

 private:
  struct Command {
    Ctl type;  // or kReset, queued when a control connection starts
    std::uint64_t request;
    std::uint32_t a, b;
    std::vector<unsigned char> payload;
  };
  void session(Connection& c);
  void step_loop();
  void apply(const Command& cmd);
  void send(Ctl type, std::uint64_t request, std::uint32_t a, std::uint32_t b, std::uint16_t flags,
            const void* payload, std::size_t n);
  void kv_accept_loop();
  void heartbeat_loop();
  void forget_peer(std::uint32_t index);
  void connect_peer(std::uint32_t index, const std::string& addr);
  WorkerLoad load() const;

  Backend& be_;
  WorkerOptions opt_;
  std::unique_ptr<TcpListener> control_, kv_;
  std::unique_ptr<Engine> engine_;          // both, decode
  std::unique_ptr<PrefillWorker> prefill_;  // prefill
  std::map<std::uint32_t, std::pair<std::unique_ptr<Connection>, std::unique_ptr<KVSender>>> peers_;
  // Step thread only: each peer's address, and when we last tried to reconnect to it.
  std::map<std::uint32_t, std::string> peer_addr_;
  std::map<std::uint32_t, std::chrono::steady_clock::time_point> peer_retry_;
  std::vector<std::pair<std::unique_ptr<Connection>, std::unique_ptr<KVReceiver>>> receivers_;
  std::mutex recv_mu_;

  std::mutex cmd_mu_;
  std::condition_variable cmd_cv_, stop_cv_;
  std::deque<Command> commands_;
  // False from the moment a control connection starts until the step thread has dropped
  // the previous session's requests: until then their tokens and errors are not sent, so
  // none reaches the new control plane. Hello and Load go out at once.
  std::atomic<bool> session_clean_{false};
  std::mutex out_mu_;
  Connection* out_ = nullptr;
  std::atomic<bool> stopping_{false};
  // The latest load, refreshed after every step and sent every 20 ms by its own thread:
  // the control plane's heartbeat must not stop while a long step runs.
  mutable std::mutex load_mu_;
  WorkerLoad load_{};
  std::uint64_t steps_ = 0;
  std::thread step_thread_, kv_thread_, heartbeat_thread_;
};

std::string role_name(WorkerRole r);

}  // namespace relay
