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
//                    Error   request, payload: message (the request is dropped)
//                    Load    payload WorkerLoad, about every 20 ms
//   plane -> worker  Submit  request, a = decode peer (prefill role) or kNoPeer,
//                            payload BeginInfo + prompt
//                    Resume  request, b = generated count, payload BeginInfo + prompt + generated
//                    Cancel  request
//                    Peer    a = peer index, payload "host:port" of a decode worker's KV port
// Request ids are chosen by the control plane and unique across the cluster.
#pragma once

#include <atomic>
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
    Ctl type;
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
  WorkerLoad load() const;

  Backend& be_;
  WorkerOptions opt_;
  std::unique_ptr<TcpListener> control_, kv_;
  std::unique_ptr<Engine> engine_;          // both, decode
  std::unique_ptr<PrefillWorker> prefill_;  // prefill
  std::map<std::uint32_t, std::pair<std::unique_ptr<Connection>, std::unique_ptr<KVSender>>> peers_;
  std::vector<std::pair<std::unique_ptr<Connection>, std::unique_ptr<KVReceiver>>> receivers_;
  std::mutex recv_mu_;

  std::mutex cmd_mu_;
  std::condition_variable cmd_cv_;
  std::deque<Command> commands_;
  std::mutex out_mu_;
  Connection* out_ = nullptr;
  std::atomic<bool> stopping_{false};
  std::thread step_thread_, kv_thread_;
};

std::string role_name(WorkerRole r);

}  // namespace relay
