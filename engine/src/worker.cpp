#include "relay/worker.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace relay {

namespace {

constexpr Ctl kReset = static_cast<Ctl>(0x100);  // never on the wire

// An error another worker can recover from: the control plane retries the request.
struct RetryableError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

}  // namespace

std::string role_name(WorkerRole r) {
  switch (r) {
    case WorkerRole::Both: return "both";
    case WorkerRole::Prefill: return "prefill";
    case WorkerRole::Decode: return "decode";
  }
  return "?";
}

Worker::Worker(Backend& backend, WorkerOptions options) : be_(backend), opt_(std::move(options)) {
  control_ = std::make_unique<TcpListener>(opt_.control_port, opt_.host);
  if (opt_.role == WorkerRole::Prefill) {
    prefill_ = std::make_unique<PrefillWorker>(be_, nullptr, opt_.max_batch_tokens, opt_.prefix_caching);
  } else {
    engine_ = std::make_unique<Engine>(be_, EngineOptions{opt_.max_batch_tokens, opt_.max_seqs, opt_.prefix_caching});
  }
  if (opt_.role == WorkerRole::Decode) {
    kv_ = std::make_unique<TcpListener>(opt_.kv_port, opt_.host);
    kv_thread_ = std::thread([this] { kv_accept_loop(); });
  }
  load_ = load();
  step_thread_ = std::thread([this] { step_loop(); });
  heartbeat_thread_ = std::thread([this] { heartbeat_loop(); });
}

Worker::~Worker() {
  stop();
  if (step_thread_.joinable()) step_thread_.join();
  if (heartbeat_thread_.joinable()) heartbeat_thread_.join();
  if (kv_thread_.joinable()) kv_thread_.join();
  std::lock_guard<std::mutex> lock(recv_mu_);
  receivers_.clear();  // closes the connections and joins the receiver threads
}

void Worker::stop() {
  if (stopping_.exchange(true)) return;
  {
    std::lock_guard<std::mutex> lock(cmd_mu_);
    cmd_cv_.notify_all();
    stop_cv_.notify_all();
  }
  {
    std::lock_guard<std::mutex> lock(out_mu_);
    if (out_) out_->close();  // ends the session blocked in recv()
  }
  // Unblock accept() calls by connecting to ourselves; the loops see stopping_ and exit.
  try {
    tcp_connect(opt_.host, control_->port(), 1.0);
  } catch (...) {
  }
  if (kv_) {
    try {
      tcp_connect(opt_.host, kv_->port(), 1.0);
    } catch (...) {
    }
  }
}

void Worker::send(Ctl type, std::uint64_t request, std::uint32_t a, std::uint32_t b, std::uint16_t flags,
                  const void* payload, std::size_t n) {
  // Results of requests from an earlier session stay with the worker (see session()).
  if ((type == Ctl::Token || type == Ctl::Error) && !session_clean_) return;
  FrameHeader h{kCtlMagic, static_cast<std::uint16_t>(type), flags, request, a, b, n};
  Slice s[2] = {{&h, sizeof h}, {payload, n}};
  std::lock_guard<std::mutex> lock(out_mu_);
  if (!out_) return;  // no control plane connected: nothing to report to
  try {
    out_->send(s, 2);
  } catch (const ConnectionClosed&) {
    out_ = nullptr;
  }
}

WorkerLoad Worker::load() const {
  WorkerLoad l{};
  const KVLayout& kv = be_.kv_layout();
  l.total_blocks = kv.num_blocks;
  if (engine_) {
    EngineStats s = engine_->stats();
    l.free_blocks = engine_->free_blocks();
    l.prompt_tokens = s.prompt_tokens;
    l.prefix_hit_tokens = s.prefix_hit_tokens;
    l.forward_tokens = s.forward_tokens;
    l.steps = s.steps;
  } else {
    PrefillStats s = prefill_->stats();
    l.free_blocks = prefill_->free_blocks();
    l.prompt_tokens = s.prompt_tokens;
    l.prefix_hit_tokens = s.prefix_hit_tokens;
    l.forward_tokens = s.forward_tokens;
    l.steps = steps_;
  }
  return l;
}

void Worker::heartbeat_loop() {
  while (!stopping_) {
    WorkerLoad l;
    {
      std::lock_guard<std::mutex> lock(load_mu_);
      l = load_;
    }
    send(Ctl::Load, 0, 0, 0, 0, &l, sizeof l);
    std::unique_lock<std::mutex> lock(cmd_mu_);
    stop_cv_.wait_for(lock, std::chrono::milliseconds(20), [this] { return stopping_.load(); });
  }
}

void Worker::serve() {
  while (!stopping_) {
    std::unique_ptr<Connection> c;
    try {
      c = control_->accept();
    } catch (...) {
      return;
    }
    if (stopping_) return;
    session(*c);
  }
}

static Request decode_request(std::uint64_t id, const std::vector<unsigned char>& payload, std::uint32_t generated,
                              std::vector<int>* gen_out) {
  BeginInfo info;
  if (payload.size() < sizeof info) throw std::runtime_error("short Submit");
  std::memcpy(&info, payload.data(), sizeof info);
  if (payload.size() != sizeof info + (info.prompt_len + generated) * sizeof(std::int32_t))
    throw std::runtime_error("Submit payload size does not match");
  Request r;
  r.id = id;
  r.prompt.resize(info.prompt_len);
  std::memcpy(r.prompt.data(), payload.data() + sizeof info, info.prompt_len * sizeof(std::int32_t));
  if (gen_out) {
    gen_out->resize(generated);
    std::memcpy(gen_out->data(), payload.data() + sizeof info + info.prompt_len * sizeof(std::int32_t),
                generated * sizeof(std::int32_t));
  }
  r.params.max_new_tokens = info.max_new_tokens;
  r.params.temperature = info.temperature;
  r.params.top_p = info.top_p;
  r.params.top_k = info.top_k;
  r.params.seed = info.seed;
  r.params.ignore_eos = info.ignore_eos != 0;
  return r;
}

void Worker::session(Connection& c) {
  {
    // Whatever the previous control plane asked for is moot now. The step thread drops it
    // before any command of this session (which all queue behind the Reset); a step that is
    // running meanwhile cannot leak an old request's tokens: they are held back until then.
    // The handshake does not wait for that step, which on a CPU can take seconds.
    std::lock_guard<std::mutex> lock(cmd_mu_);
    session_clean_ = false;
    commands_.clear();
    commands_.push_back(Command{kReset, 0, 0, 0, {}});
  }
  cmd_cv_.notify_one();
  {
    std::lock_guard<std::mutex> lock(out_mu_);
    out_ = &c;
  }
  const KVLayout& kv = be_.kv_layout();
  WorkerHello hello{static_cast<std::uint32_t>(opt_.role), kv.num_blocks, kv.block_size, be_.config().vocab,
                    be_.config().layers, kv_ ? kv_->port() : 0};
  send(Ctl::Hello, 0, 0, 0, 0, &hello, sizeof hello);
  try {
    while (!stopping_) {
      FrameHeader h;
      c.recv(&h, sizeof h);
      if (h.magic != kCtlMagic) throw std::runtime_error("worker: bad control frame");
      if (h.payload > (64u << 20)) throw std::runtime_error("worker: control frame too large");
      Command cmd{static_cast<Ctl>(h.type), h.request, h.a, h.b, std::vector<unsigned char>(h.payload)};
      if (h.payload) c.recv(cmd.payload.data(), h.payload);
      {
        std::lock_guard<std::mutex> lock(cmd_mu_);
        commands_.push_back(std::move(cmd));
      }
      cmd_cv_.notify_one();
    }
  } catch (const std::exception& e) {
    // The control plane went away. Its requests cannot be reported to anyone: drop them.
    if (std::getenv("RELAY_DEBUG")) std::fprintf(stderr, "[worker %s] session ended: %s\n", role_name(opt_.role).c_str(), e.what());
  } catch (...) {
  }
  {
    std::lock_guard<std::mutex> lock(out_mu_);
    out_ = nullptr;
  }
}

void Worker::forget_peer(std::uint32_t index) {
  auto it = peers_.find(index);
  if (it == peers_.end()) return;
  KVSender* sender = it->second.second.get();
  it->second.first->close();  // a send blocked on a stalled peer fails now
  it->second.second.reset();  // joins the sender's thread
  prefill_->drop_sender(sender);
  peers_.erase(it);
}

void Worker::connect_peer(std::uint32_t index, const std::string& addr) {
  forget_peer(index);
  peer_addr_[index] = addr;
  auto colon = addr.rfind(':');
  auto conn = tcp_connect(addr.substr(0, colon), std::stoi(addr.substr(colon + 1)), 2.0);
  auto sender = std::make_unique<KVSender>(be_, *conn);
  peers_[index] = {std::move(conn), std::move(sender)};
}

void Worker::apply(const Command& cmd) {
  if (cmd.type == kReset) {
    if (engine_) engine_->cancel_all();
    if (prefill_) prefill_->cancel_all();
    return;
  }
  try {
    switch (cmd.type) {
      case Ctl::Submit: {
        Request r = decode_request(cmd.request, cmd.payload, 0, nullptr);
        if (prefill_) {
          auto it = peers_.find(cmd.a);
          if (it != peers_.end() && it->second.second->failed()) {
            // The connection broke (the decode worker restarted, or the network did):
            // reconnect, at most once a second, before giving up on the request.
            auto now = std::chrono::steady_clock::now();
            auto& last = peer_retry_[cmd.a];
            if (now - last > std::chrono::seconds(1)) {
              last = now;
              std::string addr = peer_addr_[cmd.a];
              try {
                connect_peer(cmd.a, addr);
              } catch (const std::exception&) {
                forget_peer(cmd.a);
                peer_addr_[cmd.a] = addr;  // keep it for the next attempt
              }
              it = peers_.find(cmd.a);
            }
          }
          if (it == peers_.end() || it->second.second->failed())
            throw RetryableError("no connection to decode worker " + std::to_string(cmd.a));
          prefill_->add(std::move(r), it->second.second.get());
        } else {
          engine_->add(std::move(r));
        }
        break;
      }
      case Ctl::Resume: {
        if (!engine_) throw std::runtime_error("a prefill worker cannot resume a request");
        std::vector<int> gen;
        Request r = decode_request(cmd.request, cmd.payload, cmd.b, &gen);
        engine_->add_resume(std::move(r), gen);
        break;
      }
      case Ctl::Cancel:
        if (engine_) engine_->cancel(cmd.request);
        if (prefill_) prefill_->cancel(cmd.request);
        break;
      case Ctl::Peer: {
        if (!prefill_) throw std::runtime_error("only prefill workers connect to peers");
        forget_peer(cmd.a);
        peer_addr_.erase(cmd.a);
        if (cmd.payload.empty()) break;
        connect_peer(cmd.a, std::string(cmd.payload.begin(), cmd.payload.end()));
        break;
      }
      default:
        throw std::runtime_error("unexpected control frame " + std::to_string(static_cast<int>(cmd.type)));
    }
  } catch (const RetryableError& e) {
    std::string msg = e.what();
    send(Ctl::Error, cmd.request, 0, 0, kErrorRetryable, msg.data(), msg.size());
  } catch (const std::exception& e) {
    std::string msg = e.what();
    send(Ctl::Error, cmd.request, 0, 0, 0, msg.data(), msg.size());
  }
}

void Worker::step_loop() {
  auto last_load = std::chrono::steady_clock::now() - std::chrono::seconds(1);
  while (!stopping_) {
    std::deque<Command> cmds;
    {
      std::unique_lock<std::mutex> lock(cmd_mu_);
      bool busy = (engine_ && engine_->has_work()) || (prefill_ && prefill_->has_work());
      if (!busy && commands_.empty()) cmd_cv_.wait_for(lock, std::chrono::milliseconds(2));
      cmds.swap(commands_);
    }
    for (const Command& c : cmds) {
      apply(c);
      if (c.type == kReset) {
        // Only if no newer session has queued another Reset meanwhile.
        std::lock_guard<std::mutex> lock(cmd_mu_);
        if (std::none_of(commands_.begin(), commands_.end(), [](const Command& x) { return x.type == kReset; }))
          session_clean_ = true;
      }
    }
    if (opt_.step_delay_ms > 0 && ((engine_ && engine_->has_work()) || (prefill_ && prefill_->has_work())))
      std::this_thread::sleep_for(std::chrono::milliseconds(opt_.step_delay_ms));
    std::vector<TokenEvent> events;
    bool stepped = false;
    try {
      if (engine_ && engine_->has_work()) {
        events = engine_->step();
        stepped = true;
      }
      if (prefill_ && prefill_->has_work()) {
        events = prefill_->step();
        stepped = true;
        ++steps_;
      }
    } catch (const std::exception& e) {
      std::string msg = std::string("worker step failed: ") + e.what();
      send(Ctl::Error, 0, 0, 0, 0, msg.data(), msg.size());
    }
    for (const TokenEvent& e : events)
      send(Ctl::Token, e.id, static_cast<std::uint32_t>(e.token), static_cast<std::uint32_t>(e.index),
           static_cast<std::uint16_t>(e.finish), nullptr, 0);
    if (prefill_) {
      static const std::string lost = "the KV cache did not reach the decode worker (connection lost)";
      for (std::uint64_t id : prefill_->take_failed())
        send(Ctl::Error, id, 0, 0, kErrorRetryable, lost.data(), lost.size());
    }
    auto now = std::chrono::steady_clock::now();
    if (stepped || now - last_load > std::chrono::milliseconds(20)) {
      WorkerLoad l = load();
      std::lock_guard<std::mutex> lock(load_mu_);
      load_ = l;
      last_load = now;
    }
  }
}

void Worker::kv_accept_loop() {
  while (!stopping_) {
    std::unique_ptr<Connection> c;
    try {
      c = kv_->accept();
    } catch (...) {
      return;
    }
    if (stopping_) return;
    auto rx = std::make_unique<KVReceiver>(*engine_, be_, *c, [this](std::uint64_t, int) { cmd_cv_.notify_one(); });
    std::lock_guard<std::mutex> lock(recv_mu_);
    receivers_.emplace_back(std::move(c), std::move(rx));
  }
}

}  // namespace relay
