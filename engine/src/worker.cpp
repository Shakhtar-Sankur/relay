#include "relay/worker.h"

#include <chrono>
#include <cstring>
#include <stdexcept>

namespace relay {

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
  step_thread_ = std::thread([this] { step_loop(); });
}

Worker::~Worker() {
  stop();
  if (step_thread_.joinable()) step_thread_.join();
  if (kv_thread_.joinable()) kv_thread_.join();
  std::lock_guard<std::mutex> lock(recv_mu_);
  receivers_.clear();  // closes the connections and joins the receiver threads
}

void Worker::stop() {
  if (stopping_.exchange(true)) return;
  cmd_cv_.notify_all();
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
  }
  return l;
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
  } catch (...) {
    // The control plane went away. Its requests cannot be reported to anyone: drop them.
  }
  {
    std::lock_guard<std::mutex> lock(out_mu_);
    out_ = nullptr;
  }
}

void Worker::apply(const Command& cmd) {
  try {
    switch (cmd.type) {
      case Ctl::Submit: {
        Request r = decode_request(cmd.request, cmd.payload, 0, nullptr);
        if (prefill_) {
          auto it = peers_.find(cmd.a);
          if (it == peers_.end()) throw std::runtime_error("no decode worker " + std::to_string(cmd.a));
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
        std::string addr(cmd.payload.begin(), cmd.payload.end());
        auto colon = addr.rfind(':');
        auto conn = tcp_connect(addr.substr(0, colon), std::stoi(addr.substr(colon + 1)));
        auto sender = std::make_unique<KVSender>(be_, *conn);
        peers_[cmd.a] = {std::move(conn), std::move(sender)};
        break;
      }
      default:
        throw std::runtime_error("unexpected control frame " + std::to_string(static_cast<int>(cmd.type)));
    }
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
    for (const Command& c : cmds) apply(c);
    std::vector<TokenEvent> events;
    try {
      if (engine_ && engine_->has_work()) events = engine_->step();
      if (prefill_ && prefill_->has_work()) events = prefill_->step();
    } catch (const std::exception& e) {
      std::string msg = std::string("worker step failed: ") + e.what();
      send(Ctl::Error, 0, 0, 0, 0, msg.data(), msg.size());
    }
    for (const TokenEvent& e : events)
      send(Ctl::Token, e.id, static_cast<std::uint32_t>(e.token), static_cast<std::uint32_t>(e.index),
           static_cast<std::uint16_t>(e.finish), nullptr, 0);
    auto now = std::chrono::steady_clock::now();
    if (now - last_load > std::chrono::milliseconds(20)) {
      WorkerLoad l = load();
      send(Ctl::Load, 0, 0, 0, 0, &l, sizeof l);
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
