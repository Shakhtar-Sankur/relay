// Workers over the control protocol: a colocated worker, and a prefill worker handing
// requests to a decode worker, must both produce what one engine produces. Each worker
// runs in this process on its own thread and is driven over TCP exactly as the Swift
// control plane drives it.
#include <atomic>
#include <chrono>
#include <cstring>
#include <map>
#include <random>
#include <thread>

#include "check.h"
#include "relay/engine.h"
#include "relay/kv_transfer.h"
#include "relay/worker.h"

using namespace relay;

namespace {

const char* kModel = RELAY_FIXTURES "/llama-gqa-tied-bf16";

struct RunningWorker {
  std::unique_ptr<Backend> be;
  std::unique_ptr<Worker> w;
  std::thread t;
  RunningWorker(const HostWeights& weights, WorkerRole role, int blocks = 64) {
    be = check::make_backend(weights, blocks, 8);
    WorkerOptions o;
    o.role = role;
    w = std::make_unique<Worker>(*be, o);
    t = std::thread([this] { w->serve(); });
  }
  ~RunningWorker() {
    w->stop();
    t.join();
  }
};

// The control plane's side of one worker connection.
struct Client {
  std::unique_ptr<Connection> c;
  std::thread reader;
  std::mutex mu;
  WorkerHello hello{};
  WorkerLoad last_load{};
  std::map<std::uint64_t, std::vector<std::pair<int, int>>> tokens;  // id -> (index, token)
  std::map<std::uint64_t, int> finished;
  std::vector<std::string> errors;

  explicit Client(int port) : c(tcp_connect("127.0.0.1", port)) {
    FrameHeader h;
    c->recv(&h, sizeof h);
    CHECK_EQ(h.type, static_cast<std::uint16_t>(Ctl::Hello));
    c->recv(&hello, sizeof hello);
    reader = std::thread([this] {
      try {
        while (true) {
          FrameHeader h;
          c->recv(&h, sizeof h);
          std::vector<unsigned char> p(h.payload);
          if (h.payload) c->recv(p.data(), h.payload);
          std::lock_guard<std::mutex> lock(mu);
          if (h.type == static_cast<std::uint16_t>(Ctl::Token)) {
            tokens[h.request].push_back({static_cast<int>(h.b), static_cast<int>(h.a)});
            if (h.flags) finished[h.request] = h.flags;
          } else if (h.type == static_cast<std::uint16_t>(Ctl::Load)) {
            std::memcpy(&last_load, p.data(), sizeof last_load);
          } else if (h.type == static_cast<std::uint16_t>(Ctl::Error)) {
            errors.emplace_back(p.begin(), p.end());
          }
        }
      } catch (...) {
      }
    });
  }
  ~Client() {
    c->close();
    reader.join();
  }
  void frame(Ctl t, std::uint64_t id, std::uint32_t a, std::uint32_t b, const std::vector<unsigned char>& p) {
    FrameHeader h{kCtlMagic, static_cast<std::uint16_t>(t), 0, id, a, b, p.size()};
    Slice s[2] = {{&h, sizeof h}, {p.data(), p.size()}};
    c->send(s, 2);
  }
  static std::vector<unsigned char> begin_payload(const Request& r, const std::vector<int>& generated = {}) {
    BeginInfo info{static_cast<std::uint32_t>(r.prompt.size()), 0, r.params.max_new_tokens, r.params.temperature,
                   r.params.top_p, r.params.top_k, r.params.seed, r.params.ignore_eos ? 1u : 0u, 0};
    std::vector<unsigned char> p(sizeof info + (r.prompt.size() + generated.size()) * 4);
    std::memcpy(p.data(), &info, sizeof info);
    std::memcpy(p.data() + sizeof info, r.prompt.data(), r.prompt.size() * 4);
    if (!generated.empty()) std::memcpy(p.data() + sizeof info + r.prompt.size() * 4, generated.data(), generated.size() * 4);
    return p;
  }
  void submit(const Request& r, std::uint32_t peer = kNoPeer) { frame(Ctl::Submit, r.id, peer, 0, begin_payload(r)); }
  void resume(const Request& r, const std::vector<int>& gen) {
    frame(Ctl::Resume, r.id, 0, static_cast<std::uint32_t>(gen.size()), begin_payload(r, gen));
  }
  void peer(std::uint32_t index, int kv_port) {
    std::string a = "127.0.0.1:" + std::to_string(kv_port);
    frame(Ctl::Peer, 0, index, 0, std::vector<unsigned char>(a.begin(), a.end()));
  }
  void cancel(std::uint64_t id) { frame(Ctl::Cancel, id, 0, 0, {}); }
};

std::vector<Request> make_requests(int vocab, int n = 8) {
  std::mt19937 rng(31);
  std::vector<Request> rs;
  for (int i = 0; i < n; ++i) {
    Request r;
    r.id = 1000 + i;
    int len = 1 + static_cast<int>(rng() % 50);
    for (int t = 0; t < len; ++t) r.prompt.push_back(static_cast<int>(rng() % vocab));
    r.params.max_new_tokens = 2 + i % 7;
    r.params.ignore_eos = true;
    if (i % 3 == 0) {
      r.params.temperature = 0.9f;
      r.params.seed = 50 + i;
    }
    rs.push_back(r);
  }
  return rs;
}

std::map<std::uint64_t, std::vector<int>> reference(const HostWeights& w, const std::vector<Request>& rs) {
  auto be = check::make_backend(w, 64, 8);
  Engine e(*be, {512, 64});
  for (const auto& r : rs) e.add(r);
  return e.run_all();
}

// Tokens per request in index order, merged from several workers' clients.
std::map<std::uint64_t, std::vector<int>> merged(std::vector<Client*> clients) {
  std::map<std::uint64_t, std::map<int, int>> by_index;
  for (Client* c : clients) {
    std::lock_guard<std::mutex> lock(c->mu);
    for (auto& [id, v] : c->tokens)
      for (auto [idx, tok] : v) by_index[id][idx] = tok;
  }
  std::map<std::uint64_t, std::vector<int>> out;
  for (auto& [id, m] : by_index)
    for (auto& [idx, tok] : m) out[id].push_back(tok);
  return out;
}

bool wait_finished(std::vector<Client*> clients, std::size_t n, double seconds = 30) {
  auto deadline = std::chrono::steady_clock::now() + std::chrono::duration<double>(seconds);
  while (std::chrono::steady_clock::now() < deadline) {
    std::size_t done = 0;
    for (Client* c : clients) {
      std::lock_guard<std::mutex> lock(c->mu);
      done += c->finished.size();
    }
    if (done >= n) return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  return false;
}

}  // namespace

TEST(a_colocated_worker_equals_one_engine) {
  HostWeights w = HostWeights::load(kModel);
  auto rs = make_requests(w.config.vocab);
  RunningWorker both(w, WorkerRole::Both);
  Client c(both.w->control_port());
  CHECK_EQ(c.hello.role, static_cast<std::uint32_t>(WorkerRole::Both));
  for (const auto& r : rs) c.submit(r);
  CHECK(wait_finished({&c}, rs.size()));
  CHECK(merged({&c}) == reference(w, rs));
  CHECK(c.errors.empty());
}

TEST(prefill_and_decode_workers_equal_one_engine) {
  HostWeights w = HostWeights::load(kModel);
  auto rs = make_requests(w.config.vocab);
  RunningWorker prefill(w, WorkerRole::Prefill), decode(w, WorkerRole::Decode);
  Client p(prefill.w->control_port()), d(decode.w->control_port());
  CHECK(d.hello.kv_port > 0);
  p.peer(0, d.hello.kv_port);
  for (const auto& r : rs) p.submit(r, 0);
  // A request finishes on the prefill worker if it wants one token, else on the decode worker.
  CHECK(wait_finished({&p, &d}, rs.size()));
  CHECK(merged({&p, &d}) == reference(w, rs));
  CHECK(p.errors.empty() && d.errors.empty());
  // The first token of every request came from the prefill worker, the rest from decode.
  std::lock_guard<std::mutex> lock(p.mu);
  for (const auto& r : rs) CHECK(p.tokens[r.id].size() == 1 && p.tokens[r.id][0].first == 0);
}

TEST(one_prefill_worker_feeds_two_decode_workers) {
  HostWeights w = HostWeights::load(kModel);
  auto rs = make_requests(w.config.vocab, 10);
  RunningWorker prefill(w, WorkerRole::Prefill), d1(w, WorkerRole::Decode), d2(w, WorkerRole::Decode);
  Client p(prefill.w->control_port()), c1(d1.w->control_port()), c2(d2.w->control_port());
  p.peer(0, c1.hello.kv_port);
  p.peer(1, c2.hello.kv_port);
  for (std::size_t i = 0; i < rs.size(); ++i) p.submit(rs[i], static_cast<std::uint32_t>(i % 2));
  CHECK(wait_finished({&p, &c1, &c2}, rs.size()));
  CHECK(merged({&p, &c1, &c2}) == reference(w, rs));
}

TEST(resume_continues_a_request_exactly) {
  HostWeights w = HostWeights::load(kModel);
  auto rs = make_requests(w.config.vocab, 6);
  auto want = reference(w, rs);
  RunningWorker decode(w, WorkerRole::Decode);
  Client d(decode.w->control_port());
  // Each request has "already generated" its first half on some worker that is now gone.
  for (const auto& r : rs) {
    auto& all = want[r.id];
    int keep = static_cast<int>(all.size()) / 2;
    if (keep == 0) continue;
    d.resume(r, std::vector<int>(all.begin(), all.begin() + keep));
  }
  std::size_t expected = 0;
  for (const auto& r : rs) expected += want[r.id].size() / 2 > 0 ? 1 : 0;
  CHECK(wait_finished({&d}, expected));
  std::lock_guard<std::mutex> lock(d.mu);
  for (const auto& r : rs) {
    auto& all = want[r.id];
    int keep = static_cast<int>(all.size()) / 2;
    if (keep == 0) continue;
    std::vector<int> rest;
    for (auto [idx, tok] : d.tokens[r.id]) {
      CHECK(idx >= keep);
      rest.push_back(tok);
    }
    CHECK(rest == std::vector<int>(all.begin() + keep, all.end()));
  }
}

TEST(cancel_and_errors_over_the_control_protocol) {
  HostWeights w = HostWeights::load(kModel);
  RunningWorker both(w, WorkerRole::Both);
  Client c(both.w->control_port());
  Request r;
  r.id = 77;
  r.prompt = {1, 2, 3};
  r.params.max_new_tokens = 100000;  // can never fit: refused
  c.submit(r);
  Request long_one;
  long_one.id = 78;
  long_one.prompt = {4, 5, 6};
  long_one.params.max_new_tokens = 300;
  long_one.params.ignore_eos = true;
  c.submit(long_one);
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  c.cancel(78);
  // Free blocks go back to the total once the cancel is applied.
  bool freed = false;
  for (int i = 0; i < 500 && !freed; ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(4));
    std::lock_guard<std::mutex> lock(c.mu);
    freed = c.last_load.free_blocks == c.last_load.total_blocks && c.last_load.total_blocks > 0;
  }
  CHECK(freed);
  std::lock_guard<std::mutex> lock(c.mu);
  CHECK_EQ(c.errors.size(), 1u);
  CHECK(c.finished.count(78) == 0);
}

RUN_TESTS()
