// Workers over the control protocol: a colocated worker, and a prefill worker handing
// requests to a decode worker, must both produce what one engine produces. Each worker
// runs in this process on its own thread and is driven over TCP exactly as the Swift
// control plane drives it.
#include <atomic>
#include <chrono>
#include <cstring>
#include <map>
#include <random>
#include <set>
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
  RunningWorker(const HostWeights& weights, WorkerRole role, int blocks = 64, int kv_port = 0) {
    be = check::make_backend(weights, blocks, 8);
    WorkerOptions o;
    o.role = role;
    o.kv_port = kv_port;
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
  std::set<std::uint64_t> retryable;  // requests reported with kErrorRetryable

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
            if (h.flags & kErrorRetryable) retryable.insert(h.request);
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

// Waits until every request is either finished (by any client) or reported retryable.
bool wait_settled(std::vector<Client*> clients, const std::vector<Request>& rs, double seconds = 30) {
  auto deadline = std::chrono::steady_clock::now() + std::chrono::duration<double>(seconds);
  while (std::chrono::steady_clock::now() < deadline) {
    std::size_t settled = 0;
    for (const auto& r : rs) {
      bool done = false;
      for (Client* c : clients) {
        std::lock_guard<std::mutex> lock(c->mu);
        done = done || c->finished.count(r.id) || c->retryable.count(r.id);
      }
      settled += done;
    }
    if (settled == rs.size()) return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  return false;
}

std::vector<int> tokens_of(Client& c, std::uint64_t id) {
  std::lock_guard<std::mutex> lock(c.mu);
  std::map<int, int> by_index;
  for (auto [idx, tok] : c.tokens[id]) by_index[idx] = tok;
  std::vector<int> out;
  for (auto [idx, tok] : by_index) out.push_back(tok);
  return out;
}

TEST(a_dead_decode_worker_fails_only_its_requests_and_they_resume_elsewhere) {
  HostWeights w = HostWeights::load(kModel);
  auto rs = make_requests(w.config.vocab, 10);
  auto want = reference(w, rs);
  RunningWorker prefill(w, WorkerRole::Prefill), d2(w, WorkerRole::Decode);
  auto d1 = std::make_unique<RunningWorker>(w, WorkerRole::Decode);
  Client p(prefill.w->control_port()), c2(d2.w->control_port());
  auto c1 = std::make_unique<Client>(d1->w->control_port());
  p.peer(0, c1->hello.kv_port);
  p.peer(1, c2.hello.kv_port);
  c1.reset();
  d1.reset();  // decode worker 0 is gone, KV port and all
  for (std::size_t i = 0; i < rs.size(); ++i) p.submit(rs[i], static_cast<std::uint32_t>(i % 2));
  CHECK(wait_settled({&p, &c2}, rs));
  {
    std::lock_guard<std::mutex> lock(p.mu);
    for (std::size_t i = 0; i < rs.size(); ++i) CHECK_EQ(p.retryable.count(rs[i].id), i % 2 == 0 ? 1u : 0u);
  }
  // The control plane's recovery: resume each failed request on the surviving decode
  // worker with whatever it had already produced (its first token, if the prefill worker
  // reported one).
  for (std::size_t i = 0; i < rs.size(); i += 2) c2.resume(rs[i], tokens_of(p, rs[i].id));
  CHECK(wait_finished({&p, &c2}, rs.size()));
  CHECK(merged({&p, &c2}) == want);
}

TEST(forgetting_a_peer_settles_every_request_and_frees_every_block) {
  HostWeights w = HostWeights::load(kModel);
  auto rs = make_requests(w.config.vocab, 12);
  auto want = reference(w, rs);
  RunningWorker prefill(w, WorkerRole::Prefill), decode(w, WorkerRole::Decode);
  Client p(prefill.w->control_port()), d(decode.w->control_port());
  p.peer(0, d.hello.kv_port);
  for (const auto& r : rs) p.submit(r, 0);
  p.frame(Ctl::Peer, 0, 0, 0, {});  // forget decode worker 0, mid-flight
  CHECK(wait_settled({&p, &d}, rs));
  for (const auto& r : rs) {
    bool finished, retry;
    {
      std::lock_guard<std::mutex> lp(p.mu), ld(d.mu);
      finished = p.finished.count(r.id) || d.finished.count(r.id);
      retry = p.retryable.count(r.id) > 0;
    }
    CHECK(finished != retry);  // exactly one
    if (finished) CHECK(merged({&p, &d})[r.id] == want[r.id]);
  }
  bool freed = false;
  for (int i = 0; i < 500 && !freed; ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(4));
    std::lock_guard<std::mutex> lock(p.mu);
    freed = p.last_load.free_blocks == p.last_load.total_blocks && p.last_load.total_blocks > 0;
  }
  CHECK(freed);
  // A Submit to the forgotten peer is refused as retryable, not accepted and lost.
  Request late = rs[0];
  late.id = 999;
  p.submit(late, 0);
  for (int i = 0; i < 500; ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    std::lock_guard<std::mutex> lock(p.mu);
    if (p.retryable.count(999)) break;
  }
  std::lock_guard<std::mutex> lock(p.mu);
  CHECK(p.retryable.count(999) == 1);
}

TEST(a_new_control_connection_drops_the_old_ones_requests) {
  HostWeights w = HostWeights::load(kModel);
  RunningWorker both(w, WorkerRole::Both);
  Request r;
  r.id = 5;
  r.prompt = {4, 5, 6};
  r.params.max_new_tokens = 400;
  r.params.ignore_eos = true;
  {
    Client old(both.w->control_port());
    old.submit(r);
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }  // the control plane that sent it is gone
  Client now(both.w->control_port());
  bool freed = false;
  for (int i = 0; i < 500 && !freed; ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(4));
    std::lock_guard<std::mutex> lock(now.mu);
    freed = now.last_load.free_blocks == now.last_load.total_blocks && now.last_load.total_blocks > 0;
  }
  CHECK(freed);
  std::lock_guard<std::mutex> lock(now.mu);
  CHECK(now.tokens.count(5) == 0);  // nothing of the old request reaches the new plane
}

TEST(a_prefill_worker_reconnects_to_a_restarted_decode_worker) {
  HostWeights w = HostWeights::load(kModel);
  auto rs = make_requests(w.config.vocab, 2);
  auto want = reference(w, rs);
  RunningWorker prefill(w, WorkerRole::Prefill);
  Client p(prefill.w->control_port());
  int kv_port;
  {
    RunningWorker d(w, WorkerRole::Decode);
    Client c(d.w->control_port());
    kv_port = c.hello.kv_port;
    p.peer(0, kv_port);
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }  // the decode worker dies...
  RunningWorker d(w, WorkerRole::Decode, 64, kv_port);  // ...and comes back on the same KV port
  Client c(d.w->control_port());
  // The first request finds the old connection broken: it is reported for retry.
  p.submit(rs[0], 0);
  CHECK(wait_settled({&p, &c}, {rs[0]}));
  // A later one makes the prefill worker reconnect on its own, and goes through.
  bool done = false;
  for (int attempt = 0; attempt < 5 && !done; ++attempt) {
    std::this_thread::sleep_for(std::chrono::milliseconds(attempt == 0 ? 0 : 1100));
    Request r = rs[1];
    r.id = 100 + attempt;
    p.submit(r, 0);
    CHECK(wait_settled({&p, &c}, {r}));
    std::lock_guard<std::mutex> lp(p.mu), lc(c.mu);
    done = c.finished.count(r.id) > 0;
    if (done) {
      std::map<int, int> got;
      for (auto [i, t] : p.tokens[r.id]) got[i] = t;
      for (auto [i, t] : c.tokens[r.id]) got[i] = t;
      std::vector<int> v;
      for (auto [i, t] : got) v.push_back(t);
      CHECK(v == want[rs[1].id]);
    }
  }
  CHECK(done);
}

TEST(many_identical_requests_across_two_prefill_and_two_decode_workers) {
  // Identical prompts arriving together: the shape that once returned wrong tokens.
  HostWeights w = HostWeights::load(kModel);
  auto base = make_requests(w.config.vocab, 1)[0];
  base.params.max_new_tokens = 8;
  base.params.temperature = 0;
  RunningWorker p0(w, WorkerRole::Prefill), p1(w, WorkerRole::Prefill), d0(w, WorkerRole::Decode), d1(w, WorkerRole::Decode);
  Client cp0(p0.w->control_port()), cp1(p1.w->control_port()), cd0(d0.w->control_port()), cd1(d1.w->control_port());
  for (Client* p : {&cp0, &cp1}) {
    p->peer(0, cd0.hello.kv_port);
    p->peer(1, cd1.hello.kv_port);
  }
  std::vector<Request> rs;
  const int rounds = check::env_int("RELAY_STRESS_ROUNDS", 20);
  for (int round = 0; round < rounds; ++round) {
    std::vector<Request> batch;
    for (int i = 0; i < 16; ++i) {
      Request r = base;
      r.id = 10000 + round * 100 + i;
      batch.push_back(r);
      (i % 2 ? cp1 : cp0).submit(r, static_cast<std::uint32_t>((i / 2) % 2));
    }
    CHECK(wait_finished({&cp0, &cp1, &cd0, &cd1}, rs.size() + batch.size()));
    rs.insert(rs.end(), batch.begin(), batch.end());
  }
  auto want = reference(w, {base})[base.id];
  auto got = merged({&cp0, &cp1, &cd0, &cd1});
  int bad = 0;
  for (const auto& r : rs)
    if (got[r.id] != want) ++bad;
  if (bad) std::fprintf(stderr, "  %d of %zu requests differ\n", bad, rs.size());
  CHECK_EQ(bad, 0);
}

RUN_TESTS()
