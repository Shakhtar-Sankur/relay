// The KV transfer engine: transports, then prefill on one backend and decode on another.
// The disaggregated result must equal one engine doing everything: same tokens, and
// on the CPU backend the same logits bit for bit.
#include <atomic>
#include <chrono>
#include <cstring>
#include <map>
#include <random>
#include <thread>
#include <unistd.h>

#include "check.h"
#include "relay/engine.h"
#include "relay/kv_transfer.h"
#include "relay/transport.h"

using namespace relay;

namespace {

using Pair = std::pair<std::unique_ptr<Connection>, std::unique_ptr<Connection>>;

Pair tcp_pair() {
  TcpListener l(0);
  std::unique_ptr<Connection> client;
  std::thread t([&] { client = tcp_connect("127.0.0.1", l.port()); });
  auto server = l.accept();
  t.join();
  return {std::move(client), std::move(server)};
}

Pair shm_pair(std::size_t capacity) {
  std::string name = "/relay-test-" + std::to_string(::getpid()) + "-" + std::to_string(std::rand());
  auto a = shm_create(name, capacity);
  auto b = shm_open(name);
  return {std::move(a), std::move(b)};
}

std::uint64_t mix(std::uint64_t h, unsigned char c) { return (h ^ c) * 1099511628211ULL; }

// Sends frames of random sizes (some far larger than the shm ring) as several slices
// each, and checks every byte arrives in order.
void stream_check(Connection& tx, Connection& rx) {
  std::mt19937_64 rng(5);
  std::vector<std::size_t> sizes;
  for (int i = 0; i < 300; ++i) sizes.push_back(rng() % (i % 10 == 0 ? (3u << 20) : 5000u) + 1);
  std::uint64_t sent_hash = 1469598103934665603ULL, got_hash = sent_hash;
  std::thread sender([&] {
    std::mt19937 bytes(9);
    for (std::size_t n : sizes) {
      std::vector<unsigned char> a(n / 3), b(n - n / 3);
      for (auto& c : a) c = static_cast<unsigned char>(bytes());
      for (auto& c : b) c = static_cast<unsigned char>(bytes());
      for (auto c : a) sent_hash = mix(sent_hash, c);
      for (auto c : b) sent_hash = mix(sent_hash, c);
      Slice s[2] = {{a.data(), a.size()}, {b.data(), b.size()}};
      tx.send(s, 2);
    }
  });
  for (std::size_t n : sizes) {
    std::vector<unsigned char> buf(n);
    rx.recv(buf.data(), n);
    for (auto c : buf) got_hash = mix(got_hash, c);
  }
  sender.join();
  CHECK(sent_hash == got_hash);
}

std::vector<Request> make_requests(int vocab) {
  std::mt19937 rng(11);
  std::vector<Request> rs;
  int lengths[] = {3, 40, 17, 1, 25, 64, 9, 33};
  for (int i = 0; i < 8; ++i) {
    Request r;
    r.id = 500 + i;
    for (int t = 0; t < lengths[i]; ++t) r.prompt.push_back(static_cast<int>(rng() % vocab));
    r.params.max_new_tokens = i == 3 ? 1 : 4 + (i * 3) % 9;  // one request finishes at its first token
    r.params.ignore_eos = true;
    if (i % 2 == 1) {
      r.params.temperature = 0.8f;
      r.params.seed = 77 + i;
    }
    rs.push_back(r);
  }
  return rs;
}

using Log = std::map<std::pair<std::uint64_t, int>, std::vector<float>>;

// One engine does everything.
std::map<std::uint64_t, std::vector<int>> colocated(const HostWeights& w, const std::vector<Request>& rs, Log& log) {
  auto be = check::make_backend(w, 64, 8);
  Engine e(*be, {256, 64});
  e.on_logits = [&](std::uint64_t id, int idx, const float* lg) { log[{id, idx}] = {lg, lg + w.config.vocab}; };
  for (const auto& r : rs) e.add(r);
  return e.run_all();
}

// Prefill on one backend, decode on another, KV cache over `conn`.
std::map<std::uint64_t, std::vector<int>> disaggregated(const HostWeights& w, const std::vector<Request>& rs,
                                                        Pair conn, Log& log, int prefill_budget, int decode_blocks,
                                                        TransferStats* tstats = nullptr, EngineStats* dstats = nullptr,
                                                        TransferStats* rstats = nullptr) {
  auto pb = check::make_backend(w, 64, 8);
  auto db = check::make_backend(w, decode_blocks, 8);
  Engine decode(*db, {256, 64});
  std::mutex log_mu;
  decode.on_logits = [&](std::uint64_t id, int idx, const float* lg) {
    std::lock_guard<std::mutex> lock(log_mu);
    log[{id, idx}] = {lg, lg + w.config.vocab};
  };
  std::map<std::uint64_t, std::vector<int>> out;
  std::atomic<int> unfinished{static_cast<int>(rs.size())};
  {
    KVReceiver receiver(decode, *db, *conn.second);
    std::thread prefill_thread([&] {
      KVSender sender(*pb, *conn.first);
      PrefillWorker pw(*pb, sender, prefill_budget);
      pw.on_logits = [&](std::uint64_t id, const float* lg) {
        std::lock_guard<std::mutex> lock(log_mu);
        log[{id, 0}] = {lg, lg + w.config.vocab};
      };
      for (const auto& r : rs) pw.add(r);
      while (pw.has_work())
        for (const TokenEvent& e : pw.step()) {
          std::lock_guard<std::mutex> lock(log_mu);
          out[e.id].push_back(e.token);
          if (e.finish != Finish::None) --unfinished;
        }
      sender.flush();
      if (tstats) *tstats = sender.stats();
      CHECK_EQ(pw.free_blocks(), 64);  // every block freed once its frames were sent
    });
    while (unfinished > 0) {
      auto events = decode.step();
      if (events.empty()) std::this_thread::sleep_for(std::chrono::microseconds(100));
      for (const TokenEvent& e : events) {
        std::lock_guard<std::mutex> lock(log_mu);
        out[e.id].push_back(e.token);
        if (e.finish != Finish::None) --unfinished;
      }
    }
    prefill_thread.join();
    conn.first->close();
    receiver.join();
    CHECK(!receiver.error());
    CHECK_EQ(receiver.in_flight(), 0);
    if (rstats) *rstats = receiver.stats();
  }
  CHECK_EQ(decode.free_blocks(), decode_blocks);
  if (dstats) *dstats = decode.stats();
  return out;
}

void compare(const Log& a, const Log& b) {
  CHECK_EQ(a.size(), b.size());
  int differing = 0;
  for (const auto& [k, row] : a) {
    auto it = b.find(k);
    if (it == b.end() || std::memcmp(row.data(), it->second.data(), row.size() * 4) != 0) ++differing;
  }
  CHECK_EQ(differing, 0);
}

const char* kModel = RELAY_FIXTURES "/llama-gqa-tied-bf16";

}  // namespace

TEST(tcp_carries_every_byte_in_order) {
  auto [a, b] = tcp_pair();
  stream_check(*a, *b);
  stream_check(*b, *a);
}

TEST(shm_ring_carries_every_byte_in_order) {
  auto [a, b] = shm_pair(64 << 10);  // a 64 KiB ring: most frames wrap around it many times
  stream_check(*a, *b);
  stream_check(*b, *a);
}

TEST(closing_wakes_a_blocked_receiver) {
  for (int kind = 0; kind < 2; ++kind) {
    Pair p = kind == 0 ? tcp_pair() : shm_pair(1 << 16);
    std::thread closer([&] {
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      p.first->close();
    });
    bool threw = false;
    try {
      char c;
      p.second->recv(&c, 1);
    } catch (const ConnectionClosed&) {
      threw = true;
    }
    closer.join();
    CHECK(threw);
  }
}

TEST(backends_report_each_layer_during_the_forward_pass) {
  HostWeights w = HostWeights::load(kModel);
  auto be = check::make_backend(w, 16, 8);
  struct Count : LayerObserver {
    std::vector<int> layers;
    void kv_written(int l) override { layers.push_back(l); }
  } obs;
  ForwardBatch b;
  b.seqs.push_back({{1, 2, 3, 4, 5}, 0, {0}, true});
  be->forward(b, &obs);
  std::vector<int> want(w.config.layers);
  for (int i = 0; i < w.config.layers; ++i) want[i] = i;
  CHECK(obs.layers == want);
}

TEST(disaggregated_over_tcp_equals_colocated) {
  HostWeights w = HostWeights::load(kModel);
  auto rs = make_requests(w.config.vocab);
  Log one, two;
  TransferStats ts;
  auto a = colocated(w, rs, one);
  auto b = disaggregated(w, rs, tcp_pair(), two, 512, 64, &ts);
  CHECK(a == b);
  if (check::exact_backend()) compare(one, two);
  std::fprintf(stderr, "  %llu frames, %llu KV bytes, %llu requests\n", (unsigned long long)ts.frames,
               (unsigned long long)ts.kv_bytes, (unsigned long long)ts.requests);
  CHECK_EQ(ts.requests, rs.size());
}

TEST(disaggregated_over_shm_with_chunked_prefill_equals_colocated) {
  HostWeights w = HostWeights::load(kModel);
  auto rs = make_requests(w.config.vocab);
  Log one, two;
  auto a = colocated(w, rs, one);
  // 10 tokens per prefill step: prompts span several steps, so a block is sent once per
  // chunk that touches it, and the last version must win.
  auto b = disaggregated(w, rs, shm_pair(1 << 20), two, 10, 64);
  CHECK(a == b);
  if (check::exact_backend()) compare(one, two);
}

TEST(memory_pressure_on_the_decode_side_changes_nothing) {
  HostWeights w = HostWeights::load(kModel);
  auto rs = make_requests(w.config.vocab);
  Log one, two;
  EngineStats ds;
  TransferStats rs_stats;
  auto a = colocated(w, rs, one);
  // 12 blocks of 8 tokens on the decode side: requests arrive faster than they fit, so
  // some transfers find no room (the prompt is recomputed instead) and the decode
  // engine preempts.
  auto b = disaggregated(w, rs, tcp_pair(), two, 512, 12, nullptr, &ds, &rs_stats);
  CHECK(a == b);
  if (check::exact_backend()) compare(one, two);
  std::fprintf(stderr, "  decode side: %llu transfers fell back to recompute, %llu preemptions\n",
               (unsigned long long)rs_stats.recomputed, (unsigned long long)ds.preemptions);
  CHECK(rs_stats.recomputed + ds.preemptions > 0);
}

TEST(a_lost_sender_returns_its_blocks) {
  HostWeights w = HostWeights::load(kModel);
  auto pb = check::make_backend(w, 16, 8);
  auto db = check::make_backend(w, 16, 8);
  Engine decode(*db, {64, 8});
  auto [tx, rx] = tcp_pair();
  KVReceiver receiver(decode, *db, *rx);
  {
    KVSender sender(*pb, *tx);
    Request r;
    r.id = 1;
    r.prompt.assign(20, 7);  // 3 blocks
    sender.begin(r, 3);
    sender.layer(1, 0, 0, {0, 1, 2});
    sender.flush();
  }
  // Wait until the receiver has reserved the blocks, then the prefill worker "dies".
  for (int i = 0; i < 1000 && receiver.in_flight() == 0; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(1));
  CHECK_EQ(receiver.in_flight(), 1);
  CHECK_EQ(decode.free_blocks(), 13);
  tx->close();
  receiver.join();
  CHECK_EQ(receiver.in_flight(), 0);
  CHECK_EQ(decode.free_blocks(), 16);
  CHECK(!decode.has_work());
}

TEST(a_different_model_is_refused) {
  HostWeights a = HostWeights::load(RELAY_FIXTURES "/llama-mha");
  HostWeights b = HostWeights::load(RELAY_FIXTURES "/llama-gqa-tied-bf16");
  auto pa = check::make_backend(a, 16, 8);
  auto db = check::make_backend(b, 16, 8);
  Engine decode(*db, {64, 8});
  auto [tx, rx] = tcp_pair();
  KVReceiver receiver(decode, *db, *rx);
  {
    KVSender sender(*pa, *tx);
    sender.flush();
  }
  receiver.join();
  CHECK(receiver.error() != nullptr);
}

RUN_TESTS()
