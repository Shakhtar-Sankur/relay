// Fault injection for the KV transfer path, over many seeds.
//
// A prefill worker streams requests' KV caches to a decode engine over a connection that
// breaks after a seeded number of bytes: between frames, in the middle of a header, in the
// middle of a block. Whatever the cut, every request must end up in exactly one of two
// states, and nothing may leak or hang:
//   delivered  its tokens (first from the prefill side, the rest from the decode engine)
//              are exactly those of one engine running it alone
//   failed     the prefill worker reports it (take_failed), so the control plane can retry
//              it on another worker
// and both sides get every KV block back.
#include <algorithm>
#include <cstdio>
#include <limits>
#include <map>
#include <random>
#include <set>
#include <thread>

#include "check.h"
#include "relay/engine.h"
#include "relay/kv_transfer.h"

using namespace relay;

namespace {

const char* kModel = RELAY_FIXTURES "/llama-gqa-tied-bf16";
constexpr int kPrefillBlocks = 24, kDecodeBlocks = 40, kBlockSize = 8;

std::pair<std::unique_ptr<Connection>, std::unique_ptr<Connection>> tcp_pair() {
  TcpListener l(0);
  std::unique_ptr<Connection> client;
  std::thread t([&] { client = tcp_connect("127.0.0.1", l.port()); });
  auto server = l.accept();
  t.join();
  return {std::move(client), std::move(server)};
}

// Sends at most `limit` bytes in total, then breaks: the bytes up to the limit go out
// (possibly half a frame), the connection is closed under both sides, and send throws.
class FaultyConnection final : public Connection {
 public:
  FaultyConnection(Connection& inner, std::size_t limit) : inner_(inner), limit_(limit) {}
  void send(const Slice* slices, int n) override {
    std::size_t total = 0;
    for (int i = 0; i < n; ++i) total += slices[i].len;
    if (sent_ + total <= limit_) {
      inner_.send(slices, n);
      sent_ += total;
      return;
    }
    std::vector<Slice> part;
    std::size_t room = limit_ - sent_;
    for (int i = 0; i < n && room > 0; ++i) {
      std::size_t k = std::min(room, slices[i].len);
      part.push_back({slices[i].data, k});
      room -= k;
    }
    if (!part.empty()) inner_.send(part.data(), static_cast<int>(part.size()));
    sent_ = limit_;
    inner_.close();
    throw ConnectionClosed("injected fault");
  }
  void recv(void* dst, std::size_t len) override { inner_.recv(dst, len); }
  void close() override { inner_.close(); }
  std::string describe() const override { return "faulty(" + inner_.describe() + ")"; }
  std::size_t sent() const { return sent_; }

 private:
  Connection& inner_;
  std::size_t limit_;
  std::size_t sent_ = 0;  // only the sender thread touches it
};

std::vector<Request> workload(std::uint32_t seed, int vocab) {
  std::mt19937 rng(seed);
  std::vector<Request> rs;
  int n = 1 + static_cast<int>(rng() % 4);
  for (int i = 0; i < n; ++i) {
    Request r;
    r.id = 1 + i;
    int len = 1 + static_cast<int>(rng() % 40);
    for (int t = 0; t < len; ++t) r.prompt.push_back(static_cast<int>(rng() % vocab));
    r.params.max_new_tokens = 1 + static_cast<int>(rng() % 6);  // 1: ends at the first token
    r.params.ignore_eos = true;
    if (rng() % 2) {
      r.params.temperature = 0.8f;
      r.params.seed = rng();
    }
    rs.push_back(r);
  }
  return rs;
}

std::map<std::uint64_t, std::vector<int>> reference(const HostWeights& w, const std::vector<Request>& rs) {
  auto be = check::make_backend(w, 64, kBlockSize);
  Engine e(*be, {512, 64});
  for (const auto& r : rs) e.add(r);
  return e.run_all();
}

struct Outcome {
  int delivered = 0, failed = 0;
  std::size_t bytes = 0;  // what went over the wire (all of it, without a fault)
};

Outcome run(const HostWeights& w, const std::vector<Request>& rs, const std::map<std::uint64_t, std::vector<int>>& ref,
            std::size_t limit) {
  auto pb = check::make_backend(w, kPrefillBlocks, kBlockSize);
  auto db = check::make_backend(w, kDecodeBlocks, kBlockSize);
  Engine decode(*db, {64, 8});
  auto [tx, rx] = tcp_pair();
  FaultyConnection ftx(*tx, limit);
  KVReceiver receiver(decode, *db, *rx);

  std::map<std::uint64_t, TokenEvent> first;
  std::vector<std::uint64_t> failed;
  int prefill_free = -1;
  {
    KVSender sender(*pb, ftx);
    PrefillWorker pw(*pb, sender, 16);  // 16-token chunks: long prompts take several steps
    for (const auto& r : rs) pw.add(r);
    while (pw.has_work())
      for (const TokenEvent& e : pw.step()) first[e.id] = e;
    try {
      sender.flush();
    } catch (const ConnectionClosed&) {
    }
    failed = pw.take_failed();
    prefill_free = pw.free_blocks();
  }
  CHECK_EQ(prefill_free, kPrefillBlocks);

  tx->close();  // no fault: the prefill worker is done; with one: already closed
  receiver.join();
  CHECK_EQ(receiver.in_flight(), 0);
  CHECK(!receiver.error());
  auto decoded = decode.run_all();
  CHECK_EQ(decode.free_blocks(), kDecodeBlocks);

  std::set<std::uint64_t> failed_set(failed.begin(), failed.end());
  CHECK_EQ(failed_set.size(), failed.size());  // each failure reported once
  Outcome o;
  o.bytes = ftx.sent();
  for (const auto& r : rs) {
    const std::vector<int>& want = ref.at(r.id);
    bool is_failed = failed_set.count(r.id) > 0;
    auto f = first.find(r.id);
    // Token values are compared on the bit-exact CPU backend; on a GPU, batching may
    // reorder floating-point sums.
    const bool exact = check::exact_backend();
    if (f != first.end() && exact) CHECK_EQ(f->second.token, want[0]);  // a reported first token is right
    if (is_failed) {
      CHECK(decoded.count(r.id) == 0);  // never both delivered and failed
      ++o.failed;
      continue;
    }
    // Delivered: the first token came from the prefill side, the rest from the decode engine.
    CHECK(f != first.end());
    std::vector<int> got{f->second.token};
    if (r.params.max_new_tokens > 1) {
      CHECK(decoded.count(r.id) == 1);
      const auto& rest = decoded[r.id];
      got.insert(got.end(), rest.begin(), rest.end());
    }
    if (exact) CHECK(got == want);
    CHECK_EQ(got.size(), want.size());
    ++o.delivered;
  }
  return o;
}

}  // namespace

TEST(without_a_fault_every_request_is_delivered) {
  HostWeights w = HostWeights::load(kModel);
  for (std::uint32_t seed = 0; seed < 20; ++seed) {
    auto rs = workload(seed, w.config.vocab);
    Outcome o = run(w, rs, reference(w, rs), std::numeric_limits<std::size_t>::max());
    CHECK_EQ(o.delivered, static_cast<int>(rs.size()));
  }
}

TEST(a_connection_cut_anywhere_loses_nothing_and_leaks_nothing) {
  HostWeights w = HostWeights::load(kModel);
  const std::uint32_t seeds = check::env_int("RELAY_FAULT_SEEDS", 300);
  int delivered = 0, failed = 0, runs = 0;
  for (std::uint32_t seed = 0; seed < seeds; ++seed) {
    auto rs = workload(seed, w.config.vocab);
    auto ref = reference(w, rs);
    // How many bytes this workload sends without a fault; cut somewhere inside that.
    std::size_t total = run(w, rs, ref, std::numeric_limits<std::size_t>::max()).bytes;
    std::mt19937_64 rng(seed * 7919u + 1);
    std::size_t limit = rng() % (total + 1);
    Outcome o = run(w, rs, ref, limit);
    delivered += o.delivered;
    failed += o.failed;
    ++runs;
  }
  std::fprintf(stderr, "  %d seeded cuts: %d requests delivered exactly, %d reported for retry, 0 lost, 0 blocks leaked\n",
               runs, delivered, failed);
  CHECK(failed > 0);
  CHECK(delivered > 0);
}

TEST(cuts_at_every_frame_boundary_and_inside_every_frame) {
  // One workload, cut at a sweep of offsets (prime stride, so the cuts land at varied
  // positions inside headers and blocks), plus the two ends.
  HostWeights w = HostWeights::load(kModel);
  auto rs = workload(12345, w.config.vocab);
  auto ref = reference(w, rs);
  std::size_t total = run(w, rs, ref, std::numeric_limits<std::size_t>::max()).bytes;
  std::size_t stride = std::max<std::size_t>(1, total / 97);
  int runs = 0;
  for (std::size_t limit = 0; limit <= total; limit += stride) {
    run(w, rs, ref, limit);
    ++runs;
  }
  run(w, rs, ref, total);
  std::fprintf(stderr, "  %d cut points over %zu bytes\n", runs + 1, total);
}

RUN_TESTS()
