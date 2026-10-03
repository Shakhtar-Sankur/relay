#include <algorithm>
#include <cstring>
#include <set>

#include "check.h"
#include "relay/engine.h"

using namespace relay;

TEST(allocator_hands_out_each_block_once) {
  BlockAllocator a(8);
  std::set<int> got;
  for (int i = 0; i < 8; ++i) got.insert(a.allocate());
  CHECK_EQ(got.size(), 8u);
  CHECK_EQ(a.allocate(), -1);
  CHECK_EQ(a.num_free(), 0);
  a.release(3);
  CHECK_EQ(a.allocate(), 3);
}

TEST(allocator_is_all_or_nothing) {
  BlockAllocator a(4);
  std::vector<int> x, y;
  CHECK(a.allocate(3, x));
  CHECK(!a.allocate(2, y));
  CHECK(y.empty());
  CHECK_EQ(a.num_free(), 1);
  a.release_all(x);
  CHECK(x.empty());
  CHECK_EQ(a.num_free(), 4);
}

TEST(shared_blocks_are_freed_by_the_last_owner) {
  BlockAllocator a(2);
  int b = a.allocate();
  a.retain(b);
  CHECK_EQ(a.refcount(b), 2);
  a.release(b);
  CHECK_EQ(a.num_free(), 1);
  a.release(b);
  CHECK_EQ(a.num_free(), 2);
  bool threw = false;
  try {
    a.release(b);
  } catch (const std::logic_error&) {
    threw = true;
  }
  CHECK(threw);
}

TEST(layout_offsets_are_disjoint_and_dense) {
  KVLayout l{3, 2, 8, 4, 5};
  std::set<std::int64_t> seen;
  for (int b = 0; b < l.num_blocks; ++b)
    for (int h = 0; h < l.kv_heads; ++h)
      for (int s = 0; s < l.block_size; ++s) seen.insert(l.offset(b, h, s));
  CHECK_EQ(static_cast<std::int64_t>(seen.size()), l.layer_elems() / l.head_dim);
  CHECK_EQ(*seen.rbegin() + l.head_dim, l.layer_elems());
  CHECK_EQ(l.blocks_for(0), 0);
  CHECK_EQ(l.blocks_for(4), 1);
  CHECK_EQ(l.blocks_for(5), 2);
}

// Disaggregation in miniature: one backend computes a prompt's KV cache, its blocks are
// copied layer by layer into a second backend (different block ids), and decoding
// continues there. The logits must equal decoding on the first backend.
TEST(kv_blocks_moved_to_another_backend_decode_identically) {
  HostWeights w = HostWeights::load(RELAY_FIXTURES "/llama-mha");
  auto prefill = check::make_backend(w, 16, 4);
  auto decode = check::make_backend(w, 16, 4);
  std::vector<int> prompt = {5, 17, 99, 3, 42, 7, 64, 1, 8, 120, 33};
  const int n = static_cast<int>(prompt.size());
  std::vector<int> src = {0, 1, 2};    // blocks on the prefill side
  std::vector<int> dst = {13, 6, 9};   // where the decode side put them

  ForwardBatch b;
  b.seqs.push_back({prompt, 0, src, true});
  std::vector<float> first = prefill->forward(b);
  int next = argmax(first.data(), w.config.vocab);

  std::vector<unsigned char> kbuf(prefill->kv_block_bytes()), vbuf(prefill->kv_block_bytes());
  for (int l = 0; l < w.config.layers; ++l)
    for (std::size_t i = 0; i < src.size(); ++i) {
      prefill->read_kv_block(l, src[i], kbuf.data(), vbuf.data());
      decode->write_kv_block(l, dst[i], kbuf.data(), vbuf.data());
    }

  for (int step = 0; step < 5; ++step) {
    int p = n + step;
    if (p / 4 >= static_cast<int>(src.size())) {
      src.push_back(3 + step);
      dst.push_back(2 + step);
    }
    ForwardBatch a, c;
    a.seqs.push_back({{next}, p, src, true});
    c.seqs.push_back({{next}, p, dst, true});
    std::vector<float> la = prefill->forward(a), lc = decode->forward(c);
    if (check::exact_backend()) CHECK(std::memcmp(la.data(), lc.data(), la.size() * 4) == 0);
    CHECK_EQ(argmax(la.data(), w.config.vocab), argmax(lc.data(), w.config.vocab));
    next = argmax(la.data(), w.config.vocab);
  }
}

RUN_TESTS()
