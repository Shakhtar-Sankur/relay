// The paged KV cache, host side: which fixed-size blocks belong to which sequence.
//
// The cache memory itself lives in a backend (host RAM or GPU memory), laid out per
// layer as  K[layer][block][kv_head][slot][head_dim]  and the same for V, so one
// layer's slice of one block is a single contiguous run of bytes. That is what the
// KV transfer engine (M2) ships between workers, layer by layer.
//
// A sequence's block table lists its blocks in order; token position p lives in
// block table[p / block_size] at slot p % block_size. Blocks are reference counted
// so that sequences with a common prefix can share them (prefix caching, M4).
#pragma once

#include <cstdint>
#include <vector>

namespace relay {

struct KVLayout {
  int layers = 0;
  int kv_heads = 0;
  int head_dim = 0;
  int block_size = 16;
  int num_blocks = 0;

  // Elements in one layer's K (or V) slice of one block.
  std::int64_t block_elems() const { return static_cast<std::int64_t>(kv_heads) * block_size * head_dim; }
  // Elements in one layer's K (or V) for all blocks.
  std::int64_t layer_elems() const { return block_elems() * num_blocks; }
  // Offset of (block, kv_head, slot) inside one layer's K or V.
  std::int64_t offset(int block, int kv_head, int slot) const {
    return (static_cast<std::int64_t>(block) * kv_heads + kv_head) * block_size * head_dim +
           static_cast<std::int64_t>(slot) * head_dim;
  }
  int blocks_for(int tokens) const { return (tokens + block_size - 1) / block_size; }
};

class BlockAllocator {
 public:
  explicit BlockAllocator(int num_blocks);

  int num_blocks() const { return static_cast<int>(refcount_.size()); }
  int num_free() const { return static_cast<int>(free_.size()); }

  // Returns a block with reference count 1, or -1 when none is free.
  int allocate();
  // Allocates n blocks, all or nothing; returns false (and allocates none) if
  // fewer than n are free.
  bool allocate(int n, std::vector<int>& out);
  void retain(int block);
  // Drops one reference; the block is free again when the count reaches zero.
  void release(int block);
  void release_all(std::vector<int>& blocks);
  int refcount(int block) const { return refcount_.at(block); }

 private:
  std::vector<int> refcount_;
  std::vector<int> free_;  // used as a stack: the most recently freed block is reused first
};

}  // namespace relay
