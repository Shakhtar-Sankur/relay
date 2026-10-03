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
#include <list>
#include <unordered_map>
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

// The engine's block pool, with prefix caching.
//
// A full block's contents depend only on the tokens up to and including it, so each full
// block gets a chain hash: hash(previous block's hash, this block's tokens). Full blocks are
// indexed by that hash. A new request first looks its prompt up block by block and reuses
// every leading block that is already in the cache; only the rest is computed.
//
// Freed blocks that are indexed are not returned to the free list but kept, least recently
// used first, and evicted only when nothing free is left. So the cache holds the most recent
// prefixes (system prompts, earlier turns of a conversation) for as long as memory allows.
// Shared blocks are never written: a request only writes positions past its matched prefix,
// which are in blocks of its own.
//
// Hashes are 64-bit; a collision would silently reuse the wrong block, at odds of about
// one in 2^64 per lookup (vLLM makes the same trade).
class BlockManager {
 public:
  BlockManager(int num_blocks, int block_size, bool prefix_caching);

  int num_blocks() const { return static_cast<int>(ref_.size()); }
  // Blocks that can be allocated now: free, or cached and unused (evictable).
  int num_available() const { return static_cast<int>(free_.size() + lru_.size()); }
  int num_cached() const { return static_cast<int>(index_.size()); }

  // n blocks, all or nothing. Takes free blocks first, then evicts the least recently used.
  bool allocate(int n, std::vector<int>& out);
  void release_all(std::vector<int>& blocks);

  // The longest cached prefix of tokens[0, max_tokens), in whole blocks: its blocks are
  // retained and appended to `blocks`, their hashes to `hashes`. Returns tokens matched.
  int match_prefix(const std::vector<int>& tokens, int max_tokens, std::vector<int>& blocks,
                   std::vector<std::uint64_t>& hashes);
  // Indexes the blocks of a sequence that have become full: tokens[0, computed) are in the
  // cache, in `blocks`; `hashes` holds the hashes of the blocks indexed so far.
  void register_full(const std::vector<int>& tokens, int computed, const std::vector<int>& blocks,
                     std::vector<std::uint64_t>& hashes);

  std::uint64_t hit_tokens() const { return hit_tokens_; }
  std::uint64_t lookup_tokens() const { return lookup_tokens_; }

 private:
  std::uint64_t block_hash(std::uint64_t parent, const int* tokens) const;
  void retain(int b);
  void evict_one();

  int block_size_;
  bool enabled_;
  std::vector<int> ref_;
  std::vector<std::uint64_t> hash_of_;  // 0: not indexed
  std::vector<int> free_;
  std::list<int> lru_;  // unused indexed blocks, least recently used first
  std::vector<std::list<int>::iterator> lru_pos_;
  std::vector<char> in_lru_;
  std::unordered_map<std::uint64_t, int> index_;
  std::uint64_t hit_tokens_ = 0, lookup_tokens_ = 0;
};

}  // namespace relay
