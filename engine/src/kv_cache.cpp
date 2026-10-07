#include "relay/kv_cache.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace relay {

BlockAllocator::BlockAllocator(int num_blocks) : refcount_(num_blocks, 0) {
  if (num_blocks <= 0) throw std::invalid_argument("BlockAllocator needs at least one block");
  free_.reserve(num_blocks);
  // Pushed in reverse so that blocks are handed out as 0, 1, 2, ...
  for (int b = num_blocks - 1; b >= 0; --b) free_.push_back(b);
}

int BlockAllocator::allocate() {
  if (free_.empty()) return -1;
  int b = free_.back();
  free_.pop_back();
  refcount_[b] = 1;
  return b;
}

bool BlockAllocator::allocate(int n, std::vector<int>& out) {
  if (n > num_free()) return false;
  for (int i = 0; i < n; ++i) out.push_back(allocate());
  return true;
}

void BlockAllocator::retain(int block) {
  if (refcount_.at(block) <= 0) throw std::logic_error("retain of a free block " + std::to_string(block));
  ++refcount_[block];
}

void BlockAllocator::release(int block) {
  if (refcount_.at(block) <= 0) throw std::logic_error("release of a free block " + std::to_string(block));
  if (--refcount_[block] == 0) free_.push_back(block);
}

void BlockAllocator::release_all(std::vector<int>& blocks) {
  // Released last block first, so a sequence that is re-admitted gets the same blocks back.
  for (auto it = blocks.rbegin(); it != blocks.rend(); ++it) release(*it);
  blocks.clear();
}

BlockManager::BlockManager(int num_blocks, int block_size, bool prefix_caching)
    : block_size_(block_size),
      enabled_(prefix_caching),
      ref_(num_blocks, 0),
      hash_of_(num_blocks, 0),
      lru_pos_(num_blocks),
      in_lru_(num_blocks, 0) {
  if (num_blocks <= 0 || block_size <= 0) throw std::invalid_argument("BlockManager needs blocks");
  free_.reserve(num_blocks);
  for (int b = num_blocks - 1; b >= 0; --b) free_.push_back(b);
}

std::uint64_t BlockManager::block_hash(std::uint64_t parent, const int* tokens) const {
  std::uint64_t h = parent ^ 0x6a09e667f3bcc908ULL;
  for (int i = 0; i < block_size_; ++i) {
    h ^= static_cast<std::uint64_t>(static_cast<std::uint32_t>(tokens[i])) + 0x9E3779B97F4A7C15ULL + (h << 6) + (h >> 2);
    h = (h ^ (h >> 31)) * 0xBF58476D1CE4E5B9ULL;
  }
  h ^= h >> 29;
  return h == 0 ? 1 : h;
}

void BlockManager::drop_cache() {
  while (!lru_.empty()) evict_one();
  for (auto& [h, b] : index_) hash_of_[b] = 0;  // still referenced by running requests
  index_.clear();
}

void BlockManager::evict_one() {
  int b = lru_.front();
  lru_.pop_front();
  in_lru_[b] = 0;
  index_.erase(hash_of_[b]);
  hash_of_[b] = 0;
  free_.push_back(b);
}

bool BlockManager::allocate(int n, std::vector<int>& out) {
  if (n > num_available()) return false;
  for (int i = 0; i < n; ++i) {
    if (free_.empty()) evict_one();
    int b = free_.back();
    free_.pop_back();
    ref_[b] = 1;
    out.push_back(b);
  }
  return true;
}

void BlockManager::retain(int b) {
  if (in_lru_[b]) {
    lru_.erase(lru_pos_[b]);
    in_lru_[b] = 0;
  }
  ++ref_[b];
}

void BlockManager::release_all(std::vector<int>& blocks) {
  for (auto it = blocks.rbegin(); it != blocks.rend(); ++it) {
    int b = *it;
    if (ref_.at(b) <= 0) throw std::logic_error("release of a free block " + std::to_string(b));
    if (--ref_[b] > 0) continue;
    if (hash_of_[b] != 0) {
      lru_pos_[b] = lru_.insert(lru_.end(), b);
      in_lru_[b] = 1;
    } else {
      free_.push_back(b);
    }
  }
  blocks.clear();
}

int BlockManager::match_prefix(const std::vector<int>& tokens, int max_tokens, std::vector<int>& blocks,
                               std::vector<std::uint64_t>& hashes) {
  lookup_tokens_ += static_cast<std::uint64_t>(std::max(0, max_tokens));
  if (!enabled_) return 0;
  int full = std::max(0, max_tokens) / block_size_, matched = 0;
  std::uint64_t h = 0;
  for (int i = 0; i < full; ++i) {
    h = block_hash(h, tokens.data() + static_cast<std::size_t>(i) * block_size_);
    auto it = index_.find(h);
    if (it == index_.end()) break;
    retain(it->second);
    blocks.push_back(it->second);
    hashes.push_back(h);
    ++matched;
  }
  hit_tokens_ += static_cast<std::uint64_t>(matched) * block_size_;
  return matched * block_size_;
}

void BlockManager::register_full(const std::vector<int>& tokens, int computed, const std::vector<int>& blocks,
                                 std::vector<std::uint64_t>& hashes) {
  if (!enabled_) return;
  int full = computed / block_size_;
  for (int i = static_cast<int>(hashes.size()); i < full; ++i) {
    std::uint64_t parent = i == 0 ? 0 : hashes[i - 1];
    std::uint64_t h = block_hash(parent, tokens.data() + static_cast<std::size_t>(i) * block_size_);
    hashes.push_back(h);
    int b = blocks[i];
    if (hash_of_[b] == 0 && index_.find(h) == index_.end()) {
      index_[h] = b;
      hash_of_[b] = h;
    }
  }
}

}  // namespace relay
