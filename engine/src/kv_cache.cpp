#include "relay/kv_cache.h"

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

}  // namespace relay
