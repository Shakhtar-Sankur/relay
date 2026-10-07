#include "relay/backend.h"

#include <cstring>
#include <stdexcept>

namespace relay {

void Backend::update_weight(const std::string& name, const float* src, std::size_t n) {
  throw std::logic_error(this->name() + " backend: update_weight is not supported; write the host weights");
}

void Backend::read_kv_layer(int layer, const std::vector<int>& blocks, void* out) {
  const std::size_t bb = kv_block_bytes(), n = blocks.size();
  auto* k = static_cast<unsigned char*>(out);
  auto* v = k + n * bb;
  for (std::size_t i = 0; i < n; ++i) read_kv_block(layer, blocks[i], k + i * bb, v + i * bb);
}

void Backend::write_kv_layer(int layer, const std::vector<int>& blocks, const void* in) {
  const std::size_t bb = kv_block_bytes(), n = blocks.size();
  const auto* k = static_cast<const unsigned char*>(in);
  const auto* v = k + n * bb;
  for (std::size_t i = 0; i < n; ++i) write_kv_block(layer, blocks[i], k + i * bb, v + i * bb);
}

}  // namespace relay
