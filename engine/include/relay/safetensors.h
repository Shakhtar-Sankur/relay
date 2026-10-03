// Reads .safetensors files without copying them: the file is memory-mapped and
// each tensor is a view into the mapping.
//
// Format: 8 bytes little-endian header length N, then N bytes of JSON
//   {"name": {"dtype": "BF16", "shape": [r, c], "data_offsets": [begin, end]}, ...}
// then the raw tensor bytes; offsets are relative to the end of the header.
#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace relay {

enum class DType { F32, F16, BF16 };

std::size_t dtype_size(DType d);
const char* dtype_name(DType d);

struct TensorView {
  DType dtype;
  std::vector<std::int64_t> shape;
  const void* data;
  std::size_t nbytes;

  std::int64_t numel() const;
  // Converts to float32 (bf16 and f16 are exact in f32).
  std::vector<float> to_f32() const;
};

float bf16_to_f32(std::uint16_t h);
float f16_to_f32(std::uint16_t h);
std::uint16_t f32_to_f16(float f);  // round to nearest even

class SafeTensors {
 public:
  explicit SafeTensors(const std::string& path);
  ~SafeTensors();
  SafeTensors(const SafeTensors&) = delete;
  SafeTensors& operator=(const SafeTensors&) = delete;
  SafeTensors(SafeTensors&& o) noexcept;
  SafeTensors& operator=(SafeTensors&&) = delete;

  bool has(const std::string& name) const { return tensors_.count(name) != 0; }
  const TensorView& get(const std::string& name) const;
  const std::map<std::string, TensorView>& tensors() const { return tensors_; }

 private:
  void* map_ = nullptr;
  std::size_t size_ = 0;
  std::map<std::string, TensorView> tensors_;
};

// A model directory may hold one file (model.safetensors) or several shards
// listed in model.safetensors.index.json. This opens whichever is there.
class WeightFiles {
 public:
  explicit WeightFiles(const std::string& dir);
  bool has(const std::string& name) const;
  const TensorView& get(const std::string& name) const;

 private:
  std::vector<SafeTensors> files_;
};

}  // namespace relay
