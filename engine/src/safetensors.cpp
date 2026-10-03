#include "relay/safetensors.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>

#include "relay/json.h"

namespace relay {

std::size_t dtype_size(DType d) { return d == DType::F32 ? 4 : 2; }

const char* dtype_name(DType d) {
  switch (d) {
    case DType::F32: return "F32";
    case DType::F16: return "F16";
    case DType::BF16: return "BF16";
  }
  return "?";
}

float bf16_to_f32(std::uint16_t h) {
  std::uint32_t bits = static_cast<std::uint32_t>(h) << 16;
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}

float f16_to_f32(std::uint16_t h) {
  std::uint32_t sign = (h & 0x8000u) << 16;
  std::uint32_t exp = (h >> 10) & 0x1F;
  std::uint32_t mant = h & 0x3FF;
  std::uint32_t bits;
  if (exp == 0) {
    if (mant == 0) {
      bits = sign;
    } else {  // subnormal: normalize
      exp = 127 - 15 + 1;
      while ((mant & 0x400) == 0) {
        mant <<= 1;
        --exp;
      }
      mant &= 0x3FF;
      bits = sign | (exp << 23) | (mant << 13);
    }
  } else if (exp == 0x1F) {
    bits = sign | 0x7F800000u | (mant << 13);
  } else {
    bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
  }
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}

std::uint16_t f32_to_f16(float f) {
  std::uint32_t x;
  std::memcpy(&x, &f, 4);
  std::uint32_t sign = (x >> 16) & 0x8000u;
  std::uint32_t exp = (x >> 23) & 0xFF;
  std::uint32_t mant = x & 0x7FFFFFu;
  if (exp == 0xFF) return static_cast<std::uint16_t>(sign | 0x7C00u | (mant ? 0x200u : 0));
  int e = static_cast<int>(exp) - 127 + 15;
  if (e >= 0x1F) return static_cast<std::uint16_t>(sign | 0x7C00u);  // overflow: inf
  if (e <= 0) {                                                      // subnormal or zero
    if (e < -10) return static_cast<std::uint16_t>(sign);
    mant |= 0x800000u;
    int shift = 14 - e;
    std::uint32_t half = mant >> shift;
    std::uint32_t rem = mant & ((1u << shift) - 1);
    std::uint32_t mid = 1u << (shift - 1);
    if (rem > mid || (rem == mid && (half & 1))) ++half;
    return static_cast<std::uint16_t>(sign | half);
  }
  std::uint32_t half = sign | (static_cast<std::uint32_t>(e) << 10) | (mant >> 13);
  std::uint32_t rem = mant & 0x1FFF;
  if (rem > 0x1000 || (rem == 0x1000 && (half & 1))) ++half;  // may carry into the exponent, which is right
  return static_cast<std::uint16_t>(half);
}

std::int64_t TensorView::numel() const {
  std::int64_t n = 1;
  for (auto d : shape) n *= d;
  return n;
}

std::vector<float> TensorView::to_f32() const {
  std::int64_t n = numel();
  std::vector<float> out(static_cast<std::size_t>(n));
  if (dtype == DType::F32) {
    std::memcpy(out.data(), data, static_cast<std::size_t>(n) * 4);
  } else {
    const auto* p = static_cast<const std::uint16_t*>(data);
    if (dtype == DType::BF16) {
      for (std::int64_t i = 0; i < n; ++i) out[i] = bf16_to_f32(p[i]);
    } else {
      for (std::int64_t i = 0; i < n; ++i) out[i] = f16_to_f32(p[i]);
    }
  }
  return out;
}

SafeTensors::SafeTensors(const std::string& path) {
  int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) throw std::runtime_error("safetensors: cannot open " + path + ": " + std::strerror(errno));
  struct stat st;
  if (::fstat(fd, &st) != 0) {
    ::close(fd);
    throw std::runtime_error("safetensors: cannot stat " + path);
  }
  size_ = static_cast<std::size_t>(st.st_size);
  if (size_ < 8) {
    ::close(fd);
    throw std::runtime_error("safetensors: file too small: " + path);
  }
  map_ = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0);
  ::close(fd);
  if (map_ == MAP_FAILED) {
    map_ = nullptr;
    throw std::runtime_error("safetensors: mmap failed: " + path);
  }
  const auto* base = static_cast<const unsigned char*>(map_);
  std::uint64_t header_len = 0;
  for (int i = 7; i >= 0; --i) header_len = (header_len << 8) | base[i];
  if (header_len > size_ - 8) throw std::runtime_error("safetensors: bad header length in " + path);
  Json header = Json::parse(std::string_view(reinterpret_cast<const char*>(base + 8), header_len));
  const unsigned char* data = base + 8 + header_len;
  std::size_t data_size = size_ - 8 - header_len;
  for (const auto& [name, info] : header.as_object()) {
    if (name == "__metadata__") continue;
    TensorView t;
    const std::string& dt = info["dtype"].as_string();
    if (dt == "F32") t.dtype = DType::F32;
    else if (dt == "F16") t.dtype = DType::F16;
    else if (dt == "BF16") t.dtype = DType::BF16;
    else throw std::runtime_error("safetensors: unsupported dtype " + dt + " for " + name);
    for (const auto& d : info["shape"].as_array()) t.shape.push_back(d.as_int());
    const auto& off = info["data_offsets"].as_array();
    auto begin = static_cast<std::size_t>(off.at(0).as_int());
    auto end = static_cast<std::size_t>(off.at(1).as_int());
    if (end < begin || end > data_size) throw std::runtime_error("safetensors: bad offsets for " + name);
    t.data = data + begin;
    t.nbytes = end - begin;
    if (t.nbytes != static_cast<std::size_t>(t.numel()) * dtype_size(t.dtype))
      throw std::runtime_error("safetensors: size does not match shape for " + name);
    tensors_.emplace(name, std::move(t));
  }
}

SafeTensors::~SafeTensors() {
  if (map_) ::munmap(map_, size_);
}

SafeTensors::SafeTensors(SafeTensors&& o) noexcept
    : map_(o.map_), size_(o.size_), tensors_(std::move(o.tensors_)) {
  o.map_ = nullptr;
  o.size_ = 0;
}

const TensorView& SafeTensors::get(const std::string& name) const {
  auto it = tensors_.find(name);
  if (it == tensors_.end()) throw std::runtime_error("safetensors: no tensor named " + name);
  return it->second;
}

WeightFiles::WeightFiles(const std::string& dir) {
  namespace fs = std::filesystem;
  fs::path single = fs::path(dir) / "model.safetensors";
  fs::path index = fs::path(dir) / "model.safetensors.index.json";
  if (fs::exists(single)) {
    files_.emplace_back(single.string());
  } else if (fs::exists(index)) {
    std::ifstream in(index);
    std::stringstream ss;
    ss << in.rdbuf();
    Json j = Json::parse(ss.str());
    std::set<std::string> shards;
    for (const auto& [name, file] : j["weight_map"].as_object()) shards.insert(file.as_string());
    for (const auto& s : shards) files_.emplace_back((fs::path(dir) / s).string());
  } else {
    throw std::runtime_error("no model.safetensors or model.safetensors.index.json in " + dir);
  }
}

bool WeightFiles::has(const std::string& name) const {
  for (const auto& f : files_)
    if (f.has(name)) return true;
  return false;
}

const TensorView& WeightFiles::get(const std::string& name) const {
  for (const auto& f : files_)
    if (f.has(name)) return f.get(name);
  throw std::runtime_error("no tensor named " + name);
}

}  // namespace relay
