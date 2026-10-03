// A small JSON reader: enough for config.json and the safetensors header.
// Numbers are kept as double (and as int64 when they are integers), strings are
// UTF-8 with escapes decoded, objects keep their keys in a map.
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace relay {

struct JsonError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

class Json {
 public:
  enum class Kind { Null, Bool, Number, String, Array, Object };

  static Json parse(std::string_view text);

  Kind kind() const { return kind_; }
  bool is_null() const { return kind_ == Kind::Null; }
  bool is_number() const { return kind_ == Kind::Number; }
  bool is_string() const { return kind_ == Kind::String; }
  bool is_object() const { return kind_ == Kind::Object; }
  bool is_array() const { return kind_ == Kind::Array; }

  bool as_bool() const;
  double as_double() const;
  std::int64_t as_int() const;
  const std::string& as_string() const;
  const std::vector<Json>& as_array() const;
  const std::map<std::string, Json>& as_object() const;

  bool has(const std::string& key) const;
  // Throws JsonError if this is not an object or the key is missing.
  const Json& operator[](const std::string& key) const;

 private:
  friend class JsonParser;
  Kind kind_ = Kind::Null;
  bool bool_ = false;
  double number_ = 0;
  std::int64_t integer_ = 0;
  bool is_integer_ = false;
  std::string string_;
  std::vector<Json> array_;
  std::map<std::string, Json> object_;
};

}  // namespace relay
