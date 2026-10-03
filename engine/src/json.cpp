#include "relay/json.h"

#include <cerrno>
#include <cstdlib>

namespace relay {

class JsonParser {
 public:
  explicit JsonParser(std::string_view s) : s_(s) {}

  Json document() {
    Json v = value();
    skip_space();
    if (i_ != s_.size()) fail("trailing characters");
    return v;
  }

 private:
  [[noreturn]] void fail(const std::string& what) const {
    throw JsonError("json: " + what + " at offset " + std::to_string(i_));
  }

  void skip_space() {
    while (i_ < s_.size() && (s_[i_] == ' ' || s_[i_] == '\n' || s_[i_] == '\r' || s_[i_] == '\t')) ++i_;
  }

  char peek() {
    skip_space();
    if (i_ >= s_.size()) fail("unexpected end");
    return s_[i_];
  }

  void expect(char c) {
    if (peek() != c) fail(std::string("expected '") + c + "'");
    ++i_;
  }

  bool consume_word(std::string_view w) {
    if (s_.substr(i_, w.size()) == w) {
      i_ += w.size();
      return true;
    }
    return false;
  }

  Json value() {
    Json v;
    char c = peek();
    if (c == '{') {
      v.kind_ = Json::Kind::Object;
      ++i_;
      if (peek() == '}') {
        ++i_;
        return v;
      }
      while (true) {
        if (peek() != '"') fail("expected a key");
        std::string key = string();
        expect(':');
        v.object_[key] = value();
        char d = peek();
        ++i_;
        if (d == '}') break;
        if (d != ',') fail("expected ',' or '}'");
      }
    } else if (c == '[') {
      v.kind_ = Json::Kind::Array;
      ++i_;
      if (peek() == ']') {
        ++i_;
        return v;
      }
      while (true) {
        v.array_.push_back(value());
        char d = peek();
        ++i_;
        if (d == ']') break;
        if (d != ',') fail("expected ',' or ']'");
      }
    } else if (c == '"') {
      v.kind_ = Json::Kind::String;
      v.string_ = string();
    } else if (consume_word("true")) {
      v.kind_ = Json::Kind::Bool;
      v.bool_ = true;
    } else if (consume_word("false")) {
      v.kind_ = Json::Kind::Bool;
    } else if (consume_word("null")) {
      v.kind_ = Json::Kind::Null;
    } else {
      number(v);
    }
    return v;
  }

  void number(Json& v) {
    std::size_t start = i_;
    bool integral = true;
    if (i_ < s_.size() && s_[i_] == '-') ++i_;
    while (i_ < s_.size()) {
      char c = s_[i_];
      if (c >= '0' && c <= '9') {
        ++i_;
      } else if (c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-') {
        integral = false;
        ++i_;
      } else {
        break;
      }
    }
    if (start == i_) fail("unexpected character");
    std::string text(s_.substr(start, i_ - start));
    v.kind_ = Json::Kind::Number;
    errno = 0;
    v.number_ = std::strtod(text.c_str(), nullptr);
    if (integral) {
      v.integer_ = std::strtoll(text.c_str(), nullptr, 10);
      v.is_integer_ = errno == 0;
    }
  }

  static void append_utf8(std::string& out, std::uint32_t cp) {
    if (cp < 0x80) {
      out += static_cast<char>(cp);
    } else if (cp < 0x800) {
      out += static_cast<char>(0xC0 | (cp >> 6));
      out += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
      out += static_cast<char>(0xE0 | (cp >> 12));
      out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
      out += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
      out += static_cast<char>(0xF0 | (cp >> 18));
      out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
      out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
      out += static_cast<char>(0x80 | (cp & 0x3F));
    }
  }

  std::uint32_t hex4() {
    if (i_ + 4 > s_.size()) fail("short \\u escape");
    std::uint32_t v = 0;
    for (int k = 0; k < 4; ++k) {
      char c = s_[i_++];
      v <<= 4;
      if (c >= '0' && c <= '9') v |= c - '0';
      else if (c >= 'a' && c <= 'f') v |= c - 'a' + 10;
      else if (c >= 'A' && c <= 'F') v |= c - 'A' + 10;
      else fail("bad \\u escape");
    }
    return v;
  }

  std::string string() {
    expect('"');
    std::string out;
    while (true) {
      if (i_ >= s_.size()) fail("unterminated string");
      char c = s_[i_++];
      if (c == '"') break;
      if (c != '\\') {
        out += c;
        continue;
      }
      if (i_ >= s_.size()) fail("unterminated escape");
      char e = s_[i_++];
      switch (e) {
        case '"': out += '"'; break;
        case '\\': out += '\\'; break;
        case '/': out += '/'; break;
        case 'b': out += '\b'; break;
        case 'f': out += '\f'; break;
        case 'n': out += '\n'; break;
        case 'r': out += '\r'; break;
        case 't': out += '\t'; break;
        case 'u': {
          std::uint32_t cp = hex4();
          if (cp >= 0xD800 && cp < 0xDC00 && s_.substr(i_, 2) == "\\u") {
            i_ += 2;
            std::uint32_t lo = hex4();
            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
          }
          append_utf8(out, cp);
          break;
        }
        default: fail("bad escape");
      }
    }
    return out;
  }

  std::string_view s_;
  std::size_t i_ = 0;
};

Json Json::parse(std::string_view text) { return JsonParser(text).document(); }

bool Json::as_bool() const {
  if (kind_ != Kind::Bool) throw JsonError("json: not a bool");
  return bool_;
}

double Json::as_double() const {
  if (kind_ != Kind::Number) throw JsonError("json: not a number");
  return number_;
}

std::int64_t Json::as_int() const {
  if (kind_ != Kind::Number || !is_integer_) throw JsonError("json: not an integer");
  return integer_;
}

const std::string& Json::as_string() const {
  if (kind_ != Kind::String) throw JsonError("json: not a string");
  return string_;
}

const std::vector<Json>& Json::as_array() const {
  if (kind_ != Kind::Array) throw JsonError("json: not an array");
  return array_;
}

const std::map<std::string, Json>& Json::as_object() const {
  if (kind_ != Kind::Object) throw JsonError("json: not an object");
  return object_;
}

bool Json::has(const std::string& key) const {
  return kind_ == Kind::Object && object_.count(key) != 0;
}

const Json& Json::operator[](const std::string& key) const {
  auto it = as_object().find(key);
  if (it == object_.end()) throw JsonError("json: missing key '" + key + "'");
  return it->second;
}

}  // namespace relay
