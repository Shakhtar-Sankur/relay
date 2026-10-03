#include "relay/config.h"

#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "relay/json.h"

namespace relay {

ModelConfig ModelConfig::from_file(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open " + path);
  std::stringstream ss;
  ss << in.rdbuf();
  return from_json_text(ss.str());
}

ModelConfig ModelConfig::from_json_text(const std::string& text) {
  Json j = Json::parse(text);
  ModelConfig c;
  c.model_type = j["model_type"].as_string();
  if (c.model_type != "llama" && c.model_type != "qwen2")
    throw std::runtime_error("unsupported model_type " + c.model_type + " (llama and qwen2 are supported)");
  c.hidden = static_cast<int>(j["hidden_size"].as_int());
  c.intermediate = static_cast<int>(j["intermediate_size"].as_int());
  c.layers = static_cast<int>(j["num_hidden_layers"].as_int());
  c.heads = static_cast<int>(j["num_attention_heads"].as_int());
  c.kv_heads = j.has("num_key_value_heads") && !j["num_key_value_heads"].is_null()
                   ? static_cast<int>(j["num_key_value_heads"].as_int())
                   : c.heads;
  c.head_dim = j.has("head_dim") && !j["head_dim"].is_null() ? static_cast<int>(j["head_dim"].as_int())
                                                             : c.hidden / c.heads;
  c.vocab = static_cast<int>(j["vocab_size"].as_int());
  c.max_position = static_cast<int>(j["max_position_embeddings"].as_int());
  if (j.has("rms_norm_eps")) c.rms_eps = j["rms_norm_eps"].as_double();
  if (j.has("tie_word_embeddings")) c.tie_embeddings = j["tie_word_embeddings"].as_bool();
  // Qwen2 always has q/k/v biases; Llama has them only with attention_bias.
  c.qkv_bias = c.model_type == "qwen2" || (j.has("attention_bias") && j["attention_bias"].as_bool());
  if (j.has("mlp_bias") && j["mlp_bias"].as_bool()) throw std::runtime_error("mlp_bias is not supported");
  if (j.has("use_sliding_window") && j["use_sliding_window"].as_bool())
    throw std::runtime_error("sliding-window attention is not supported");
  // RoPE settings: transformers 4 writes rope_theta and rope_scaling at the top level,
  // transformers 5 writes both into one rope_parameters object.
  const Json* rope = nullptr;
  if (j.has("rope_parameters") && !j["rope_parameters"].is_null()) rope = &j["rope_parameters"];
  else if (j.has("rope_scaling") && !j["rope_scaling"].is_null()) rope = &j["rope_scaling"];
  if (j.has("rope_theta")) c.rope_theta = j["rope_theta"].as_double();
  if (rope && rope->has("rope_theta")) c.rope_theta = (*rope)["rope_theta"].as_double();
  if (rope) {
    const Json& rs = *rope;
    std::string type = rs.has("rope_type") ? rs["rope_type"].as_string() : rs["type"].as_string();
    if (type == "default") {
      // no scaling
    } else if (type == "llama3") {
      c.rope_scaling.type = type;
      c.rope_scaling.factor = rs["factor"].as_double();
      c.rope_scaling.low_freq_factor = rs["low_freq_factor"].as_double();
      c.rope_scaling.high_freq_factor = rs["high_freq_factor"].as_double();
      c.rope_scaling.original_max_position = rs["original_max_position_embeddings"].as_double();
    } else {
      throw std::runtime_error("unsupported rope_scaling type " + type);
    }
  }
  if (j.has("eos_token_id") && !j["eos_token_id"].is_null()) {
    const Json& e = j["eos_token_id"];
    if (e.is_array()) {
      for (const auto& x : e.as_array()) c.eos_ids.push_back(static_cast<int>(x.as_int()));
    } else {
      c.eos_ids.push_back(static_cast<int>(e.as_int()));
    }
  }
  if (c.heads % c.kv_heads != 0) throw std::runtime_error("num_attention_heads must be a multiple of num_key_value_heads");
  if (c.head_dim % 2 != 0) throw std::runtime_error("head_dim must be even");
  return c;
}

std::vector<float> ModelConfig::rope_inv_freq() const {
  int half = head_dim / 2;
  std::vector<float> inv(half);
  for (int i = 0; i < half; ++i) {
    // transformers computes this in float32 from an int64 arange: 1 / theta^(2i/d)
    float exponent = static_cast<float>(2 * i) / static_cast<float>(head_dim);
    inv[i] = 1.0f / std::pow(static_cast<float>(rope_theta), exponent);
  }
  if (rope_scaling.type == "llama3") {
    const double pi = 3.14159265358979323846;
    double low_wavelen = rope_scaling.original_max_position / rope_scaling.low_freq_factor;
    double high_wavelen = rope_scaling.original_max_position / rope_scaling.high_freq_factor;
    for (int i = 0; i < half; ++i) {
      float f = inv[i];
      float wavelen = 2.0f * static_cast<float>(pi) / f;
      float scaled;
      if (wavelen > low_wavelen) {
        scaled = f / static_cast<float>(rope_scaling.factor);
      } else {
        scaled = f;
      }
      if (!(wavelen < high_wavelen) && !(wavelen > low_wavelen)) {  // medium frequencies: interpolate
        float smooth = static_cast<float>((rope_scaling.original_max_position / wavelen - rope_scaling.low_freq_factor) /
                                          (rope_scaling.high_freq_factor - rope_scaling.low_freq_factor));
        scaled = (1 - smooth) * f / static_cast<float>(rope_scaling.factor) + smooth * f;
      }
      inv[i] = scaled;
    }
  }
  return inv;
}

}  // namespace relay
