// Model hyperparameters, read from a Hugging Face config.json.
// Supported: Llama-architecture models (Llama 2/3/3.1/3.2, TinyLlama, SmolLM2)
// and Qwen2 (the same architecture with biases on the q/k/v projections).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace relay {

struct RopeScaling {
  std::string type;  // "" (none) or "llama3"
  double factor = 1;
  double low_freq_factor = 1;
  double high_freq_factor = 4;
  double original_max_position = 8192;
};

struct ModelConfig {
  std::string model_type;
  int hidden = 0;
  int intermediate = 0;
  int layers = 0;
  int heads = 0;
  int kv_heads = 0;
  int head_dim = 0;
  int vocab = 0;
  int max_position = 0;
  double rms_eps = 1e-5;
  double rope_theta = 10000;
  RopeScaling rope_scaling;
  bool tie_embeddings = false;
  bool qkv_bias = false;
  std::vector<int> eos_ids;

  static ModelConfig from_file(const std::string& path);
  static ModelConfig from_json_text(const std::string& text);

  int q_dim() const { return heads * head_dim; }
  int kv_dim() const { return kv_heads * head_dim; }

  // The inverse rotary frequencies, head_dim / 2 of them, with Llama 3's
  // long-context scaling applied when the config asks for it. Same formula as
  // transformers' _compute_llama3_parameters.
  std::vector<float> rope_inv_freq() const;
};

}  // namespace relay
