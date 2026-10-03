// A model's weights in float32 host memory, with the projections that are always
// applied to the same input fused into one matrix: q/k/v into wqkv, gate/up into
// w_gate_up. Each backend builds its own layout (fp32 on CPU, fp16 on CUDA) from this.
#pragma once

#include <string>
#include <vector>

#include "relay/config.h"

namespace relay {

struct LayerWeights {
  std::vector<float> attn_norm;  // [hidden]
  std::vector<float> wqkv;       // [q_dim + 2 * kv_dim, hidden]
  std::vector<float> bqkv;       // [q_dim + 2 * kv_dim] or empty
  std::vector<float> wo;         // [hidden, q_dim]
  std::vector<float> mlp_norm;   // [hidden]
  std::vector<float> w_gate_up;  // [2 * intermediate, hidden]: gate rows, then up rows
  std::vector<float> w_down;     // [hidden, intermediate]
};

struct HostWeights {
  ModelConfig config;
  std::vector<float> embed;       // [vocab, hidden]
  std::vector<LayerWeights> layers;
  std::vector<float> final_norm;  // [hidden]
  std::vector<float> lm_head;     // [vocab, hidden]; empty when tied to embed

  const std::vector<float>& lm_head_or_embed() const { return lm_head.empty() ? embed : lm_head; }

  // Loads config.json and the safetensors weights of a Hugging Face model directory.
  static HostWeights load(const std::string& model_dir);
};

}  // namespace relay
