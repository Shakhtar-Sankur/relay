#include "relay/weights.h"

#include <filesystem>
#include <stdexcept>

#include "relay/safetensors.h"

namespace relay {
namespace {

std::vector<float> take(const WeightFiles& w, const std::string& name, std::vector<std::int64_t> shape) {
  const TensorView& t = w.get(name);
  if (t.shape != shape) {
    std::string want, got;
    for (auto d : shape) want += std::to_string(d) + ",";
    for (auto d : t.shape) got += std::to_string(d) + ",";
    throw std::runtime_error(name + ": expected shape [" + want + "] but the file has [" + got + "]");
  }
  return t.to_f32();
}

void append(std::vector<float>& dst, const std::vector<float>& src) { dst.insert(dst.end(), src.begin(), src.end()); }

}  // namespace

HostWeights HostWeights::load(const std::string& model_dir) {
  namespace fs = std::filesystem;
  HostWeights hw;
  hw.config = ModelConfig::from_file((fs::path(model_dir) / "config.json").string());
  const ModelConfig& c = hw.config;
  WeightFiles w(model_dir);
  const std::int64_t H = c.hidden, Q = c.q_dim(), KV = c.kv_dim(), I = c.intermediate, V = c.vocab;

  hw.embed = take(w, "model.embed_tokens.weight", {V, H});
  hw.final_norm = take(w, "model.norm.weight", {H});
  if (w.has("lm_head.weight") && !c.tie_embeddings) hw.lm_head = take(w, "lm_head.weight", {V, H});
  else if (!c.tie_embeddings) throw std::runtime_error("lm_head.weight is missing and the embeddings are not tied");

  hw.layers.resize(c.layers);
  for (int l = 0; l < c.layers; ++l) {
    std::string p = "model.layers." + std::to_string(l) + ".";
    LayerWeights& L = hw.layers[l];
    L.attn_norm = take(w, p + "input_layernorm.weight", {H});
    L.mlp_norm = take(w, p + "post_attention_layernorm.weight", {H});
    L.wqkv.reserve(static_cast<std::size_t>((Q + 2 * KV) * H));
    append(L.wqkv, take(w, p + "self_attn.q_proj.weight", {Q, H}));
    append(L.wqkv, take(w, p + "self_attn.k_proj.weight", {KV, H}));
    append(L.wqkv, take(w, p + "self_attn.v_proj.weight", {KV, H}));
    if (c.qkv_bias) {
      append(L.bqkv, take(w, p + "self_attn.q_proj.bias", {Q}));
      append(L.bqkv, take(w, p + "self_attn.k_proj.bias", {KV}));
      append(L.bqkv, take(w, p + "self_attn.v_proj.bias", {KV}));
    }
    L.wo = take(w, p + "self_attn.o_proj.weight", {H, Q});
    L.w_gate_up.reserve(static_cast<std::size_t>(2 * I * H));
    append(L.w_gate_up, take(w, p + "mlp.gate_proj.weight", {I, H}));
    append(L.w_gate_up, take(w, p + "mlp.up_proj.weight", {I, H}));
    L.w_down = take(w, p + "mlp.down_proj.weight", {H, I});
  }
  return hw;
}

}  // namespace relay
