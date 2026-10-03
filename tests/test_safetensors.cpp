#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>

#include "check.h"
#include "relay/config.h"
#include "relay/json.h"
#include "relay/safetensors.h"

using namespace relay;

TEST(json_parses_nested_values_and_escapes) {
  Json j = Json::parse(R"({"a": [1, 2.5, -3e2, true, null], "s": "x\"yé😀", "o": {}})");
  CHECK_EQ(j["a"].as_array().size(), 5u);
  CHECK_EQ(j["a"].as_array()[0].as_int(), 1);
  CHECK(j["a"].as_array()[1].as_double() == 2.5);
  CHECK(j["a"].as_array()[2].as_double() == -300.0);
  CHECK(j["a"].as_array()[3].as_bool());
  CHECK(j["a"].as_array()[4].is_null());
  CHECK_EQ(j["s"].as_string(), std::string("x\"y\xC3\xA9\xF0\x9F\x98\x80"));
  CHECK(j["o"].as_object().empty());
  bool threw = false;
  try {
    Json::parse("{\"a\": 1,}");
  } catch (const JsonError&) {
    threw = true;
  }
  CHECK(threw);
}

TEST(f16_round_trips_every_value) {
  int mismatches = 0;
  for (std::uint32_t h = 0; h < 65536; ++h) {
    float f = f16_to_f32(static_cast<std::uint16_t>(h));
    if (std::isnan(f)) continue;
    if (f32_to_f16(f) != h) ++mismatches;
  }
  CHECK_EQ(mismatches, 0);
  CHECK_EQ(f32_to_f16(1.0f), 0x3C00);
  CHECK_EQ(f32_to_f16(65504.0f), 0x7BFF);
  CHECK_EQ(f32_to_f16(1e6f), 0x7C00);              // overflow to infinity
  CHECK_EQ(f32_to_f16(1.0f + 1.0f / 2048), 0x3C00);  // halfway: round to even
  CHECK(bf16_to_f32(0x3F80) == 1.0f);
}

TEST(reads_a_hand_written_file) {
  auto path = std::filesystem::temp_directory_path() / "relay_test.safetensors";
  std::string header = R"({"a":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]},)"
                       R"("b":{"dtype":"BF16","shape":[3],"data_offsets":[16,22]},"__metadata__":{"x":"y"}})";
  {
    std::ofstream f(path, std::ios::binary);
    std::uint64_t n = header.size();
    f.write(reinterpret_cast<const char*>(&n), 8);
    f << header;
    float a[4] = {1, 2, 3, 4};
    f.write(reinterpret_cast<const char*>(a), 16);
    std::uint16_t b[3] = {0x3F80, 0x4000, 0xC040};  // 1, 2, -3
    f.write(reinterpret_cast<const char*>(b), 6);
  }
  SafeTensors st(path.string());
  CHECK_EQ(st.tensors().size(), 2u);
  auto a = st.get("a").to_f32();
  CHECK(a == std::vector<float>({1, 2, 3, 4}));
  CHECK(st.get("b").dtype == DType::BF16);
  CHECK(st.get("b").to_f32() == std::vector<float>({1, 2, -3}));
  std::filesystem::remove(path);
}

TEST(reads_both_config_formats) {
  // transformers 4 style
  ModelConfig a = ModelConfig::from_json_text(R"({"model_type":"llama","hidden_size":64,"intermediate_size":128,
    "num_hidden_layers":2,"num_attention_heads":4,"num_key_value_heads":2,"vocab_size":100,
    "max_position_embeddings":128,"rope_theta":500000.0,"eos_token_id":[1,2],
    "rope_scaling":{"rope_type":"llama3","factor":32.0,"low_freq_factor":1.0,"high_freq_factor":4.0,
                    "original_max_position_embeddings":64}})");
  // transformers 5 style
  ModelConfig b = ModelConfig::from_json_text(R"({"model_type":"llama","hidden_size":64,"intermediate_size":128,
    "num_hidden_layers":2,"num_attention_heads":4,"num_key_value_heads":2,"vocab_size":100,
    "max_position_embeddings":128,"eos_token_id":2,
    "rope_parameters":{"rope_type":"llama3","rope_theta":500000.0,"factor":32.0,"low_freq_factor":1.0,
                       "high_freq_factor":4.0,"original_max_position_embeddings":64}})");
  CHECK(a.rope_theta == 500000.0 && b.rope_theta == 500000.0);
  CHECK(a.rope_scaling.type == "llama3" && b.rope_scaling.type == "llama3");
  CHECK(a.rope_inv_freq() == b.rope_inv_freq());
  CHECK_EQ(a.head_dim, 16);
  CHECK(a.eos_ids == std::vector<int>({1, 2}));
  // Llama 3 scaling leaves the highest frequency alone and divides the lowest by the factor.
  ModelConfig plain = a;
  plain.rope_scaling.type = "";
  auto s = a.rope_inv_freq(), p = plain.rope_inv_freq();
  CHECK(s.front() == p.front());
  CHECK(std::fabs(s.back() - p.back() / 32.0f) < 1e-12f);
}

RUN_TESTS()
