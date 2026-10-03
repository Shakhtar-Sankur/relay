// relay's logits against Hugging Face transformers' on the same weights and tokens.
//
// Fixtures: tests/fixtures/<model>/relay-reference.json, made by scripts/make_fixtures.py.
// Real models: set RELAY_REFERENCE_MODELS=dir1:dir2 (each with a relay-reference.json).
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>

#include "check.h"
#include "relay/engine.h"
#include "relay/json.h"
#include "relay/sampler.h"

using namespace relay;
namespace fs = std::filesystem;

namespace {

struct Reference {
  std::vector<int> prompt, generated;
  std::vector<std::vector<float>> prompt_logits, step_logits;
  bool last_only = false;
};

std::vector<float> floats(const Json& a) {
  std::vector<float> v;
  for (const auto& x : a.as_array()) v.push_back(static_cast<float>(x.as_double()));
  return v;
}

Reference load_reference(const fs::path& dir) {
  std::ifstream in(dir / "relay-reference.json");
  std::stringstream ss;
  ss << in.rdbuf();
  Json j = Json::parse(ss.str());
  Reference r;
  for (const auto& x : j["prompt"].as_array()) r.prompt.push_back(static_cast<int>(x.as_int()));
  for (const auto& x : j["generated"].as_array()) r.generated.push_back(static_cast<int>(x.as_int()));
  for (const auto& row : j["prompt_logits"].as_array()) r.prompt_logits.push_back(floats(row));
  for (const auto& row : j["step_logits"].as_array()) r.step_logits.push_back(floats(row));
  r.last_only = j.has("prompt_logits_last_only") && j["prompt_logits_last_only"].as_bool();
  return r;
}

// Largest |a - b| relative to the largest |b| in the row.
double rel_err(const float* a, const std::vector<float>& b) {
  double diff = 0, scale = 1e-6;
  for (std::size_t i = 0; i < b.size(); ++i) {
    diff = std::max(diff, std::fabs(static_cast<double>(a[i]) - b[i]));
    scale = std::max(scale, std::fabs(static_cast<double>(b[i])));
  }
  return diff / scale;
}

std::vector<fs::path> model_dirs() {
  std::vector<fs::path> dirs;
  for (const auto& e : fs::directory_iterator(RELAY_FIXTURES))
    if (fs::exists(e.path() / "relay-reference.json")) dirs.push_back(e.path());
  std::sort(dirs.begin(), dirs.end());
  if (const char* extra = std::getenv("RELAY_REFERENCE_MODELS")) {
    std::stringstream ss(extra);
    for (std::string d; std::getline(ss, d, ':');)
      if (!d.empty()) dirs.push_back(d);
  }
  return dirs;
}

// float32 on CPU should agree with transformers to float32 rounding; fp16 on GPU less so.
double tolerance() { return check::exact_backend() ? 1e-4 : 2e-2; }

}  // namespace

TEST(prompt_logits_match_transformers) {
  for (const fs::path& dir : model_dirs()) {
    HostWeights w = HostWeights::load(dir.string());
    Reference ref = load_reference(dir);
    auto be = check::make_backend(w, 64, 16);
    ForwardBatch b;
    std::vector<int> table(be->kv_layout().blocks_for(static_cast<int>(ref.prompt.size())));
    for (std::size_t i = 0; i < table.size(); ++i) table[i] = static_cast<int>(i);
    b.seqs.push_back({ref.prompt, 0, table, true});
    b.all_logits = !ref.last_only;
    std::vector<float> lg = be->forward(b);
    const int V = w.config.vocab;
    double worst = 0;
    std::size_t rows = ref.prompt_logits.size();
    for (std::size_t r = 0; r < rows; ++r) worst = std::max(worst, rel_err(lg.data() + r * V, ref.prompt_logits[r]));
    std::fprintf(stderr, "  %s (%s): %zu prompt rows, largest relative error %.2e\n", dir.filename().c_str(),
                 be->name().c_str(), rows, worst);
    CHECK(worst < tolerance());
  }
}

TEST(greedy_generation_matches_transformers) {
  for (const fs::path& dir : model_dirs()) {
    HostWeights w = HostWeights::load(dir.string());
    Reference ref = load_reference(dir);
    auto be = check::make_backend(w, 64, 16);
    Engine engine(*be, {512, 8});
    std::vector<std::vector<float>> seen;
    engine.on_logits = [&](std::uint64_t, int, const float* lg) { seen.emplace_back(lg, lg + w.config.vocab); };
    SamplingParams sp;
    sp.max_new_tokens = static_cast<int>(ref.generated.size());
    sp.ignore_eos = true;
    engine.add({1, ref.prompt, sp});
    auto out = engine.run_all();
    CHECK(out[1] == ref.generated);
    double worst = 0;
    for (std::size_t i = 0; i < seen.size() && i < ref.step_logits.size(); ++i)
      worst = std::max(worst, rel_err(seen[i].data(), ref.step_logits[i]));
    std::fprintf(stderr, "  %s (%s): %zu generated tokens %s, largest relative error %.2e\n", dir.filename().c_str(),
                 be->name().c_str(), ref.generated.size(), out[1] == ref.generated ? "identical" : "DIFFERENT", worst);
    CHECK(worst < tolerance());
  }
}

RUN_TESTS()
