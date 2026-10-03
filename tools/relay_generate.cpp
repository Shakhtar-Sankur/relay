// relay-generate: run one or more prompts (as token ids) through the engine.
//
//   relay-generate --model DIR --ids "1 2 3" [--ids "4 5"] [--max-new 32]
//                  [--backend cpu|cuda] [--temperature 0.8 --top-p 0.95 --seed 1]
//                  [--blocks 512 --block-size 16 --batch-tokens 512]
//
// Prints each request's generated ids and the throughput. Text in and out comes
// with the tokenizer in the Swift control plane (M3); scripts/tokenize.py helps until then.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "relay/backend.h"
#include "relay/engine.h"

using namespace relay;

static void usage() {
  std::fprintf(stderr,
               "usage: relay-generate --model DIR --ids \"1 2 3\" [--ids ...] [--max-new N] [--backend cpu|cuda]\n"
               "       [--temperature T] [--top-p P] [--top-k K] [--seed S] [--blocks N] [--block-size N]\n"
               "       [--batch-tokens N] [--ignore-eos]\n");
  std::exit(2);
}

int main(int argc, char** argv) {
  std::string model, backend_name = "cpu";
  std::vector<std::vector<int>> prompts;
  SamplingParams sp;
  int blocks = 512, block_size = 16, batch_tokens = 512;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) usage();
      return argv[++i];
    };
    if (a == "--model") model = next();
    else if (a == "--ids") {
      std::vector<int> ids;
      std::istringstream ss(next());
      for (int x; ss >> x;) ids.push_back(x);
      prompts.push_back(ids);
    } else if (a == "--max-new") sp.max_new_tokens = std::stoi(next());
    else if (a == "--backend") backend_name = next();
    else if (a == "--temperature") sp.temperature = std::stof(next());
    else if (a == "--top-p") sp.top_p = std::stof(next());
    else if (a == "--top-k") sp.top_k = std::stoi(next());
    else if (a == "--seed") sp.seed = std::stoull(next());
    else if (a == "--blocks") blocks = std::stoi(next());
    else if (a == "--block-size") block_size = std::stoi(next());
    else if (a == "--batch-tokens") batch_tokens = std::stoi(next());
    else if (a == "--ignore-eos") sp.ignore_eos = true;
    else usage();
  }
  if (model.empty() || prompts.empty()) usage();

  auto t0 = std::chrono::steady_clock::now();
  HostWeights w = HostWeights::load(model);
  std::unique_ptr<Backend> be;
  if (backend_name == "cpu") {
    be = make_cpu_backend(w, blocks, block_size);
  } else if (backend_name == "cuda") {
#ifdef RELAY_CUDA
    be = make_cuda_backend(w, blocks, block_size, 0, batch_tokens, blocks * block_size);
#else
    std::fprintf(stderr, "built without CUDA (configure with -DRELAY_CUDA=ON)\n");
    return 1;
#endif
  } else {
    usage();
  }
  auto t1 = std::chrono::steady_clock::now();

  Engine engine(*be, {batch_tokens, 64});
  for (std::size_t i = 0; i < prompts.size(); ++i) engine.add({i, prompts[i], sp});
  auto out = engine.run_all();
  auto t2 = std::chrono::steady_clock::now();

  std::size_t generated = 0;
  for (auto& [id, toks] : out) {
    std::cout << "request " << id << ":";
    for (int t : toks) std::cout << ' ' << t;
    std::cout << '\n';
    generated += toks.size();
  }
  double load_s = std::chrono::duration<double>(t1 - t0).count();
  double run_s = std::chrono::duration<double>(t2 - t1).count();
  std::fprintf(stderr, "%s backend: loaded in %.2f s; %zu tokens generated in %.2f s (%.1f tok/s), %llu steps, %llu preemptions\n",
               be->name().c_str(), load_s, generated, run_s, generated / run_s,
               static_cast<unsigned long long>(engine.stats().steps),
               static_cast<unsigned long long>(engine.stats().preemptions));
  return 0;
}
