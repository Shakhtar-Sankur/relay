// relay-worker: one model on one backend, serving the control plane.
//
//   relay-worker --model DIR --role both|prefill|decode [--backend cpu|cuda] [--device 0]
//                [--host 127.0.0.1] [--control-port 0] [--kv-port 0] [--blocks 512]
//                [--block-size 16] [--batch-tokens 512] [--no-prefix-cache]
//
// Prints one line "relay-worker <role> control=<port> kv=<port>" once it is listening, so a
// launcher started with port 0 can read the ports it got.
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <string>
#ifdef __linux__
#include <sys/prctl.h>
#endif

#include "relay/backend.h"
#include "relay/worker.h"

using namespace relay;

static void usage() {
  std::fprintf(stderr,
               "usage: relay-worker --model DIR --role both|prefill|decode [--backend cpu|cuda] [--device N]\n"
               "       [--host H] [--control-port P] [--kv-port P] [--blocks N] [--block-size N] [--batch-tokens N]\n"
               "       [--no-prefix-cache]\n");
  std::exit(2);
}

int main(int argc, char** argv) {
  // A launcher may start us from a thread that blocks signals (Foundation's Process does),
  // and the mask is inherited: unblock everything so SIGTERM stops the worker.
  sigset_t all;
  sigfillset(&all);
  sigprocmask(SIG_UNBLOCK, &all, nullptr);
#ifdef __linux__
  prctl(PR_SET_PDEATHSIG, SIGTERM);  // and stop when the launcher dies
#endif
  std::string model, backend = "cpu", role = "both";
  int blocks = 512, block_size = 16, device = 0;
  WorkerOptions o;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) usage();
      return argv[++i];
    };
    if (a == "--model") model = next();
    else if (a == "--role") role = next();
    else if (a == "--backend") backend = next();
    else if (a == "--device") device = std::stoi(next());
    else if (a == "--host") o.host = next();
    else if (a == "--control-port") o.control_port = std::stoi(next());
    else if (a == "--kv-port") o.kv_port = std::stoi(next());
    else if (a == "--blocks") blocks = std::stoi(next());
    else if (a == "--block-size") block_size = std::stoi(next());
    else if (a == "--batch-tokens") o.max_batch_tokens = std::stoi(next());
    else if (a == "--no-prefix-cache") o.prefix_caching = false;
    else usage();
  }
  if (model.empty()) usage();
  if (role == "both") o.role = WorkerRole::Both;
  else if (role == "prefill") o.role = WorkerRole::Prefill;
  else if (role == "decode") o.role = WorkerRole::Decode;
  else usage();
  try {
    HostWeights w = HostWeights::load(model);
    std::unique_ptr<Backend> be;
    if (backend == "cuda") {
#ifdef RELAY_CUDA
      be = make_cuda_backend(w, blocks, block_size, device, o.max_batch_tokens, blocks * block_size);
#else
      std::fprintf(stderr, "relay-worker: built without CUDA\n");
      return 1;
#endif
    } else {
      be = make_cpu_backend(w, blocks, block_size);
    }
    Worker worker(*be, o);
    std::printf("relay-worker %s control=%d kv=%d\n", role.c_str(), worker.control_port(), worker.kv_port());
    std::fflush(stdout);
    worker.serve();
  } catch (const std::exception& e) {
    std::fprintf(stderr, "relay-worker: %s\n", e.what());
    return 1;
  }
  return 0;
}
