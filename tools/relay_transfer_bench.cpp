// relay-transfer-bench: how fast the KV transfer engine moves data, and how much of a
// prompt's KV transfer it hides behind the prefill forward pass.
//
//   relay-transfer-bench raw [--transport tcp|shm]
//       A child process receives frames from the parent; GB/s per message size.
//
//   relay-transfer-bench stream --model DIR [--backend cpu|cuda] [--transport tcp|shm]
//                               [--prompt 512] [--repeats 5]
//       One request prefilled on one backend and handed to a decode engine on another,
//       with the KV cache streamed layer by layer during the forward pass, and again with
//       the KV cache sent only after the forward pass. Reports, as medians:
//         forward      the prefill forward pass
//         ready        from the start of prefill to the request being runnable on the decode side
//         exposed      ready - forward: transfer time left on the critical path
//         hidden       1 - exposed(streaming) / exposed(after forward)
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "relay/backend.h"
#include "relay/engine.h"
#include "relay/kv_transfer.h"
#include "relay/transport.h"

using namespace relay;
using Clock = std::chrono::steady_clock;

namespace {

double secs(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double>(b - a).count(); }

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

std::string arg(int argc, char** argv, const std::string& name, const std::string& def) {
  for (int i = 1; i + 1 < argc; ++i)
    if (argv[i] == name) return argv[i + 1];
  return def;
}

int raw(const std::string& transport) {
  const std::size_t sizes[] = {4 << 10, 64 << 10, 1 << 20, 16 << 20};
  std::string name = "/relay-bench-" + std::to_string(::getpid());
  std::unique_ptr<TcpListener> listener;
  if (transport == "tcp") listener = std::make_unique<TcpListener>(0);
  int port = listener ? listener->port() : 0;
  std::unique_ptr<Connection> parent;
  if (transport == "shm") parent = shm_create(name, 64 << 20);
  pid_t pid = ::fork();
  if (pid == 0) {  // receiver
    auto c = transport == "tcp" ? tcp_connect("127.0.0.1", port) : shm_open(name);
    std::vector<unsigned char> buf(16 << 20);
    for (std::size_t s : sizes) {
      std::uint64_t n;
      c->recv(&n, 8);
      for (std::uint64_t i = 0; i < n; ++i) c->recv(buf.data(), s);
      char ack = 1;
      c->send(&ack, 1);
    }
    std::_Exit(0);
  }
  if (transport == "tcp") parent = listener->accept();
  std::vector<unsigned char> data(16 << 20, 7);
  std::printf("transport %s, sender and receiver in separate processes\n", transport.c_str());
  std::printf("%10s %10s %10s\n", "message", "messages", "GB/s");
  for (std::size_t s : sizes) {
    std::uint64_t n = std::max<std::uint64_t>(8, (std::uint64_t(1) << 30) / s);  // about 1 GiB per size
    parent->send(&n, 8);
    auto t0 = Clock::now();
    for (std::uint64_t i = 0; i < n; ++i) parent->send(data.data(), s);
    char ack;
    parent->recv(&ack, 1);
    double gbs = static_cast<double>(n * s) / secs(t0, Clock::now()) / 1e9;
    std::printf("%9zuK %10llu %10.2f\n", s >> 10, static_cast<unsigned long long>(n), gbs);
  }
  ::waitpid(pid, nullptr, 0);
  return 0;
}

struct Pair {
  std::unique_ptr<Connection> tx, rx;
};

Pair connect(const std::string& transport) {
  Pair p;
  if (transport == "shm") {
    std::string name = "/relay-bench-" + std::to_string(::getpid()) + "-" + std::to_string(std::rand());
    p.tx = shm_create(name, 256 << 20);
    p.rx = shm_open(name);
  } else {
    TcpListener l(0);
    std::thread t([&] { p.tx = tcp_connect("127.0.0.1", l.port()); });
    p.rx = l.accept();
    t.join();
  }
  return p;
}

std::unique_ptr<Backend> make(const HostWeights& w, const std::string& backend, int blocks, int max_tokens) {
#ifdef RELAY_CUDA
  if (backend == "cuda") return make_cuda_backend(w, blocks, 16, 0, max_tokens, blocks * 16);
#endif
  if (backend != "cpu") {
    std::fprintf(stderr, "backend %s not available in this build\n", backend.c_str());
    std::exit(1);
  }
  return make_cpu_backend(w, blocks, 16);
}

int stream(const std::string& model, const std::string& backend, const std::string& transport, int prompt_len,
           int repeats) {
  HostWeights w = HostWeights::load(model);
  int blocks = (prompt_len + 15) / 16 + 8;
  auto pb = make(w, backend, blocks, prompt_len);
  auto db = make(w, backend, blocks, prompt_len);
  std::mt19937 rng(3);
  Request r;
  r.prompt.resize(prompt_len);
  for (int& t : r.prompt) t = static_cast<int>(rng() % w.config.vocab);
  r.params.max_new_tokens = 2;
  r.params.ignore_eos = true;

  std::printf("model %s, %s backend, %s transport, %d-token prompt, %d layers, %.2f MB of KV cache\n", model.c_str(),
              backend.c_str(), transport.c_str(), prompt_len, w.config.layers,
              2.0 * w.config.layers * ((prompt_len + 15) / 16) * pb->kv_block_bytes() / 1e6);
  double kv_bytes = 0;
  std::vector<double> fwd[2], ready[2];
  for (int rep = 0; rep < repeats + 1; ++rep) {  // the first round warms up and is not counted
    for (int mode = 0; mode < 2; ++mode) {     // 0: streaming, 1: after the forward pass
      Pair p = connect(transport);
      Engine decode(*db, {prompt_len, 8});
      std::mutex mu;
      std::condition_variable cv;
      bool is_ready = false;
      Clock::time_point t_ready;
      KVReceiver rx(decode, *db, *p.rx, [&](std::uint64_t, int) {
        std::lock_guard<std::mutex> lock(mu);
        t_ready = Clock::now();
        is_ready = true;
        cv.notify_all();
      });
      KVSender tx(*pb, *p.tx);
      tx.flush();  // Hello out of the way
      PrefillWorker pw(*pb, tx, prompt_len);
      pw.stream_layers = mode == 0;
      r.id = static_cast<std::uint64_t>(rep * 2 + mode + 1);
      pw.add(r);
      auto t0 = Clock::now();
      pw.step();
      auto t1 = Clock::now();
      {
        std::unique_lock<std::mutex> lock(mu);
        cv.wait(lock, [&] { return is_ready; });
      }
      tx.flush();
      kv_bytes = static_cast<double>(tx.stats().kv_bytes);
      if (rep > 0) {
        fwd[mode].push_back(secs(t0, t1));
        ready[mode].push_back(secs(t0, t_ready));
      }
      while (decode.has_work()) decode.step();
      p.tx->close();
      rx.join();
    }
  }
  double f[2], rd[2], ex[2];
  for (int m = 0; m < 2; ++m) {
    f[m] = median(fwd[m]);
    rd[m] = median(ready[m]);
    ex[m] = std::max(0.0, rd[m] - f[m]);
  }
  std::printf("%-22s %10s %10s %10s\n", "", "forward", "ready", "exposed");
  std::printf("%-22s %8.2fms %8.2fms %8.2fms\n", "send after forward", f[1] * 1e3, rd[1] * 1e3, ex[1] * 1e3);
  std::printf("%-22s %8.2fms %8.2fms %8.2fms\n", "stream layer by layer", f[0] * 1e3, rd[0] * 1e3, ex[0] * 1e3);
  std::printf("transfer: %.2f MB; after-forward exposed time is %.2f GB/s effective\n", kv_bytes / 1e6,
              ex[1] > 0 ? kv_bytes / ex[1] / 1e9 : 0.0);
  std::printf("hidden by streaming: %.0f%% of the exposed transfer time\n", ex[1] > 0 ? 100.0 * (1 - ex[0] / ex[1]) : 0.0);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: relay-transfer-bench raw|stream [options]\n");
    return 2;
  }
  std::string mode = argv[1];
  std::string transport = arg(argc, argv, "--transport", "tcp");
  if (mode == "raw") return raw(transport);
  if (mode == "stream")
    return stream(arg(argc, argv, "--model", ""), arg(argc, argv, "--backend", "cpu"), transport,
                  std::stoi(arg(argc, argv, "--prompt", "512")), std::stoi(arg(argc, argv, "--repeats", "5")));
  std::fprintf(stderr, "unknown mode %s\n", mode.c_str());
  return 2;
}
