// A minimal test harness: TEST(name) { ... CHECK(cond); } and main() from RUN_TESTS().
#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "relay/backend.h"

namespace check {

struct Case {
  const char* name;
  std::function<void()> fn;
};

inline std::vector<Case>& cases() {
  static std::vector<Case> c;
  return c;
}

struct Register {
  Register(const char* name, std::function<void()> fn) { cases().push_back({name, std::move(fn)}); }
};

inline int failures = 0;

inline void fail(const char* file, int line, const std::string& what) {
  std::fprintf(stderr, "  FAILED %s:%d: %s\n", file, line, what.c_str());
  ++failures;
}

inline int run_all() {
  int failed_cases = 0;
  for (auto& c : cases()) {
    int before = failures;
    std::fprintf(stderr, "[ RUN  ] %s\n", c.name);
    try {
      c.fn();
    } catch (const std::exception& e) {
      fail(__FILE__, __LINE__, std::string("exception: ") + e.what());
    }
    bool ok = failures == before;
    if (!ok) ++failed_cases;
    std::fprintf(stderr, "[ %s ] %s\n", ok ? " OK " : "FAIL", c.name);
  }
  std::fprintf(stderr, "%zu tests, %d failed\n", cases().size(), failed_cases);
  return failed_cases ? 1 : 0;
}

// Which backend the model tests run on: RELAY_TEST_BACKEND=cpu (default) or cuda.
inline std::string test_backend() {
  const char* b = std::getenv("RELAY_TEST_BACKEND");
  return b ? b : "cpu";
}

inline std::unique_ptr<relay::Backend> make_backend(const relay::HostWeights& w, int blocks, int block_size,
                                                    int max_batch_tokens = 1024) {
  if (test_backend() == "cuda") {
#ifdef RELAY_CUDA
    return relay::make_cuda_backend(w, blocks, block_size, 0, max_batch_tokens, blocks * block_size);
#else
    std::fprintf(stderr, "RELAY_TEST_BACKEND=cuda but built without CUDA\n");
    std::exit(1);
#endif
  }
  return relay::make_cpu_backend(w, blocks, block_size);
}

// The CPU backend is the float32 reference: it must match Hugging Face closely and
// itself exactly. The CUDA backend stores weights and the KV cache in fp16 and its
// GEMMs depend on batch shape, so it gets a looser tolerance and no bitwise checks.
inline bool exact_backend() { return test_backend() == "cpu"; }

}  // namespace check

#define TEST(name)                                                  \
  static void name();                                               \
  static check::Register name##_registered(#name, name);            \
  static void name()

#define CHECK(cond)                                                       \
  do {                                                                    \
    if (!(cond)) check::fail(__FILE__, __LINE__, "CHECK(" #cond ")");     \
  } while (0)

#define CHECK_EQ(a, b)                                                                   \
  do {                                                                                   \
    auto _va = (a);                                                                      \
    auto _vb = (b);                                                                      \
    if (!(_va == _vb)) {                                                                 \
      std::ostringstream _os;                                                            \
      _os << #a " == " #b " (" << _va << " vs " << _vb << ")";                           \
      check::fail(__FILE__, __LINE__, _os.str());                                        \
    }                                                                                    \
  } while (0)

#define RUN_TESTS() \
  int main() { return check::run_all(); }
