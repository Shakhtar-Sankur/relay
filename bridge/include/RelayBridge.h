// The C++ surface the Swift control plane calls, through Swift's C++ interoperability.
//
// Swift imports this header directly: the structs become Swift value types, std::string
// and std::vector<int32_t> map to their Swift overlays, and LocalWorker is imported as a
// Swift class whose lifetime is managed by the retain/release functions below
// (SWIFT_SHARED_REFERENCE). The engine's own headers stay behind a pimpl, so Swift never
// has to import std::function, unique_ptr or the engine's templates. No C++ exception
// crosses into Swift: every call reports failure through its return value.
#pragma once

// <swift/bridging> ships with the Swift toolchain: present when Swift imports this
// header, possibly absent when a plain C++ compiler builds RelayBridge.cpp.
// Without it (the include path is not always passed), clang takes the same attributes
// spelled out; other compilers ignore them.
#if __has_include(<swift/bridging>)
#include <swift/bridging>
#elif defined(__clang__)
#define SWIFT_SHARED_REFERENCE(_retain, _release)                                        \
  __attribute__((swift_attr("import_reference"))) __attribute__((swift_attr("retain:" #_retain))) \
  __attribute__((swift_attr("release:" #_release)))
#define SWIFT_RETURNS_RETAINED __attribute__((swift_attr("returns_retained")))
#else
#define SWIFT_SHARED_REFERENCE(_retain, _release)
#define SWIFT_RETURNS_RETAINED
#endif

#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

// Swift finds a reference type's retain/release functions by name at global scope.
namespace relaybridge {
class LocalWorker;
}
void relaybridge_retain(relaybridge::LocalWorker* w);
void relaybridge_release(relaybridge::LocalWorker* w);

namespace relaybridge {

// Names Swift can spell for the vector types used below.
using TokenVector = std::vector<int32_t>;

struct SamplingOptions {
  float temperature = 0.0f;
  float topP = 1.0f;
  int32_t topK = 0;
  int32_t maxNewTokens = 16;
  uint64_t seed = 0;
  bool ignoreEos = false;
};

// finish: 0 = still generating, 1 = hit max tokens, 2 = hit an end-of-sequence token
struct TokenEvent {
  uint64_t request = 0;
  int32_t token = 0;
  int32_t index = 0;
  int32_t finish = 0;
};

struct StepResult {
  std::vector<TokenEvent> events;
  std::string error;  // empty on success
};

struct WorkerOptions {
  std::string modelDir;
  int32_t blocks = 512;
  int32_t blockSize = 16;
  int32_t maxBatchTokens = 512;
  int32_t maxSeqs = 64;
};

struct ModelInfo {
  std::string modelType;
  int32_t vocab = 0;
  int32_t layers = 0;
  int32_t hidden = 0;
  int32_t maxPosition = 0;
  std::vector<int32_t> eosIds;
};

// A model, its KV cache and a continuous-batching engine in this process (CPU backend).
class LocalWorker final {
 public:
  // nullptr on failure; lastError() then says why.
  static LocalWorker* create(const WorkerOptions& options) SWIFT_RETURNS_RETAINED;
  static std::string lastError();

  // Empty string on success, otherwise the reason the request was refused.
  std::string add(uint64_t request, const TokenVector& prompt, const SamplingOptions& sampling);
  // Stops a request and frees its cache; false if it is not (or no longer) in the engine.
  bool cancel(uint64_t request);
  // One forward pass over everything running.
  StepResult step();
  bool hasWork() const;
  ModelInfo info() const;
  int32_t freeBlocks() const;

 private:
  struct Impl;
  explicit LocalWorker(Impl* impl);
  ~LocalWorker();
  friend void ::relaybridge_retain(LocalWorker*);
  friend void ::relaybridge_release(LocalWorker*);
  std::atomic<int> refs_{1};
  Impl* impl_;
} SWIFT_SHARED_REFERENCE(relaybridge_retain, relaybridge_release);

}  // namespace relaybridge
