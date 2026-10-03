// A backend runs the model's forward pass over one batch and owns the KV cache memory.
//
// A batch is a list of sequence chunks. Each chunk is some consecutive tokens of one
// sequence, starting at position start_pos: a whole prompt, a piece of a long prompt
// (chunked prefill), or the single newest token while decoding. Chunks of different
// sequences and lengths run in the same forward pass: the tokens are concatenated,
// every matrix multiply runs over all of them at once, and only attention looks at
// which sequence a token belongs to (through its block table).
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "relay/config.h"
#include "relay/kv_cache.h"
#include "relay/weights.h"

namespace relay {

struct SeqChunk {
  std::vector<int> tokens;       // the new tokens
  int start_pos = 0;             // position of tokens[0]; positions before it are already in the cache
  std::vector<int> block_table;  // must cover positions [0, start_pos + tokens.size())
  bool want_logits = true;       // logits for the chunk's last token (or all, see ForwardBatch)
};

struct ForwardBatch {
  std::vector<SeqChunk> seqs;
  bool all_logits = false;  // logits for every token instead of each chunk's last (tests)
};

// Told during a forward pass as each layer's K and V for the whole batch are in the
// cache (for the CUDA backend: enqueued, see Backend::wait_kv_written). The KV
// transfer engine uses it to send layer l while the backend computes layer l + 1.
class LayerObserver {
 public:
  virtual ~LayerObserver() = default;
  virtual void kv_written(int layer) = 0;
};

class Backend {
 public:
  virtual ~Backend() = default;
  virtual const ModelConfig& config() const = 0;
  virtual const KVLayout& kv_layout() const = 0;
  virtual std::string name() const = 0;

  // Runs the batch, writing each token's K and V into the cache, and returns logits:
  // [rows, vocab] float32, one row per chunk with want_logits (or per token with all_logits).
  virtual std::vector<float> forward(const ForwardBatch& batch, LayerObserver* observer = nullptr) = 0;

  // One layer's K and V for one block, as bytes in the cache's own dtype.
  // These are what the KV transfer engine moves between workers.
  virtual std::size_t kv_block_bytes() const = 0;
  virtual void read_kv_block(int layer, int block, void* k_out, void* v_out) = 0;
  virtual void write_kv_block(int layer, int block, const void* k_in, const void* v_in) = 0;

  // Several blocks of one layer at once: out holds all K blocks then all V blocks,
  // in the order given. Safe to call from another thread while forward() runs, as
  // long as forward() is not writing those blocks of that layer.
  virtual void read_kv_layer(int layer, const std::vector<int>& blocks, void* out);
  virtual void write_kv_layer(int layer, const std::vector<int>& blocks, const void* in);

  // Where one layer's K (or V) of a block lives in host memory, or nullptr if the cache
  // is not in host memory. Lets the transfer engine send and receive without a copy.
  virtual void* kv_host_ptr(int layer, int block, bool value) { return nullptr; }

  // Blocks until the K/V writes of `layer` from the latest forward() are complete.
  // A no-op where kv_written() is only called after the writes are done (CPU).
  virtual void wait_kv_written(int layer) {}
};

std::unique_ptr<Backend> make_cpu_backend(const HostWeights& weights, int num_blocks, int block_size);
#ifdef RELAY_CUDA
std::unique_ptr<Backend> make_cuda_backend(const HostWeights& weights, int num_blocks, int block_size, int device,
                                           int max_batch_tokens, int max_context);
#endif

}  // namespace relay
