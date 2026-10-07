/* relay's C interface, for Python (ctypes) and any other language with a C FFI.
 *
 * Load a model, read and write its weights (an RL trainer copies its parameters into
 * them after each update), and run an engine that generates with per-token
 * log-probabilities. Functions that can fail return 0 or a valid pointer on success
 * and -1 or NULL on failure; relay_last_error() then says why (per thread).
 *
 * The weights are the model's float32 host copy, in relay's fused layout (see
 * relay/weights.h): q/k/v projections stacked into wqkv, gate and up into w_gate_up.
 * After writing them, call relay_engine_reload_weights() before the next rollout: the
 * CUDA backend keeps its own fp16 copy and uploads it again, and every backend drops
 * its prefix cache, whose KV entries were computed with the old weights. (The CPU
 * backend reads these host weights in place, so on CPU a write is visible to the next
 * forward pass even before the reload; never write while a rollout is running.)
 */
#ifndef RELAY_C_API_H
#define RELAY_C_API_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct relay_model relay_model;
typedef struct relay_engine relay_engine;

#define RELAY_MAX_EOS 8

typedef struct {
  int hidden, intermediate, layers, heads, kv_heads, head_dim, vocab, max_position;
  double rms_eps, rope_theta;
  int tie_embeddings, qkv_bias;
  int num_eos;
  int eos_ids[RELAY_MAX_EOS];
} relay_config;

typedef struct {
  float temperature; /* 0: greedy */
  float top_p;
  int top_k; /* 0: no limit */
  int max_new_tokens;
  uint64_t seed;
  int ignore_eos;
} relay_sampling;

enum { RELAY_FINISH_NONE = 0, RELAY_FINISH_LENGTH = 1, RELAY_FINISH_STOP = 2 };

typedef struct {
  uint64_t id;
  int token;
  int index; /* 0 for the first generated token */
  int finish;
  float logprob; /* log softmax(logits / temperature)[token], whole vocabulary */
} relay_event;

typedef struct {
  uint64_t steps, forward_tokens, preemptions, prompt_tokens, prefix_hit_tokens;
} relay_stats;

const char* relay_last_error(void);
int relay_cuda_available(void);

relay_model* relay_model_load(const char* dir);
void relay_model_free(relay_model* m);
int relay_model_config(const relay_model* m, relay_config* out);
/* head_dim / 2 inverse rotary frequencies, Llama 3 scaling applied: what the backends use. */
int relay_model_rope_inv_freq(const relay_model* m, float* out);
/* A float32 weight of the model, writable in place: "embed", "final_norm", "lm_head"
 * (absent when tied), and "layers.<i>.<name>" for attn_norm, wqkv, bqkv (Qwen2 only),
 * wo, mlp_norm, w_gate_up, w_down. NULL if there is no such tensor. */
float* relay_model_tensor(relay_model* m, const char* name, int64_t* numel);

/* backend: "cpu" or "cuda". The engine reports log-probabilities with every token. */
relay_engine* relay_engine_new(relay_model* m, const char* backend, int device, int num_blocks, int block_size,
                               int max_batch_tokens, int max_seqs, int prefix_caching);
void relay_engine_free(relay_engine* e);
int relay_engine_add(relay_engine* e, uint64_t id, const int* prompt, int n, const relay_sampling* sp);
/* A request that already produced `generated` (an unfinished rollout from an earlier
 * step): its prompt and those tokens are recomputed, then it continues with token
 * index ng. Sampling depends on (seed, index), so with the same weights it continues
 * exactly as it would have. */
int relay_engine_add_resume(relay_engine* e, uint64_t id, const int* prompt, int n, const relay_sampling* sp,
                            const int* generated, int ng);
/* One forward pass. Writes up to cap events (cap >= max_seqs is always enough) and
 * returns how many, or -1. */
int relay_engine_step(relay_engine* e, relay_event* out, int cap);
int relay_engine_has_work(relay_engine* e);
int relay_engine_cancel(relay_engine* e, uint64_t id);
void relay_engine_cancel_all(relay_engine* e);
/* Makes the engine use the model's current weights: rebuilds the backend (the CUDA
 * backend uploads them again) and drops the prefix cache. Only between rollouts: fails
 * while requests are in flight. */
int relay_engine_reload_weights(relay_engine* e);
/* The fast weight sync: overwrites one weight of the engine's backend from numel
 * float32 values in relay's layout. On the CUDA backend `src` may be a device pointer
 * on any GPU (a trainer's parameter: converted to fp16 on the device, no host round
 * trip, no rebuild) or host memory; the model's host copy is then NOT updated, and
 * relay_engine_reload_weights() is refused from then on, since it would bring the old
 * weights back. On the CPU backend `src` is host memory, copied into the model's
 * weights, which that backend reads in place. Only between rollouts. Call
 * relay_engine_finish_update() after the last tensor. */
int relay_engine_update_tensor(relay_engine* e, const char* name, const float* src, int64_t numel);
/* Ends a weight update: drops the prefix cache, whose KV entries used the old weights. */
int relay_engine_finish_update(relay_engine* e);
int relay_engine_stats(relay_engine* e, relay_stats* out);

#ifdef __cplusplus
}
#endif

#endif
