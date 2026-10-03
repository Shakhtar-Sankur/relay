# M0 walkthrough: the engine core

Read this with the code open. Each section says which file to look at, what it does,
and the questions an interviewer is likely to ask about it.

## 1. Loading a model (`safetensors.cpp`, `config.cpp`, `weights.cpp`)

A Hugging Face model directory has `config.json` (the hyperparameters) and
`model.safetensors` (the weights). A safetensors file is 8 bytes giving the header length,
a JSON header mapping each tensor name to its dtype, shape and byte range, then the raw
bytes. relay memory-maps the file (`mmap`), so nothing is read until it is used, and each
tensor is a pointer into the mapping (`TensorView`).

`HostWeights::load` converts every tensor to float32 and fuses projections that always see
the same input: q, k and v into one `wqkv` matrix, gate and up into `w_gate_up`. One
multiply instead of three (or two) means fewer, larger GEMMs.

`ModelConfig` reads both config formats: transformers 4 writes `rope_theta` and
`rope_scaling` at the top level; transformers 5 puts them in `rope_parameters`.
Qwen2 is the Llama architecture plus biases on q/k/v, so it is a flag, not a new model.

*Q: Why mmap instead of reading the file?* The OS pages in only what is touched, the
load is lazy and needs no extra copy, and the page cache is shared between processes
loading the same model (several workers on one machine).

*Q: Why does `rope_inv_freq` have special code for Llama 3?* Llama 3.1/3.2 extend the
context by scaling the rotary frequencies: low frequencies (long wavelengths) are divided
by `factor`, high ones are kept, and the band between is interpolated. Get it slightly
wrong and the model still produces fluent text but different logits: the fixture
`llama3-rope` catches exactly that (a 7.7e-3 error without scaling against a 1e-4
tolerance).

## 2. The forward pass (`cpu_backend.cpp`)

Input: a `ForwardBatch`, a list of `SeqChunk`s. Each chunk is some tokens of one sequence
starting at `start_pos`, plus that sequence's block table. The backend concatenates all
tokens (T of them) and runs, for each layer:

1. RMSNorm each token's hidden state.
2. `qkv = xn @ wqkv^T` for all T tokens at once (one GEMM).
3. Add biases (Qwen2), apply RoPE to q and k at each token's own position, write k and v
   into the cache at (block_table[pos / B], pos % B).
4. Attention: token at position p of sequence s attends to positions 0..p of s, reading
   K and V from the cache through s's block table. Grouped-query attention: query head h
   uses KV head h / (heads / kv_heads).
5. `x += attn @ wo^T`, RMSNorm, `gu = xn @ w_gate_up^T`, `act = silu(gate) * up`,
   `x += act @ w_down^T`.

Finally RMSNorm and the LM head, only on the rows that need logits (each chunk's last
token), because the LM head is the largest single multiply (vocab x hidden).

*Q: Why write K/V into the cache before attention, for the current chunk too?* Then
attention has one code path: everything it reads, including the chunk's own earlier
tokens, comes from the cache. Causality holds because token p reads only positions <= p.

*Q: How can the CPU backend be bit-exact between batched and unbatched runs?* Every output
element is a dot product computed with 8 partial sums added in a fixed order, and
attention is computed per token. Nothing depends on how many tokens are in the batch, so
the same inputs give the same bits. That turns "batching does not change results" into
an exact test.

*Q: What does the CUDA backend do differently?* Weights and KV cache in fp16, residual
stream in fp32. GEMMs go to cuBLAS (fp16 inputs, fp32 accumulate); the residual add is
folded into the GEMM with beta = 1. Small kernels do the rest. Its attention kernel puts
all scores of one (token, head) in shared memory; M1 replaces it with kernels that scale.

## 3. The paged KV cache (`kv_cache.h`)

Memory is a pool of blocks of B tokens. Layout per layer: K[block][kv_head][slot][dim].
A sequence owns a list of blocks (its block table); blocks need not be contiguous.

*Q: Why paging?* Without it each request reserves memory for its maximum length up
front, and fragmentation wastes most of the cache. With blocks, a request holds only
what it uses (at most B - 1 slots wasted), and blocks can be shared between requests with
a common prefix (reference counts, M4).

*Q: Why this layout?* One layer's slice of one block is contiguous. The prefill worker
can send layer l's blocks as soon as layer l is done, while it computes layer l + 1.
That is M2's transfer engine.

*Q: What decides the block size?* Smaller blocks waste less memory and allow finer prefix
sharing; larger blocks mean shorter block tables and more contiguous reads in attention.
16 is the common default.

## 4. Continuous batching (`engine.cpp`)

`Engine::step()` builds one batch and runs one forward pass:

1. Every request that is generating contributes its newest token (oldest request first).
   If its next token needs a new block and none is free, the youngest running request is
   preempted: blocks freed, back to the front of the queue, recomputed later.
2. Remaining token budget goes to prompts: first requests part-way through their prompt,
   then new requests from the queue. A long prompt is split across steps (chunked prefill),
   so it never stalls the decoding requests for a whole step.
3. After the forward pass, each request whose chunk reached its end samples a token;
   finished requests free their blocks immediately.

*Q: Why decode first?* Decoding requests have users waiting on every token; prefill can
be chunked to fit around them. That is the policy that keeps inter-token latency steady.

*Q: Why preempt the youngest?* It has the least work invested, so recomputing it costs
least, and the oldest requests (closest to finishing) keep making progress.

*Q: After preemption, how do you know the recomputed request produces the same tokens?*
Sampling is a pure function of (logits, seed, token index). Recompute gives the same logits
(bit-exact on CPU), so the same tokens. The test `preemption_and_recompute_change_nothing`
forces preemption with a tiny cache and compares every logit.

## 5. Tests (`tests/`)

- `test_reference`: logits against Hugging Face transformers on four tiny random models
  (MHA, GQA with tied bf16 weights, Llama 3 RoPE scaling, Qwen2 biases) and, given
  `RELAY_REFERENCE_MODELS`, real ones. Every prompt position and every generated step.
- `test_batching`: batched == alone, chunked == whole, tight cache with preemption ==
  roomy cache, block size 1 == block size 32: tokens equal and (CPU) every logit equal.
- `test_kv_cache`: the allocator, the layout, and disaggregation in miniature: compute a
  prompt on one backend, copy its KV blocks layer by layer into another backend under
  different block ids, keep decoding there, and get identical logits.
- `test_safetensors`, `test_sampler`: the file format, fp16 conversion (all 65536 values),
  JSON, both config formats, sampling determinism and distribution.

*Q: How do you know your tests can fail?* Remove Llama 3's RoPE scaling from a copy of the
fixture's config and the reference test fails (7.7e-3 relative error). The tests are
checked to catch the mistakes they are meant to catch.

## Numbers (CPU, this container)

| Model | Largest relative error vs transformers (float32) | Greedy tokens |
|---|---|---|
| 4 tiny fixtures | 4e-7 to 7e-7 | identical |
| SmolLM2-135M | 2.0e-6 | 12/12 identical |
| Qwen2.5-0.5B | 2.7e-6 | 12/12 identical |
| TinyLlama-1.1B | 1.6e-6 | 12/12 identical |

GPU results come from `scripts/colab_m0.sh` on a T4.
