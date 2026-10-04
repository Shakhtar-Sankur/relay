# M1 walkthrough: CUDA kernels

Files: `engine/src/cuda_kernels.cuh` (the kernels), `engine/src/cuda_backend.cu` (which
token goes to which kernel, int8 weights), `tests/test_attention.cpp` (M1 against M0 on the
same GPU), `tools/relay_attention_bench.cpp` (timings), `scripts/colab_m1.sh` (both on a T4).
Raw output: [`results/t4/m1-2026-10-04.txt`](../results/t4/m1-2026-10-04.txt).

The promise: the same logits as M0's attention kernel up to fp16 rounding, for every batch
the engine can build (prefill chunks at any offset, decode tokens, both in one batch, any
block table), and faster where attention costs something.

## 1. What M0 did and why it was slow

M0's kernel ran one block of 128 threads per (query token, head). Each block computed the
token's full score row into shared memory, took a softmax, then summed the V rows. Two
problems:

- **Prefill re-reads everything.** A 2,048-token prompt with 32 heads is 65,536 blocks, each
  reading its whole prefix of K and V from global memory, in scalar loads, on CUDA cores.
  Work is quadratic and nothing is shared between neighbouring queries.
- **Decode does not spread.** One sequence's token is `heads` blocks. With grouped-query
  attention (TinyLlama: 32 query heads over 4 KV heads) each K/V row is read by 8 blocks.
  And the score row in shared memory caps the context (`max_context` floats per block).

*Q: Why not just call a library (FlashAttention, FlashInfer)?* The cache is relay's own
paged layout, `[block][kv_head][slot][D]`, and the point of the milestone is to show the
kernels. The layout is close to vLLM's, so the same ideas apply directly.

## 2. Routing tokens to kernels

`CudaBackend::forward` already flattens a batch into tokens with (position, chunk) metadata.
M1 adds one pass over the chunks: a chunk of one token is a decode token; a longer chunk is
cut into tiles of up to 64 consecutive rows. The tile list `(first token, rows)` goes to the
GPU next to the tokens. Both kernels then run for the layer, writing disjoint rows of the
same output, so a batch that mixes decode and prefill (continuous batching with chunked
prefill) needs nothing special.

The KV cache is written before attention by the existing `qkv_post_kernel` (bias, RoPE,
cache write), so attention reads only the cache, never the fresh K/V in `qkv`: a chunk
that starts at position 1,500 sees its earlier 1,500 keys and its own exactly the same way.

## 3. Prefill: flash attention on tensor cores

One block per (tile, query head), four warps, 16 query rows per warp.

1. The tile's 64 Q rows are loaded into shared memory as fp16.
2. The block walks keys 32 at a time, up to its last row's position. Each step gathers the
   32 K and V rows from their cache blocks (16-byte loads, through the block table).
3. Each warp computes S = Q Kᵀ for its 16 rows with WMMA 16x16x16 (fp16 in, fp32 out).
   K is stored row-major `[key][d]`, which is exactly a column-major Kᵀ, so no transpose.
4. Online softmax: two lanes per row, 16 keys each. Causal masking and padding rows set
   scores to -inf. Running max m and sum l per row; P = exp(S - m_new) is written as fp16;
   alpha = exp(m_old - m_new).
5. O = alpha·O + P V, again with WMMA; O lives in shared memory in fp32.
6. At the end, out = O / l.

Shared memory at D = 64: 45.8 KB per block, independent of the context length.

*Q: Why is the prefill speedup larger for the longer prompt (8.7x at 2,048, 4.0x at 512)?*
Attention is quadratic and the GEMMs are linear. At 2,048 tokens M0 spent most of the
3.7 s in attention; M1 moves those FLOPs to tensor cores and reads each K/V tile once per 64
queries instead of once per query.

*Q: Why keep O in shared memory and not in the WMMA accumulator?* The rescale by alpha is
per row, and a WMMA accumulator's element-to-row mapping is unspecified. Storing O lets a
plain loop apply alpha by row; it costs shared-memory traffic, which is the next thing to
remove (raw `mma.sync` with a known fragment layout, as in FlashAttention-2).

## 4. Decode: flash-decoding over the paged cache

One block per (decode token, KV head, 512-key slice of the context), 128 threads.

- **All G query heads of a KV head together.** Q for those G heads goes to shared memory;
  each thread scores one key against all G heads, so each K row is read once, not G times.
  G is a template parameter (1..8), so the per-head arrays stay in registers.
- **Running softmax per head** across 128-key steps, with block-wide max/sum reductions.
  The step's V rows are staged in shared memory, and each thread accumulates a few
  (head, dim) outputs.
- **Split the context.** Each slice writes a partial (m, l, O). A second kernel merges them:
  O = Σ e^(m_p − M) O_p / Σ e^(m_p − M) l_p. A long context becomes many blocks, so even a
  batch of one fills the GPU.

*Q: Why does decode gain little at batch 1 and short context (1.1x for TinyLlama)?* One
token's step reads all 2.2 GB of weights; attention over 512 keys is a small part of it. At
32 sequences of 2,048 tokens attention dominates, and the gain is 5.4x.

*Q: Which bug did the compiler catch?* `alpha[g]` was first read with `g` computed at run
time (from the thread index). A register array indexed at run time is spilled to local
memory: ptxas reported a stack frame. The per-step alpha moved to shared memory.

## 5. Weight-only int8

`RELAY_CUDA_INT8=1` quantizes every linear layer at load: one scale per output row,
scale = max|w| / 127, rounded to nearest. Embeddings, the output layer and norms stay as
they are, so TinyLlama's weights go from 2.2 GB to 1.2 GB.

- **Up to 8 tokens:** `gemv_int8_kernel`. A block computes 8 output rows; for each 512-wide
  slice of K it stages the tokens' activations in shared memory once, then each lane loads 16
  int8 weights of its row (one 16-byte load) and multiplies them with every token.
- **More tokens:** dequantize the matrix to fp16 into a scratch buffer and use cuBLAS.

Measured on the T4: one TinyLlama sequence decodes in 7.2 ms instead of 11.3 ms (1.55x), the
weights take 44% less memory, and logits stay within 0.6-2.8% (relative) of fp16. From 8
sequences on, int8 is slower than fp16: cuBLAS on tensor cores is efficient there, and the
dequantize path reads the int8 weights and writes and reads them again in fp16.

*Q: How would you make int8 win at batch 8-32?* Do not dequantize the whole matrix: fuse the
dequantize into the GEMM's tile loads (int8 to shared memory, convert to fp16 while
building the fragments), so global memory sees only int8. In the GEMV, convert each lane's 16
weights to float once instead of once per token, and give each block more rows so the
staged activations are reused more.

## 6. Tests

- **M1 against M0, same GPU, same inputs** (`test_attention`): two CUDA backends, one forced
  to M0's kernel, run the same schedule: several prompts (1 to 1,300 tokens) in chunks,
  several per forward pass, then decode steps where every other step also carries a fresh
  prompt. The first backend's greedy tokens feed both. Covers chunk offsets, tiles crossing
  chunks, contexts past one 512-key slice, mixed batches, D = 16 and 64, G = 1, 2, 3 and 8.
  Result: worst relative logit difference 2.0e-4 to 8.9e-4 over 6 models, argmax identical
  in every forward pass (94/94 and 43/43).
- **The schedule itself** runs on the CPU backend against itself and must agree exactly.
  The first T4 run failed with "block table too short": the harness re-sent a sequence's
  first generated token as prompt. A harness that is only ever run on the GPU can only be
  debugged on the GPU; running it on the CPU made the bug reproducible locally.
- **Every existing CUDA test** (Hugging Face references, batching, KV transfer) passes with
  the M1 kernels: largest relative error against transformers 3.5e-4 to 1.1e-3, greedy
  continuations identical.
- **int8 against fp16**: printed, with a loose bound (0.25) as a sanity check; the numbers are
  the measurement.

## 7. Results (Tesla T4, median of 5)

| | TinyLlama-1.1B: M0 → M1 | SmolLM2-135M: M0 → M1 |
|---|---|---|
| Prefill, 2,048 tokens | 3,734 → 431 ms (8.7x) | 1,549 → 149 ms (10.4x) |
| Prefill, 512 tokens | 258 → 64 ms (4.0x) | 104 → 20.5 ms (5.1x) |
| Decode, 32 seqs at 2,048 | 176.8 → 33.0 ms/step (5.4x) | 66.6 → 14.1 ms/step (4.7x) |
| Decode, 8 seqs at 2,048 | 48.2 → 16.1 ms/step (3.0x) | 23.1 → 5.7 ms/step (4.1x) |
| Decode, 1 seq at 512 | 12.5 → 11.1 ms/step (1.1x) | 5.3 → 3.5 ms/step (1.5x) |

End to end (`relay-generate`, TinyLlama, 10-token prompt, 128 new tokens per request):
1,217 → 1,785 tok/s for 32 requests (1.47x), 535 → 628 tok/s for 8, unchanged at 85-93 tok/s
for one (weight-bound); with int8, 132 tok/s for one.

These are relay against its own M0, on one GPU. Comparisons with vLLM and serving on several
GPUs are M6.
