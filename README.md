# relay

A disaggregated LLM inference cluster: prompt processing (prefill) and token generation
(decode) run on different workers, the KV cache streams between them layer by layer, and a
server-side Swift control plane routes and schedules requests. The engine is C++20 and
CUDA, written from scratch.

```
               ┌──────────────── Swift control plane (Linux) ────────────────┐
  clients ───▶ │ OpenAI-compatible API · cache-aware router · P/D scheduler   │
  (HTTP/SSE)   │ admission control · health/failover                          │
               └───────────┬──────────────────────────────┬───────────────────┘
              ┌────────────▼─────────┐  KV cache   ┌──────▼──────────────┐
              │ Prefill workers      │ ══════════▶ │ Decode workers      │
              │ C++20/CUDA engine    │ layer by    │ paged KV cache,     │
              │                      │ layer       │ continuous batching │
              └──────────────────────┘             └─────────────────────┘
```

Why and how: [`docs/design.md`](docs/design.md). Code walkthrough with interview
questions: [`docs/m0-walkthrough.md`](docs/m0-walkthrough.md) (engine),
[`docs/m2-walkthrough.md`](docs/m2-walkthrough.md) (KV transfer).

## Status

| | Milestone | State |
|---|---|---|
| M0 | Engine core: Llama-architecture models (Llama 2/3.x, TinyLlama, SmolLM2, Qwen2), float32 CPU reference backend, CUDA backend (fp16, cuBLAS), paged KV cache, continuous batching with chunked prefill and preemption | done |
| M1 | CUDA kernels: paged decode attention, flash-style prefill, fused RMSNorm/RoPE, int8 weights | |
| M2 | KV transfer engine: layer-by-layer streaming during the forward pass, zero-copy TCP and shared-memory transports, recompute fallback under memory pressure | done |
| M3 | Swift control plane: Swift↔C++ interop, gRPC, OpenAI API, tokenizer | |
| M4 | Disaggregated scheduling: prefix-cache-aware routing, prefill/decode pools, admission | |
| M5 | Fault tolerance and deterministic cluster simulation | |
| M6 | Benchmarks on 2-4 GPUs against colocated serving and vLLM | |

## Correctness so far

Logits against Hugging Face transformers on the same weights and tokens, every prompt
position and every generated step. CPU backend: float32. CUDA backend: fp16 weights and
KV cache, fp32 accumulation, on a Tesla T4 (raw output:
[`results/t4/m0-2026-10-03.txt`](results/t4/m0-2026-10-03.txt)).

| Model | CPU: largest relative error | CUDA (T4): largest relative error | Greedy continuation |
|---|---|---|---|
| 4 tiny random fixtures (MHA; GQA + tied bf16; Llama 3 RoPE scaling; Qwen2 biases) | 4e-7 to 7e-7 | 3e-4 to 1.1e-3 | identical on both |
| SmolLM2-135M | 2.4e-6 | 1.2e-3 | identical (12 tokens) on both |
| Qwen2.5-0.5B | 2.7e-6 | not run | identical (12 tokens) on CPU |
| TinyLlama-1.1B | 1.6e-6 | 8.7e-4 | identical (12 tokens) on both |

M0 generation speed on the T4, TinyLlama-1.1B, 128 new tokens per request: 92 tok/s for
one request, 535 tok/s for 8, 1,217 tok/s for 32 (continuous batching; cuBLAS GEMMs and a
simple attention kernel that M1 replaces). These are starting points, not comparisons.

The batching tests pass on both backends; bit for bit on the CPU backend: a request gives the same logits run alone or in a
batch of eight, with its prompt in 7-token chunks, after being preempted and recomputed,
with any block size, and when its KV blocks are copied to a second backend that continues
decoding (disaggregation in miniature).

## The KV transfer engine (M2)

A request prefilled on one backend and decoded on another, with its KV cache streamed
between them over TCP or shared memory, produces exactly the tokens of one engine doing
everything, and on the CPU backend exactly the same logits: with chunked prefill, with
the decode side out of memory (it falls back to recomputing the prompt), for sampled and
greedy requests. A sender that dies mid-transfer gives every reserved block back; a sender
running a different model is refused. The threaded code runs clean under ThreadSanitizer.

Measured on the 4-vCPU development container
([`results/cpu/m2-transfer-2026-10-03.txt`](results/cpu/m2-transfer-2026-10-03.txt)):

| | |
|---|---|
| Raw throughput between two processes | shared memory 6-8 GB/s; TCP over loopback 1.7-3.6 GB/s |
| SmolLM2-135M, 512-token prompt (23.6 MB of KV cache), transfer time left after the prefill forward pass | sent after the pass: 6.5 ms (TCP), 16.9 ms (shm); streamed per layer: 0.05 ms and 0.37 ms, 98-99% hidden |

On a Tesla T4 with the CUDA backend ([`results/t4/m2-2026-10-03.txt`](results/t4/m2-2026-10-03.txt)),
TinyLlama-1.1B, prefill on one backend and decode on another in one process:

| Prompt | KV cache | Transfer left after prefill: sent afterwards | streamed per layer |
|---|---|---|---|
| 512 tokens | 11.5 MB | 6.4 ms (TCP), 15.0 ms (shm) | 0.08 ms, 0.06 ms |
| 2,048 tokens | 46.1 MB | 21.6 ms (TCP), 56.5 ms (shm) | 0.08 ms, 0.04 ms |

Streaming hides 99-100% of the transfer. The first T4 run hid 1-14% at 0.5 GB/s
([`results/t4/m2-2026-10-03-first-run.txt`](results/t4/m2-2026-10-03-first-run.txt)): each
layer was 64 small copies into pageable memory, which cannot overlap kernels. A gather
kernel and one copy per layer into page-locked memory fixed it. The forward pass times are
M0's attention kernel, which M1 replaces; a faster prefill leaves less time to hide the
transfer in, so these numbers will be re-measured after M1.

## Build and test

```bash
cmake -S . -B build -G Ninja && cmake --build build
ctest --test-dir build --output-on-failure

# with the CUDA backend
cmake -S . -B build -G Ninja -DRELAY_CUDA=ON && cmake --build build
RELAY_TEST_BACKEND=cuda ctest --test-dir build --output-on-failure

# against a real model: write its Hugging Face reference, then test
python scripts/make_fixtures.py --model path/to/SmolLM2-135M
RELAY_REFERENCE_MODELS=path/to/SmolLM2-135M ./build/test_reference

# generate (token ids in and out until the tokenizer lands in M3)
./build/relay-generate --model path/to/model --ids "504 1296 768" --max-new 32 [--backend cuda]
```

On a GPU in Colab, `scripts/colab_m0.sh` does all of the above.

## Layout

```
engine/include/relay/   public headers: config, safetensors, weights, kv_cache, backend, engine, sampler
engine/src/             CPU backend, CUDA backend (cuda_backend.cu), engine, loaders
tools/relay_generate.cpp
tests/                  C++ tests; tests/fixtures holds tiny models and their transformers outputs
scripts/                fixture generation, Colab GPU script
docs/                   design and walkthroughs
```
