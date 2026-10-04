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
[`docs/m1-walkthrough.md`](docs/m1-walkthrough.md) (CUDA kernels),
[`docs/m2-walkthrough.md`](docs/m2-walkthrough.md) (KV transfer),
[`docs/m3-walkthrough.md`](docs/m3-walkthrough.md) (Swift control plane),
[`docs/m4-walkthrough.md`](docs/m4-walkthrough.md) (cluster),
[`docs/m5-walkthrough.md`](docs/m5-walkthrough.md) (fault tolerance).

## Status

| | Milestone | State |
|---|---|---|
| M0 | Engine core: Llama-architecture models (Llama 2/3.x, TinyLlama, SmolLM2, Qwen2), float32 CPU reference backend, CUDA backend (fp16, cuBLAS), paged KV cache, continuous batching with chunked prefill and preemption | done |
| M1 | CUDA kernels: flash-style prefill attention on tensor cores, split-context (flash-decoding) attention over the paged cache, weight-only int8 | done |
| M2 | KV transfer engine: layer-by-layer streaming during the forward pass, zero-copy TCP and shared-memory transports, recompute fallback under memory pressure | done |
| M3 | Swift control plane: Swift↔C++ interop, a tokenizer matching Hugging Face, chat templates, an OpenAI-compatible HTTP API with streaming | done (in-process engine; remote workers in M4) |
| M4 | The cluster: `relay-worker` processes (prefill, decode, colocated) behind the Swift control plane; prefix caching; prefix-aware routing (rendezvous hashing with a load guard), memory-aware decode placement, admission control | done |
| M5 | Fault tolerance: lost workers (killed, frozen, stalled) detected by connection, heartbeat and progress watchdogs; their requests resumed elsewhere with identical output; workers rejoin; seeded fault injection of the KV transfer | done |
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
simple attention kernel that M1 replaces). M1's numbers are below.

The batching tests pass on both backends; bit for bit on the CPU backend: a request gives the same logits run alone or in a
batch of eight, with its prompt in 7-token chunks, after being preempted and recomputed,
with any block size, and when its KV blocks are copied to a second backend that continues
decoding (disaggregation in miniature).

## CUDA kernels (M1)

M0's attention kernel ran one block per query token and head, with the whole score row in
shared memory. M1 replaces it with two kernels that read the paged KV cache directly
(`engine/src/cuda_kernels.cuh`):

- **Prefill**: flash attention on tensor cores. A block takes 64 query rows of one head,
  walks the keys 32 at a time (WMMA 16x16x16, fp16 in, fp32 out), and keeps a running
  softmax, so no context-by-context matrix exists and the context length no longer depends
  on shared memory.
- **Decode**: flash-decoding. A block takes one sequence, one KV head and a 512-key slice of
  the context; all query heads sharing that KV head (grouped-query attention) are scored
  together, so each K/V row is read once; a second kernel merges the slices (log-sum-exp).
- **Weight-only int8** (optional, `RELAY_CUDA_INT8=1`): one scale per output row; decode
  batches of up to 8 tokens use a matrix-vector kernel that streams the int8 weights,
  larger batches dequantize and use cuBLAS.

Checked on a T4 against M0's kernel on the same inputs (prompts split into chunks, contexts
past one slice, decode and prefill in one batch, multi-head and grouped-query models): worst
relative logit difference 2e-4 to 9e-4, every argmax the same. Every Hugging Face reference
test passes with the M1 kernels, generated tokens identical. Raw output:
[`results/t4/m1-2026-10-04.txt`](results/t4/m1-2026-10-04.txt).

Forward-pass time on the T4, M0 kernel vs M1 kernels (everything else unchanged):

| | TinyLlama-1.1B: M0 → M1 | SmolLM2-135M: M0 → M1 |
|---|---|---|
| Prefill, 2,048-token prompt | 3,734 → 431 ms (**8.7x**) | 1,549 → 149 ms (**10.4x**) |
| Prefill, 512-token prompt | 258 → 64 ms (4.0x) | 104 → 20.5 ms (5.1x) |
| Decode step, 32 sequences at 2,048 tokens | 176.8 → 33.0 ms (**5.4x**) | 66.6 → 14.1 ms (4.7x) |
| Decode step, 8 sequences at 2,048 tokens | 48.2 → 16.1 ms (3.0x) | 23.1 → 5.7 ms (4.1x) |
| Decode step, 1 sequence at 512 tokens | 12.5 → 11.1 ms (1.1x) | 5.3 → 3.5 ms (1.5x) |

The gain grows with context and batch, where attention is the cost. One short sequence is
bound by reading the weights, so attention changes little there: end-to-end TinyLlama
generation (10-token prompt, 128 new tokens) goes from 535 to 628 tok/s for 8 requests and
from 1,217 to 1,785 tok/s for 32 (1.47x), and stays at 85-93 tok/s for one.

int8 (the transformer layers; embeddings and the output layer stay fp16) cuts TinyLlama's
weights from 2.2 to 1.2 GB and helps exactly where reading them dominates: one TinyLlama
sequence decodes in 7.2 ms instead of 11.3 (1.55x; 132 vs 85 tok/s end to end). From 8
sequences on, fp16 on tensor cores wins (dequantizing for cuBLAS costs more than it saves),
so int8 is a batch-1 latency and memory option, not a throughput one. Its logits stay within
0.6-2.8% (relative) of fp16, with occasional argmax flips at near-ties.

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
kernel and one copy per layer into page-locked memory fixed it.

Re-measured with M1's kernels, whose prefill is about 9x shorter, so there is far less compute
to hide the transfer behind: streaming still hides 99-100%. For 2,048 tokens the forward pass
is now 384-386 ms, and the 46 MB that would be exposed for 29.3 ms (TCP) or 54.3 ms (shm) if
sent afterwards costs under 0.01 ms and 0.04 ms streamed; for 512 tokens, 7.2 ms and 13.7 ms
become 0.05 ms and 0.13 ms
([`results/t4/m2-2026-10-04-m1-kernels.txt`](results/t4/m2-2026-10-04-m1-kernels.txt)).
Each layer's prefill still takes longer than sending its KV cache, so the transfer fits.

## The Swift control plane (M3)

```bash
swift build -c release
.build/release/relay-server --model path/to/SmolLM2-135M --port 8000
curl localhost:8000/v1/chat/completions -d '{"messages":[{"role":"user","content":"Hi"}],"stream":true}'
```

Swift calls the C++ engine directly (Swift 6 C++ interoperability; the engine is a
reference-counted class to Swift). The tokenizer, written in Swift, gives the same ids as
Hugging Face `tokenizers` on all 25 test strings for SmolLM2, Qwen2.5 and TinyLlama, and
chat templates render exactly as `apply_chat_template`. On the real SmolLM2-135M,
`/v1/completions` returns transformers' greedy continuation word for word. Tests cover
streaming, stop strings (never leaked, request cancelled in the engine), concurrent requests
and clients hanging up. Walkthrough: [`docs/m3-walkthrough.md`](docs/m3-walkthrough.md).

## The cluster (M4)

```bash
cmake -S . -B build -G Ninja && cmake --build build      # relay-worker (add -DRELAY_CUDA=ON for GPUs)
swift build -c release                                   # relay-server
scripts/cluster.sh path/to/model disaggregated 1 2       # 1 prefill + 2 decode workers + API on :8000
scripts/cluster.sh path/to/model colocated 2             # 2 workers doing both
python scripts/loadgen.py --url http://127.0.0.1:8000 --requests 64 --concurrency 8
```

Each worker is a process serving the control plane over TCP. A prefill worker samples the
first token and streams the KV cache to the decode worker the router chose; that worker
generates the rest. Requests sharing a prefix go to the same prefill worker (rendezvous
hashing on the first KV block, unless that worker is busier than the others by a margin),
whose prefix cache then serves it; decode goes where KV memory is free; beyond a limit of
requests in flight the API answers 503. Tested end to end with real worker processes: the
disaggregated and colocated clusters reproduce transformers' output, concurrent requests,
cache hits for shared system prompts, cancellation freeing memory on every worker.
Walkthrough: [`docs/m4-walkthrough.md`](docs/m4-walkthrough.md).

## Fault tolerance (M5)

A worker can die, freeze or come back at any moment. The control plane notices (closed
connection, 2 s of silence, or 30 s without progress while it holds requests) and places each
of its requests again from where it stands: no token yet, a fresh prefill; some tokens, a
decode worker resumes it from the prompt and the tokens so far. Sampling depends only on
(seed, token index), so the client's stream carries on with exactly the tokens it would have
had, each once, in order. Workers that come back rejoin; while every worker of a role is
down, requests wait for one rather than fail.

- Seeded fault injection of the KV transfer (300 seeds + a sweep of 99 cut points, the
  connection broken anywhere, mid-frame included): every request is delivered exactly or
  reported for retry, and no KV block leaks on either side.
- Real `relay-worker` processes killed (SIGKILL), frozen (SIGSTOP) and stalled under live
  requests, and a seeded chaos test (24 concurrent requests, 8 kills and restarts): every
  output identical to a fault-free run.
- `scripts/chaos.py` runs a real cluster (worker processes + the Swift API server) serving
  SmolLM2-135M and SIGKILLs workers at random while 64 requests stream through it, then
  restarts them in place. Over 3 seeds and both cluster shapes (2 prefill + 2 decode; 3
  colocated): 384 of 384 outputs identical to a fault-free run through 60 kills, none failed;
  each 64-request run took 58-66 s instead of 46-53 s. Raw output:
  [`results/cpu/m5-chaos-2026-10-03.txt`](results/cpu/m5-chaos-2026-10-03.txt).

The tests found three bugs on the way, two of them already in M4: tokens from two workers
could overtake each other on the way to the client, and a worker restarted from a launcher's
thread was killed when that thread ended (`PR_SET_PDEATHSIG` follows the thread, not the
process). See the walkthrough:
[`docs/m5-walkthrough.md`](docs/m5-walkthrough.md).

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

On a GPU in Colab, `scripts/colab_m0.sh` does all of the above; `scripts/colab_m1.sh` checks
and times the M1 kernels (`tools/relay_attention_bench.cpp`).

## Layout

```
engine/include/relay/   public headers: config, safetensors, weights, kv_cache, backend, engine, sampler
engine/src/             CPU backend, CUDA backend (cuda_backend.cu), engine, loaders
tools/relay_generate.cpp
tests/                  C++ tests; tests/fixtures holds tiny models and their transformers outputs
scripts/                fixture generation, Colab GPU scripts, cluster launcher, load generator, chaos run
docs/                   design and walkthroughs
```
