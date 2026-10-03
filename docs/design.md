# relay: design

relay serves a large language model from a small cluster in which prompt processing
(prefill) and token generation (decode) run on different workers. This document is the
plan and the reasons behind it; each milestone adds a section on what was built.

## Why split prefill and decode

A request's life has two phases with opposite needs:

- **Prefill** runs the whole prompt through the model at once: large matrix multiplies,
  limited by compute. It produces the KV cache (every layer's keys and values for every
  prompt token) and the first generated token.
- **Decode** generates one token per step per request: tiny matrix multiplies, limited by
  how fast the GPU can read the weights and the KV cache from memory.

On one GPU the two compete: a long prompt arriving makes every running request's next
token wait behind it (higher inter-token latency), and batching is a compromise between
both. Splitting them lets each pool be sized, batched and scheduled for its own phase.
The cost is moving the KV cache from the prefill worker to the decode worker, fast enough
that it does not show up in latency. Production stacks (vLLM, SGLang, TensorRT-LLM,
NVIDIA Dynamo) all support this now. Published results also say it rarely pays off for
throughput below about 8 GPUs per model, so relay's benchmarks (M6) report what they
measure, including where colocated serving wins.

## Components

```
               ┌──────────────── Swift control plane (Linux) ────────────────┐
  clients ───▶ │ OpenAI-compatible API · cache-aware router · P/D scheduler   │
  (HTTP/SSE)   │ admission control · health/failover   (gRPC + distributed   │
               │ actors; calls the engine through Swift↔C++ interop)          │
               └───────────┬──────────────────────────────┬───────────────────┘
                           │                              │
              ┌────────────▼─────────┐  KV cache   ┌──────▼──────────────┐
              │ Prefill workers      │ ══════════▶ │ Decode workers      │
              │ C++20/CUDA engine    │ layer-by-   │ paged KV cache,     │
              │ flash-style prefill  │ layer       │ continuous batching │
              └──────────────────────┘ streaming   └─────────────────────┘
```

- **Engine (C++20, CUDA)**, `engine/`: loads a Hugging Face model, holds the paged KV
  cache, runs batched forward passes, samples tokens. Two backends: a float32 CPU
  backend that is the correctness reference, and a CUDA backend.
- **KV transfer engine (C++)**, M2: ships one layer's KV blocks at a time from a
  prefill worker to a decode worker while the prefill worker computes the next layer.
- **Control plane (Swift)**, M3-M5: the API, tokenizer, routing, scheduling, health.

## Decisions in M0

**One flattened batch, per-token metadata.** A batch is a list of sequence chunks of any
length (a whole prompt, a slice of one, or one decode token). Tokens are concatenated;
every matrix multiply runs over all of them; only attention reads which sequence a token
belongs to. This is what lets prefill chunks and decode tokens share a forward pass
(continuous batching with chunked prefill), and it is the same interface a prefill-only or
decode-only worker uses.

**Paged KV cache, laid out per layer.** Cache memory is a pool of fixed-size blocks
(default 16 tokens). A sequence's block table maps position p to block
table[p / block_size], slot p % block_size. The layout is
K[layer][block][kv_head][slot][head_dim]: one layer's slice of one block is one
contiguous run of bytes, which is the unit the transfer engine sends. Blocks are
reference counted for prefix sharing (M4).

**Sampling is a pure function.** The random number for a request's n-th token comes from
hashing (seed, n). A request therefore produces the same tokens alone or batched, after
preemption and recompute, and (later) when its prefill ran on another machine. Tests
depend on it, and so does fault recovery (M5): a decode worker that dies can be replaced
by recomputing, without changing what the client has already received.

**The CPU backend is exact about batching.** Each output element is computed by the same
floating-point operations whatever else is in the batch (fixed-order dot products,
per-token attention). So "batched equals alone" is tested bit for bit, not with a
tolerance, which catches indexing mistakes that a tolerance would hide. The CUDA backend
uses cuBLAS, whose algorithm depends on matrix shape, so it is checked against
Hugging Face with a tolerance instead.

**Preemption by recompute.** When the cache is full, the most recently admitted request
gives back its blocks and is recomputed later. Swapping to host memory is the other
option; recompute is simpler and, with prefill being fast, usually competitive.

## Milestones

| | Milestone | Proof |
|---|---|---|
| M0 | Engine core: Llama-architecture models, CPU reference and CUDA backends, paged KV cache, continuous batching with chunked prefill and preemption | logits match Hugging Face; batching, chunking, preemption and moving KV blocks between backends change nothing |
| M1 | CUDA kernels: paged decode attention, flash-style prefill, fused RMSNorm/RoPE, int8 weights | against the CPU reference; timed on a T4 |
| M2 | KV transfer engine: layer-by-layer streaming; shared memory / CUDA IPC on one machine, zero-copy TCP between machines | GB/s; share of the transfer hidden behind prefill |
| M3 | Swift control plane: Swift↔C++ interop, gRPC, OpenAI-compatible streaming API, tokenizer | end-to-end chat; tokenizer matches Hugging Face |
| M4 | Disaggregated scheduling: prefix-cache-aware routing, prefill/decode pools, admission control | routing decisions tested |
| M5 | Fault tolerance: worker crashes during transfer, recovery; deterministic simulation of the cluster | outputs identical to one machine under faults |
| M6 | Benchmarks on 2-4 GPUs against colocated serving and vLLM | raw data and charts |
