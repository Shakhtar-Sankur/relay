# M4 walkthrough: the cluster

Files: `engine/src/kv_cache.cpp` (BlockManager: prefix caching), `engine/src/worker.cpp` and
`tools/relay_worker.cpp` (the worker process), `control/Sources/RelayControl/Cluster.swift`
and `Router.swift` (the Swift side), `scripts/cluster.sh`, `scripts/loadgen.py`.

## 1. Prefix caching

Many requests start the same way: a system prompt, few-shot examples, earlier turns of a
conversation. Their KV cache for that prefix is identical, so computing it again is waste.

Each full block gets a chain hash: `h_i = hash(h_{i-1}, tokens of block i)`. Because it
chains, a hash names the whole prefix up to that block, not just the block's own tokens.
Full blocks are indexed by hash. On admission, the engine walks the prompt block by block and
reuses every block found (reference count + 1), then computes only the rest. At least one
prompt token is always computed, because its logits are what the first new token is
sampled from.

Freed blocks that are indexed go to an LRU list instead of the free list: still reusable,
evicted only when nothing free is left. A preempted request's blocks stay cached, so its
recompute is mostly lookups.

*Q: How can two requests share a block safely?* Only full blocks are shared, and a request
writes only positions past its matched prefix, which are in blocks it owns.

*Q: What does it buy?* With a 37-token shared prefix and 6 requests, 160 of 300 prompt tokens
came from the cache and the forward pass computed 179 tokens instead of 339, with logits
bit-identical to running without the cache.

## 2. The worker process

`relay-worker --role both|prefill|decode` serves one control connection (the control plane)
with a small framed protocol: Submit, Resume, Cancel, Peer in; Hello, Token, Error, Load out.

- **both**: an Engine; the whole request runs here (colocated serving).
- **prefill**: a PrefillWorker; for each request it samples the first token, streams the KV
  cache (M2) to the decode worker named in the Submit, and reports the first token.
- **decode**: an Engine plus a KV port; each prefill worker connects once, and a KVReceiver
  per connection hands finished transfers to the engine, which reports tokens 2..n.

Threads: the control reader turns frames into commands; one step thread applies commands and
runs the engine; KV receivers run on their own threads. Load reports (free blocks,
prefix-cache hits, tokens computed) go out every 20 ms.

*Q: Why does the first token come from the prefill worker?* It is sampled from the last
prompt position's logits, which only the prefill worker computes. Sending it immediately is
what keeps time-to-first-token low; the decode worker starts at index 1 with the same seed,
so the stream is identical to one engine's.

## 3. The Swift cluster (`Cluster.swift`)

`Cluster` implements the same `GenerationBackend` as the in-process engine, so the OpenAI API
runs on it unchanged. It holds one connection per worker. At start it tells every prefill
worker to connect to every decode worker's KV port. For each request it picks workers,
records the request, sends Submit, and merges tokens from the two workers into one stream in
index order. Cancellation (a client leaving, a stop string) sends Cancel to both.

## 4. Routing and admission (`Router.swift`)

- **Prefill, prefix-aware**: key = hash of the prompt's first block. Each candidate gets the
  score `mix(key, worker)` and the highest wins (rendezvous hashing): the same prefix always
  goes to the same worker, whose cache then has it, and adding a worker only moves the keys
  it wins. Only workers within `slack` queued prompt tokens of the least loaded are
  candidates, so affinity never piles work onto a busy worker.
- **Decode**: most free KV blocks after the blocks already promised to requests routed there.
- **Admission**: at most `maxInFlight` requests; beyond that the API answers 503 immediately
  rather than letting queues and latency grow without bound.

*Q: Why rendezvous hashing rather than "hash mod N"?* With mod N, adding a worker remaps
almost every key and every cache goes cold at once.

## 5. Bugs found on the way

- ThreadSanitizer: `KVReceiver` started its thread in the constructor, and the worker set its
  callback afterwards. Callbacks are now constructor arguments.
- Worker processes launched by the tests ignored SIGTERM: Foundation's `Process` starts them
  from a thread that blocks signals, and the mask is inherited. The worker unblocks signals at
  startup and asks the kernel to stop it if its parent dies (`PR_SET_PDEATHSIG`).
- `Process.waitUntilExit()` spins at 100% CPU off the main thread on Linux; the tests poll.

## 6. Tests

C++ (`test_worker`): workers in-process, driven over TCP like the control plane: colocated,
1 prefill to 1 decode, 1 prefill to 2 decodes, and Resume all equal one engine; cancel frees
memory; errors are reported. Swift (`ClusterTests`): router unit tests; then real
`relay-worker` processes behind the API: disaggregated and colocated clusters match
transformers' output, 16 concurrent requests, shared system prompts land on one prefill
worker and hit its cache, a client that leaves frees memory on every worker, admission
answers 503 when full.
