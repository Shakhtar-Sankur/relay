# M5 walkthrough: fault tolerance

Files: `control/Sources/RelayControl/Cluster.swift` (recovery, watchdog, rejoin, waiting),
`engine/src/kv_transfer.cpp` and `engine/src/worker.cpp` (per-request failure on the worker
side), `tests/test_faults.cpp` (seeded fault injection), `tests/test_worker.cpp`,
`control/Tests/RelayControlTests/FaultToleranceTests.swift` (real processes, killed),
`scripts/chaos.py` (a real model, a real cluster, workers killed at random).

The promise: a worker can die, hang or come back at any moment, and every request still
returns exactly the tokens it would have returned with no fault, or fails with a clean error
if no worker of a role it needs comes back in time. Nothing is lost silently, repeated, or
leaked.

## 1. Why recovery can be exact

Sampling is a pure function of (seed, token index, logits), and the CPU backend computes the
same logits for a request alone or batched, chunked, cached, preempted or transferred (M0,
M2). So a request that has produced tokens 0..k-1 on one worker can be resumed on another
from its prompt plus those k tokens, and token k onward comes out the same. Recovery is
`Engine::add_resume` (M4) used in anger.

*Q: What would break exactness on a GPU?* Batching can reorder floating-point sums in cuBLAS,
so a resumed request may diverge where a near-tie in the logits flips. The tests compare
tokens on the CPU backend; on CUDA they check that every request completes with the right
length.

## 2. Detecting a lost worker

Three signals, all in the control plane:

- **The connection closes**: the process died (its kernel closed the socket).
- **Silence** for `heartbeatTimeout` (2 s): the worker sends a Load report every 20 ms from a
  thread of its own, so a long step does not look like silence. A frozen process (SIGSTOP),
  a hung host or a partition does.
- **No progress** for `stallTimeout` (30 s): the worker holds requests, answers heartbeats,
  and its step counter does not move.

*Q: Why a separate heartbeat thread?* On a CPU one prefill step can take seconds. Sending
Load from the step loop would make a busy worker indistinguishable from a dead one.

## 3. Placing a request again

When a worker is lost, every request it held is placed again, from where it stands:

| The request has | New placement |
|---|---|
| no token yet | a prefill worker and a decode worker, as new; with no prefill worker left, a decode worker does the whole request |
| some tokens | a decode (or colocated) worker resumes it: recomputes prompt + tokens so far, continues at the next index |

A lost prefill worker also matters for a request whose first token was already reported but
whose decode worker has not produced one yet: the KV cache may never have arrived. Those are
moved too; once the decode worker has shown it holds the request, they are left alone.

Every placement gets a fresh **wire id**: the id the workers know it by. Tokens that arrive
from an abandoned placement name a wire id that no longer exists and are dropped, and the
old workers get a Cancel for it. The client's stream only ever gets the next index, so it
sees each token exactly once, in order.

When no worker of a needed role is up (all lost at once, or still restarting) the request
waits, for up to `placementTimeout` (30 s), and is placed as soon as one rejoins. A request
gives up after `maxPlacements` (16) placements.

## 4. Losing a KV transfer, not a worker

The KV connection between a prefill and a decode worker can break while both control
connections are fine. The sender then fails as a whole: everything queued is dropped and
every later frame is dropped as it is queued, and each End or Abort learns through its
callback whether it reached the wire. The prefill worker reports each request whose cache did
not arrive as a **retryable** error (even if its first token was already reported), and the
control plane places it again, preferring another decode worker. The prefill worker
reconnects to that decode worker on its own the next time it is asked to send there, at most
once a second.

When the control plane loses a decode worker it tells every prefill worker to forget it, so
transfers stuck behind a frozen peer fail at once instead of holding prefill memory.

## 5. Coming back

Lost workers are reconnected in the background (each attempt on its own thread, so a frozen
worker that accepts but never answers holds up nobody). A worker that answers on the same
address with the same model and role rejoins; prefill workers are pointed at a rejoining
decode worker's new KV port.

A worker drops every request it holds when a new control connection starts: the control plane
has already moved them. It greets the new plane at once and keeps the old requests' tokens to
itself until its step thread has dropped them (it does not wait for that step: on a CPU it
can take seconds, and a plane that times out waiting for the greeting would keep retrying).

## 6. Tests

- **Seeded fault injection** (`test_faults`): a prefill worker streams requests to a decode
  engine over a connection that breaks after a seeded number of bytes, anywhere: between
  frames, mid-header, mid-block. 300 seeds plus a sweep of 99 cut points. Every request is
  either delivered with exactly the reference tokens or reported for retry, never both,
  never neither; both sides get every KV block back. Two deliberately planted bugs (a failure
  not reported; blocks not freed) are caught.
- **Workers over the control protocol** (`test_worker`): a dead decode worker fails only its
  own requests, and they resume exactly elsewhere; forgetting a peer settles every request
  and frees every block; a new control connection drops the old one's requests; a prefill
  worker reconnects to a restarted decode worker; 16 identical requests at a time across 2+2
  workers, 20 rounds.
- **Real processes** (`FaultToleranceTests`): SIGKILL a decode worker mid-stream, a prefill
  worker before the first token, a colocated worker mid-stream; kill the only prefill worker
  (decode does it all) and restart it (it rejoins and is used); SIGSTOP a decode worker (the
  heartbeat catches it in 0.75 s; SIGCONT and it rejoins); a worker that answers heartbeats
  but does not step (the stall watchdog); every worker down (requests wait, and are served
  when one returns, or fail after the timeout). Then chaos: 24 concurrent requests while
  workers are killed and restarted at seeded random moments. Every output equals the
  fault-free run.
- ThreadSanitizer runs the worker, fault-injection and transfer tests in CI.

## 7. Bugs the tests found

- **A watchdog that saw 584 years of silence.** The watchdog read the clock, then a reader
  thread stamped a newer heartbeat; `now - lastHeard` on unsigned integers wrapped around,
  and healthy workers were declared dead. Found by the chaos test (workers lost that nobody
  killed), traced with an `LD_PRELOAD` shim logging every `shutdown()` on a socket with its
  backtrace. Fixed with a difference that cannot wrap; a regression test covers it.
- **Tokens overtaking each other (present since M4).** Tokens from the prefill worker and the
  decode worker arrive on different threads. They were ordered under a lock but yielded to the
  client's stream after it, so one thread could overtake the other: a stream with tokens out
  of order, or ended before an earlier token was yielded. It showed with 16 identical
  concurrent requests, about one run in five; at the token level, not in HTTP. Fixed by
  yielding under the lock.
- **Workers killed by their launcher's thread.** `relay-worker` asked the kernel to stop it
  when its launcher died (`PR_SET_PDEATHSIG`), so workers would not outlive a script. That
  signal fires when the launching *thread* exits, not the process: `scripts/chaos.py`
  restarts killed workers from its chaos thread, and when that thread finished, every worker
  it had restarted got SIGTERM at once. With a real model this left the cluster without decode
  workers, and requests failed. Found by `scripts/chaos.py` (worker processes left
  `<defunct>`, nobody had killed them). The worker now watches its parent process instead.

Two design choices came out of the same runs. A worker greets a new control plane at once
and holds back the old session's tokens (rather than greeting only after its current step,
which on a CPU can take seconds, longer than the plane waits for a greeting). And a request
that arrives while every worker of a role is down waits for one to come back instead of
getting a 503 at once: a worker restart is a gap of a second or two.

## 8. Results

On this machine (4 CPU cores, CPU backend, SmolLM2-135M), `scripts/chaos.py`: see
[`results/cpu/m5-chaos-2026-10-03.txt`](../results/cpu/m5-chaos-2026-10-03.txt).

| Cluster | Seeds | Requests | Workers SIGKILLed | Identical to the fault-free run | Failed | Time, fault-free → chaos |
|---|---|---|---|---|---|---|
| 2 prefill + 2 decode | 0, 1, 2 | 3 × 64 | 30 | 192 of 192 | 0 | 46-48 s → 58-65 s |
| 3 colocated | 0, 1, 2 | 3 × 64 | 30 | 192 of 192 | 0 | 49-53 s → 61-66 s |

Each run starts a fresh cluster (neither run inherits the other's prefix caches), samples
with temperature 0.8 and a fixed seed per request, and streams 8 requests at a time. The
extra 12-17 s is restarts (each loads the model) and recomputation of interrupted requests on
a 4-core machine with one thread per worker. CPU only: on a GPU, recovery is exact only up to
batching's floating-point order (see 1).
