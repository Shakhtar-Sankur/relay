# M2 walkthrough: the KV transfer engine

Files: `engine/include/relay/transport.h`, `engine/src/transport.cpp`,
`engine/include/relay/kv_transfer.h`, `engine/src/kv_transfer.cpp`,
`tests/test_transfer.cpp`, `tools/relay_transfer_bench.cpp`.

## 1. The problem

After prefill, the request's KV cache (every layer's keys and values for every prompt
token) lives on the prefill worker. The decode worker needs all of it before it can
generate token 2. For TinyLlama-1.1B that is 22 layers x 2 x 4 KV heads x 64 dims x 2 bytes
= 22.5 KB per token, so 46 MB for a 2,048-token prompt. If it is sent after prefill
finishes, the user waits for prefill and then for the transfer.

## 2. Layer streaming

The forward pass computes layer 0 for all tokens, then layer 1, and so on. Layer l's K and
V are final as soon as layer l's projection and RoPE are done; nothing later changes them.
So the backend tells an observer `kv_written(l)` right there, and a sender thread ships
layer l while the GPU computes layer l + 1. At the end only the last layer or two are
still in flight.

*Q: How does the sender know the GPU has actually written layer l?* The CUDA backend
records an event on the compute stream after the KV write kernel of layer l. The sender
calls `wait_kv_written(l)` (cudaEventSynchronize) before copying, and copies on a second
stream so it never queues behind the forward pass.

*Q: Why not send each block as soon as it is full instead?* All tokens of a chunk are
processed together, layer by layer: a block's layer-l data is complete exactly when layer l
is. Per-layer is the natural unit, and it gives one large message per layer instead of
many small ones.

## 3. Wire format

Frames of a 32-byte header plus payload: Hello (once: model and cache layout fingerprint),
Begin (request, sampling parameters, prompt), Layer (layer, first block index, K blocks
then V blocks), End (first token, finished flag), Abort.

*Q: Why send the prompt and sampling parameters?* The decode worker must be able to
recompute the prompt (if it has no room, or after preemption) and must sample tokens
2..n exactly as one engine would. With the seed and the token index, it does.

*Q: Why do frames name "the request's i-th block" and not block ids?* The two workers
allocate independently; ids on one side mean nothing on the other.

## 4. Zero copies

CPU cache: the sender builds an iovec list pointing straight into the cache and calls
`sendmsg` once; the receiver `recv`s each block directly into its slot. No intermediate
buffer on either side. GPU cache: gathered into a host buffer by `cudaMemcpyAsync` on the
copy stream; true zero-copy would need GPUDirect RDMA, which a T4 on Colab does not have.

## 5. The deadlock the tests found

First version: when a Begin arrived and the decode cache was full, the receiver waited
for blocks. The test with a small decode cache hung. Why: frames of several requests
share one ordered connection. The receiver was stuck on B's Begin, and A's End frame was
behind it in the stream. A's blocks were reserved, but A could never be handed to the
engine to run and finish, so its blocks were never freed. B was waiting for A, and A was
waiting behind B.

Fix: the receiver never waits. A reservation only succeeds when no request is queued in
the engine and every running request still has a free block to grow into. Otherwise the
request's KV frames are read and dropped, and the engine recomputes the prompt locally
(`add_recompute`). Deterministic sampling makes the result identical. Production systems
do the same thing: recompute as the fallback when a transfer cannot complete.

*Q: Isn't recompute wasteful?* It only happens under memory pressure, which the scheduler
(M4) tries to avoid by routing requests to decode workers with room. The fallback is about
never deadlocking and never being wrong.

## 6. Failure handling

- The prefill side frees a request's blocks only after its last frame has been sent (the
  End frame's callback runs on the sender thread).
- If the connection breaks, the receiver returns the blocks of every request whose End
  did not arrive (`a_lost_sender_returns_its_blocks`).
- If the sender runs a different model or cache layout, the Hello check refuses it
  (`a_different_model_is_refused`).

## 7. Transports

TCP: `sendmsg` with iovecs, TCP_NODELAY, 8 MB socket buffers. Shared memory: one byte ring
per direction in a POSIX shm segment, a writer-owned `head` and reader-owned `tail` counter
(monotonic, index = counter % capacity), release on publish and acquire on read. Waiting
spins, then yields, then sleeps.

*Q: Why is shm slower than TCP in the CPU streaming benchmark?* The receiver spin-waits,
taking a core from the forward pass on a 4-vCPU machine. With dedicated cores (or a GPU
doing the compute) that cost disappears; the raw benchmark between two idle processes
shows shm 2-3x faster than TCP.

## 8. Tests

`test_transfer`: byte-exact streams on both transports (including messages far larger
than the ring); closing wakes a blocked reader; layers are reported during the forward
pass; prefill/decode split equals one engine over TCP, over shm with chunked prefill, and
under decode-side memory pressure (bit-exact logits on CPU); lost sender; wrong model.
Run 30 times in a row and under ThreadSanitizer without a failure or a report.
