# M3 walkthrough: the Swift control plane

Files: `Package.swift`, `bridge/` (the C++ surface Swift calls), `control/Sources/RelayTokenizer`,
`control/Sources/RelayControl` (model bundle and chat templates, engine driver, HTTP server,
OpenAI API), `control/Sources/relay-server`, `control/Tests`.

## 1. Swift calling C++ (`bridge/include/RelayBridge.h`)

Swift 6 imports C++ headers directly (`.interoperabilityMode(.Cxx)`): structs become Swift
value types, `std::string` and `std::vector<int32_t>` get Swift overlays. relay does not hand
Swift its engine headers: they use `std::function`, `unique_ptr` and templates that Swift
either cannot import or imports awkwardly. Instead a small facade:

- `LocalWorker` owns the weights, a backend and an `Engine` behind a pimpl. It is marked
  `SWIFT_SHARED_REFERENCE(relaybridge_retain, relaybridge_release)`, so Swift treats it as a
  class and its intrusive reference count decides when it is deleted.
- Every method catches C++ exceptions and returns an error string: an exception crossing
  into Swift would terminate the process.

*Q: Why not wrap the engine in C and call the C API?* That works too; C++ interop keeps the
types (strings, vectors, structs with defaults) without hand-written conversion code.

*Q: What broke while building it?* Two things. The retain/release functions must be found
at global scope, not inside the namespace. And `<swift/bridging>` is not always on the
include path, so the header spells out the same clang attributes when it is missing.

## 2. The tokenizer (`Tokenizer.swift`)

Reads `tokenizer.json` and reproduces the `tokenizers` library for three pipelines:
GPT-2 byte-level BPE with per-digit splitting (SmolLM2), NFC + model regex + byte-level
(Qwen2), and SentencePiece-style BPE with `▁` and byte fallback (TinyLlama). Steps:
added/special tokens are split out first; normalizers; pre-tokenizers (each splits every
piece further); BPE (repeatedly merge the adjacent pair with the lowest merge rank);
post-processor tokens (`<s>`). Anything else in a tokenizer.json is rejected at load time.

*Q: The detail that matters most?* Regex semantics. Hugging Face matches Unicode scalars;
Swift's `Regex` matches grapheme clusters by default, where "\r\n" is one character. The
regexes run with `.matchingSemantics(.unicodeScalar)`.

*Q: How is it tested?* 25 strings (emoji, CJK, combining accents, contractions, digits,
Windows line ends, special tokens) through three small tokenizers trained with each pipeline
(committed) and the three real models: ids and decoded text equal Hugging Face's, every
case. Removing one contraction from the GPT-2 regex fails the test.

`StreamingDecoder` turns tokens into text as they arrive, holding back a character whose
bytes are split across tokens (decode a short window, emit the difference unless it ends in
U+FFFD), the method text-generation-inference uses.

## 3. Chat templates (`ModelBundle.swift`)

Instead of a Jinja interpreter, the template text is matched to a known format (ChatML,
Zephyr, Llama 3), including the default system prompt some templates insert. Tested against
transformers' `apply_chat_template` for every model. The rendered text is tokenized without
adding special tokens, as transformers does.

## 4. Driving the engine (`Generation.swift`)

`GenerationBackend` is the one interface the API uses: token ids in, an
`AsyncThrowingStream` of tokens out. `LocalEngine` implements it over `LocalWorker`: one
thread runs the engine's step loop (it sleeps on a condition variable when idle); requests
are added from any task; each request's tokens go to its own stream. When a consumer stops
iterating (the client hung up, or a stop string matched), the stream's `onTermination`
cancels the request in the engine, which frees its KV blocks (`Engine::cancel`).
M4's cluster implements the same protocol with remote workers.

## 5. The HTTP server and the OpenAI API

`HTTPServer` is POSIX sockets: an accept thread, a thread per connection, the async handler
run as a Task. Responses with no Content-Length end with the connection, which is how SSE is
sent. `OpenAIServer` implements `/v1/completions`, `/v1/chat/completions` (streaming and
not), `/v1/models`, `/health`.

Stop handling: the model's end tokens (config, generation config and the tokenizer's eos,
since chat models often end turns with a token the base config does not list) and the
client's stop strings. Text that could be the beginning of a stop string is held back until
it is not, so a stop string never reaches the client.

*Q: Why not SwiftNIO?* Disk and dependency weight in this environment. The handler and
response-writer interfaces are small enough to move onto NIO (or Hummingbird) without
touching the API code.

## 6. The bug the tests found

The first server test run hung. swift-testing runs tests in parallel on Swift's cooperative
thread pool (one thread per core); each test blocked a pool thread in `recv()` waiting for
the server, and the server's handlers, which are async tasks, needed those same threads.
Nothing could run. Blocking I/O does not belong on the cooperative pool: the test client
now does its socket I/O on dedicated threads, and the server tests run one at a time.

## 7. Tests (`swift test`)

- tokenizer: 3 trained tokenizers and 3 real models, 25 cases each; streaming decode;
  unsupported pipelines rejected
- chat templates: every rendering equals transformers'
- server on the `chat-tiny` fixture (a tiny Llama, a real tokenizer, a ChatML template):
  the chat completion equals transformers' greedy output token for token, streaming equals
  non-streaming, stop strings cut and never leak (and free the cache), 12 concurrent requests
  are batched and each correct, a client that hangs up frees its request, bad requests get
  4xx errors
