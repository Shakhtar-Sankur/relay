// The OpenAI-compatible API: /v1/completions and /v1/chat/completions, with or without
// streaming (server-sent events), plus /v1/models and /health.
//
// Text handling lives here, not in the engine: prompts are tokenized (chat messages are
// rendered with the model's chat template first), generated tokens are turned back into
// text as they arrive, and generation stops at the model's end tokens or at any of the
// client's stop strings. Text that might be the start of a stop string is held back until
// it is clear it is not, so a stop string never leaks into the output.
import Foundation
import RelayTokenizer

struct StopList: Decodable {
  var values: [String]
  init(from decoder: Decoder) throws {
    let c = try decoder.singleValueContainer()
    if let s = try? c.decode(String.self) {
      values = [s]
    } else {
      values = try c.decode([String].self)
    }
  }
}

struct CompletionRequest: Decodable {
  var model: String?
  var prompt: String
  var max_tokens: Int?
  var temperature: Float?
  var top_p: Float?
  var top_k: Int?
  var seed: UInt64?
  var stream: Bool?
  var stop: StopList?
  var n: Int?
  var ignore_eos: Bool?
}

struct ChatRequest: Decodable {
  var model: String?
  var messages: [ChatMessage]
  var max_tokens: Int?
  var max_completion_tokens: Int?
  var temperature: Float?
  var top_p: Float?
  var top_k: Int?
  var seed: UInt64?
  var stream: Bool?
  var stop: StopList?
  var n: Int?
  var ignore_eos: Bool?
}

struct Usage: Codable {
  var prompt_tokens: Int
  var completion_tokens: Int
  var total_tokens: Int
}

struct CompletionChoice: Codable {
  var index: Int
  var text: String
  var finish_reason: String?
}

struct CompletionResponse: Codable {
  var id: String
  var object: String
  var created: Int
  var model: String
  var choices: [CompletionChoice]
  var usage: Usage?
}

struct ChatDelta: Codable {
  var role: String?
  var content: String?
}

struct ChatChoice: Codable {
  var index: Int
  var message: ChatMessage?
  var delta: ChatDelta?
  var finish_reason: String?
}

struct ChatResponse: Codable {
  var id: String
  var object: String
  var created: Int
  var model: String
  var choices: [ChatChoice]
  var usage: Usage?
}

struct APIError: Codable {
  struct Body: Codable {
    var message: String
    var type: String
  }
  var error: Body
}

public final class OpenAIServer: Sendable {
  let bundle: ModelBundle
  let backend: GenerationBackend

  public init(bundle: ModelBundle, backend: GenerationBackend) {
    self.bundle = bundle
    self.backend = backend
  }

  public func handle(_ req: HTTPRequest, _ w: ResponseWriter) async {
    switch (req.method, req.path) {
    case ("GET", "/health"):
      w.respondJSON(["status": "ok"])
    case ("GET", "/v1/models"):
      struct Model: Codable { var id: String; var object = "model"; var owned_by = "relay" }
      struct Models: Codable { var object = "list"; var data: [Model] }
      w.respondJSON(Models(data: [Model(id: bundle.name)]))
    case ("POST", "/v1/completions"):
      await completions(req, w)
    case ("POST", "/v1/chat/completions"):
      await chat(req, w)
    case (_, "/v1/completions"), (_, "/v1/chat/completions"), (_, "/v1/models"), (_, "/health"):
      fail(w, 405, "method not allowed")
    default:
      fail(w, 404, "no route for \(req.path)")
    }
  }

  // 503 when the cluster is full or has no worker for the request, 400 when the engine
  // refused it, 500 otherwise.
  static func status(for error: Error) -> Int {
    if error is AdmissionError { return 503 }
    if case ClusterError.noWorker = error { return 503 }
    if case GenerationError.rejected = error { return 400 }
    return 500
  }

  func fail(_ w: ResponseWriter, _ status: Int, _ message: String) {
    w.respondJSON(status: status, APIError(error: .init(message: message, type: status >= 500 ? "server_error" : "invalid_request_error")))
  }

  struct Plan {
    var prompt: [Int]
    var sampling: Sampling
    var stops: [String]
    var stream: Bool
  }

  func plan(prompt: [Int], maxTokens: Int?, defaultMax: Int, temperature: Float?, topP: Float?, topK: Int?, seed: UInt64?,
            stop: StopList?, n: Int?, ignoreEOS: Bool?, stream: Bool?) throws -> Plan {
    if let n, n != 1 { throw GenerationError.rejected("n must be 1") }
    if prompt.isEmpty { throw GenerationError.rejected("empty prompt") }
    var s = Sampling()
    s.temperature = temperature ?? 1.0
    s.topP = topP ?? 1.0
    s.topK = topK ?? 0
    s.seed = seed ?? UInt64.random(in: 0...UInt64.max)
    s.ignoreEOS = ignoreEOS ?? false
    let room = bundle.maxContext - prompt.count
    if room <= 0 { throw GenerationError.rejected("prompt of \(prompt.count) tokens fills the \(bundle.maxContext)-token context") }
    s.maxTokens = min(maxTokens ?? defaultMax, room)
    if s.maxTokens <= 0 { throw GenerationError.rejected("max_tokens must be positive") }
    return Plan(prompt: prompt, sampling: s, stops: (stop?.values ?? []).filter { !$0.isEmpty }, stream: stream ?? false)
  }

  // Runs one generation, calling `emit` with each new piece of text. Returns the number
  // of tokens generated and why generation ended. `emit` returning false (the client went
  // away) ends the stream, which cancels the request in the engine.
  func run(_ p: Plan, onStart: () -> Bool = { true }, emit: (String) -> Bool) async throws
    -> (tokens: Int, finish: FinishReason)
  {
    var started = false
    var decoder = StreamingDecoder(bundle.tokenizer)
    var held = ""
    var count = 0
    var finish = FinishReason.length
    let longestStop = p.stops.map(\.count).max() ?? 0
    loop: for try await t in backend.generate(prompt: p.prompt, sampling: p.sampling) {
      // Headers go out with the first token, so an error before it (overload, a refused
      // request) can still be a proper HTTP status.
      if !started {
        started = true
        if !onStart() { break }
      }
      count += 1
      if bundle.stopTokenIDs.contains(t.token) && !p.sampling.ignoreEOS {
        finish = .stop
        break
      }
      held += decoder.push(t.token)
      if t.finish != nil { held += decoder.finish() }
      if !p.stops.isEmpty {
        if let r = p.stops.compactMap({ held.range(of: $0) }).min(by: { $0.lowerBound < $1.lowerBound }) {
          if !emit(String(held[..<r.lowerBound])) { break }
          held = ""
          finish = .stop
          break loop
        }
        // Hold back a tail that could still grow into a stop string.
        var keep = 0
        for k in stride(from: min(longestStop - 1, held.count), to: 0, by: -1) {
          let tail = held.suffix(k)
          if p.stops.contains(where: { $0.hasPrefix(tail) }) {
            keep = k
            break
          }
        }
        let out = String(held.dropLast(keep))
        held = String(held.suffix(keep))
        if !out.isEmpty && !emit(out) { break }
      } else if !held.isEmpty {
        if !emit(held) { break }
        held = ""
      }
      if let f = t.finish {
        finish = f
        break
      }
    }
    if !started { _ = onStart() }
    if !held.isEmpty { _ = emit(held) }
    return (count, finish)
  }

  static func now() -> Int { Int(Date().timeIntervalSince1970) }
  static func newID(_ prefix: String) -> String { prefix + UUID().uuidString.replacingOccurrences(of: "-", with: "").lowercased().prefix(24) }

  func sse(_ w: ResponseWriter, _ object: some Encodable) -> Bool {
    let enc = JSONEncoder()
    enc.outputFormatting = [.withoutEscapingSlashes]
    guard let json = try? enc.encode(object) else { return true }
    var d = Data("data: ".utf8)
    d.append(json)
    d.append(Data("\n\n".utf8))
    return w.write(d)
  }

  func startStream(_ w: ResponseWriter) {
    w.start(status: 200, headers: [("Content-Type", "text/event-stream"), ("Cache-Control", "no-cache")])
  }

  func completions(_ req: HTTPRequest, _ w: ResponseWriter) async {
    let r: CompletionRequest
    let p: Plan
    do {
      r = try JSONDecoder().decode(CompletionRequest.self, from: req.body)
      p = try plan(prompt: bundle.tokenizer.encode(r.prompt), maxTokens: r.max_tokens, defaultMax: 16,
                   temperature: r.temperature, topP: r.top_p, topK: r.top_k, seed: r.seed, stop: r.stop, n: r.n,
                   ignoreEOS: r.ignore_eos, stream: r.stream)
    } catch {
      return fail(w, 400, "\(error)")
    }
    let id = OpenAIServer.newID("cmpl-"), created = OpenAIServer.now()
    do {
      if p.stream {
        let result = try await run(p, onStart: { startStream(w); return true }) { text in
          sse(w, CompletionResponse(id: id, object: "text_completion", created: created, model: bundle.name,
                                    choices: [CompletionChoice(index: 0, text: text, finish_reason: nil)]))
        }
        _ = sse(w, CompletionResponse(id: id, object: "text_completion", created: created, model: bundle.name,
                                      choices: [CompletionChoice(index: 0, text: "", finish_reason: result.finish.rawValue)],
                                      usage: Usage(prompt_tokens: p.prompt.count, completion_tokens: result.tokens,
                                                   total_tokens: p.prompt.count + result.tokens)))
        w.write(Data("data: [DONE]\n\n".utf8))
      } else {
        var text = ""
        let result = try await run(p) { text += $0; return true }
        w.respondJSON(CompletionResponse(id: id, object: "text_completion", created: created, model: bundle.name,
                                         choices: [CompletionChoice(index: 0, text: text, finish_reason: result.finish.rawValue)],
                                         usage: Usage(prompt_tokens: p.prompt.count, completion_tokens: result.tokens,
                                                      total_tokens: p.prompt.count + result.tokens)))
      }
    } catch {
      if w.started {
        _ = sse(w, APIError(error: .init(message: "\(error)", type: "server_error")))
      } else {
        fail(w, OpenAIServer.status(for: error), "\(error)")
      }
    }
  }

  func chat(_ req: HTTPRequest, _ w: ResponseWriter) async {
    guard let template = bundle.chatTemplate else { return fail(w, 400, "\(ChatTemplateError.noTemplate)") }
    let r: ChatRequest
    let p: Plan
    do {
      r = try JSONDecoder().decode(ChatRequest.self, from: req.body)
      // As transformers does: the rendered template already holds any special tokens.
      let ids = bundle.tokenizer.encode(template.render(r.messages), addSpecialTokens: false)
      p = try plan(prompt: ids, maxTokens: r.max_completion_tokens ?? r.max_tokens, defaultMax: 256,
                   temperature: r.temperature, topP: r.top_p, topK: r.top_k, seed: r.seed, stop: r.stop, n: r.n,
                   ignoreEOS: r.ignore_eos, stream: r.stream)
    } catch {
      return fail(w, 400, "\(error)")
    }
    let id = OpenAIServer.newID("chatcmpl-"), created = OpenAIServer.now()
    func chunk(_ delta: ChatDelta, _ finish: String?, _ usage: Usage? = nil) -> ChatResponse {
      ChatResponse(id: id, object: "chat.completion.chunk", created: created, model: bundle.name,
                   choices: [ChatChoice(index: 0, message: nil, delta: delta, finish_reason: finish)], usage: usage)
    }
    do {
      if p.stream {
        let result = try await run(p, onStart: {
          startStream(w)
          return sse(w, chunk(ChatDelta(role: "assistant", content: ""), nil))
        }) { text in sse(w, chunk(ChatDelta(role: nil, content: text), nil)) }
        _ = sse(w, chunk(ChatDelta(role: nil, content: nil), result.finish.rawValue,
                         Usage(prompt_tokens: p.prompt.count, completion_tokens: result.tokens,
                               total_tokens: p.prompt.count + result.tokens)))
        w.write(Data("data: [DONE]\n\n".utf8))
      } else {
        var text = ""
        let result = try await run(p) { text += $0; return true }
        w.respondJSON(ChatResponse(id: id, object: "chat.completion", created: created, model: bundle.name,
                                   choices: [ChatChoice(index: 0, message: ChatMessage(role: "assistant", content: text),
                                                        delta: nil, finish_reason: result.finish.rawValue)],
                                   usage: Usage(prompt_tokens: p.prompt.count, completion_tokens: result.tokens,
                                                total_tokens: p.prompt.count + result.tokens)))
      }
    } catch {
      if w.started {
        _ = sse(w, APIError(error: .init(message: "\(error)", type: "server_error")))
      } else {
        fail(w, OpenAIServer.status(for: error), "\(error)")
      }
    }
  }
}
