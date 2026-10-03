// The control plane end to end: chat templates against transformers, and the
// OpenAI-compatible server against the engine's (and transformers') own output on the
// chat-tiny fixture: a tiny Llama with a real tokenizer and a ChatML template.
import Foundation
import Testing

@testable import RelayControl
@testable import RelayTokenizer

#if canImport(Glibc)
  import Glibc
#endif

let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
  .deletingLastPathComponent().deletingLastPathComponent()
let chatTiny = root.appendingPathComponent("tests/fixtures/chat-tiny")

struct ChatCase: Decodable {
  let messages: [ChatMessage]
  let text: String
}

struct Reference: Decodable {
  let prompt: [Int]
  let generated: [Int]
  let text: String
}

func chatCases(_ dir: URL) throws -> [ChatCase] {
  try JSONDecoder().decode([ChatCase].self, from: Data(contentsOf: dir.appendingPathComponent("relay-chat-cases.json")))
}

@Test func chatTemplateMatchesTransformers() throws {
  var dirs = [chatTiny]
  if let d = ProcessInfo.processInfo.environment["RELAY_MODELS_DIR"] {
    for m in try FileManager.default.contentsOfDirectory(atPath: d).sorted() {
      let u = URL(fileURLWithPath: d).appendingPathComponent(m)
      if FileManager.default.fileExists(atPath: u.appendingPathComponent("relay-chat-cases.json").path) { dirs.append(u) }
    }
  }
  for dir in dirs {
    let bundle = try ModelBundle(directory: dir)
    let template = try #require(bundle.chatTemplate)
    for c in try chatCases(dir) {
      #expect(template.render(c.messages) == c.text, "\(dir.lastPathComponent): \(template.render(c.messages).debugDescription)")
    }
    print("  \(dir.lastPathComponent): \(template.format)")
  }
}

// ---- a minimal HTTP client --------------------------------------------------

func blockingRequest(port: Int, method: String, path: String, body: String, readLimit: Int?) throws -> (Int, String) {
  let fd = socket(AF_INET, Int32(SOCK_STREAM.rawValue), 0)
  defer { close(fd) }
  var addr = sockaddr_in()
  addr.sin_family = sa_family_t(AF_INET)
  addr.sin_port = in_port_t(UInt16(port).bigEndian)
  inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr)
  let ok = withUnsafePointer(to: &addr) {
    $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { connect(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size)) }
  }
  guard ok == 0 else { throw HTTPServerError.socket("connect") }
  let req = "\(method) \(path) HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: \(body.utf8.count)\r\n\r\n\(body)"
  _ = req.withCString { send(fd, $0, strlen($0), 0) }
  var data = [UInt8]()
  var buf = [UInt8](repeating: 0, count: 65536)
  while true {
    let n = recv(fd, &buf, buf.count, 0)
    if n <= 0 { break }
    data += buf[0..<n]
    if let limit = readLimit, data.count >= limit { break }  // hang up early
  }
  let text = String(decoding: data, as: UTF8.self)
  let parts = text.components(separatedBy: "\r\n\r\n")
  let status = Int(parts[0].split(separator: " ").dropFirst().first ?? "0") ?? 0
  return (status, parts.dropFirst().joined(separator: "\r\n\r\n"))
}

// Blocking socket I/O must not run on Swift's cooperative thread pool: with every pool
// thread waiting in recv(), the server's own async handlers could never run. Each request
// therefore runs on a thread of its own.
func request(port: Int, method: String = "POST", path: String, body: String = "", readLimit: Int? = nil) async throws
  -> (Int, String)
{
  try await withCheckedThrowingContinuation { (c: CheckedContinuation<(Int, String), Error>) in
    Thread {
      c.resume(with: Result { try blockingRequest(port: port, method: method, path: path, body: body, readLimit: readLimit) })
    }.start()
  }
}

func json(_ s: String) -> [String: Any] {
  (try? JSONSerialization.jsonObject(with: Data(s.utf8)) as? [String: Any]) ?? [:]
}

// The SSE stream's chunks, without the final [DONE].
func events(_ body: String) -> [[String: Any]] {
  body.components(separatedBy: "\n\n").compactMap { line in
    guard line.hasPrefix("data: "), !line.hasPrefix("data: [DONE]") else { return nil }
    return json(String(line.dropFirst(6)))
  }
}

final class Fixture: @unchecked Sendable {
  let bundle: ModelBundle
  let engine: LocalEngine
  let server: HTTPServer
  let reference: Reference

  init(blocks: Int = 256) throws {
    bundle = try ModelBundle(directory: chatTiny)
    engine = try LocalEngine(modelDirectory: chatTiny.path, blocks: blocks, blockSize: 16, maxBatchTokens: 256)
    let api = OpenAIServer(bundle: bundle, backend: engine)
    server = try HTTPServer(port: 0) { req, w in await api.handle(req, w) }
    server.start()
    reference = try JSONDecoder().decode(Reference.self, from: Data(contentsOf: chatTiny.appendingPathComponent("relay-reference.json")))
  }

  deinit {
    server.stop()
    engine.shutdown()
  }
}

let conversation = #"[{"role":"system","content":"Be brief."},{"role":"user","content":"What is 2+2?"}]"#

@Suite(.serialized) struct ServerTests {
  @Test func chatCompletionMatchesTransformersGreedyOutput() async throws {
    let f = try Fixture()
    // The rendered prompt tokenizes exactly as transformers tokenized it.
    let ids = f.bundle.tokenizer.encode(f.bundle.chatTemplate!.render([
      ChatMessage(role: "system", content: "Be brief."), ChatMessage(role: "user", content: "What is 2+2?"),
    ]), addSpecialTokens: false)
    #expect(ids == f.reference.prompt)
    let want = f.bundle.tokenizer.decode(f.reference.generated)

    let (status, body) = try await request(port: f.server.port, path: "/v1/chat/completions",
                                     body: #"{"messages":\#(conversation),"max_tokens":8,"temperature":0,"ignore_eos":true}"#)
    #expect(status == 200)
    let r = json(body)
    let choice = (r["choices"] as? [[String: Any]])?.first
    let content = (choice?["message"] as? [String: Any])?["content"] as? String
    #expect(content == want)
    #expect(choice?["finish_reason"] as? String == "length")
    let usage = r["usage"] as? [String: Any]
    #expect(usage?["prompt_tokens"] as? Int == f.reference.prompt.count)
    #expect(usage?["completion_tokens"] as? Int == 8)
  }

  @Test func streamingDeliversTheSameText() async throws {
    let f = try Fixture()
    let want = f.bundle.tokenizer.decode(f.reference.generated)
    let (status, body) = try await request(port: f.server.port, path: "/v1/chat/completions",
                                     body: #"{"messages":\#(conversation),"max_tokens":8,"temperature":0,"ignore_eos":true,"stream":true}"#)
    #expect(status == 200)
    #expect(body.hasSuffix("data: [DONE]\n\n"))
    let chunks = events(body)
    let text = chunks.compactMap { (($0["choices"] as? [[String: Any]])?.first?["delta"] as? [String: Any])?["content"] as? String }.joined()
    #expect(text == want)
    #expect((chunks.last?["choices"] as? [[String: Any]])?.first?["finish_reason"] as? String == "length")
    #expect(chunks.count > 3)
  }

  @Test func stopStringsCutTheOutputAndNeverLeak() async throws {
    let f = try Fixture()
    let full = f.bundle.tokenizer.decode(f.reference.generated)
    // A stop string taken from the middle of the expected output.
    let chars = Array(full)
    let stop = String(chars[(chars.count / 2)..<min(chars.count, chars.count / 2 + 3)])
    let cut = String(full[..<full.range(of: stop)!.lowerBound])
    for stream in [false, true] {
      let (_, body) = try await request(port: f.server.port, path: "/v1/chat/completions",
                                  body: #"{"messages":\#(conversation),"max_tokens":8,"temperature":0,"ignore_eos":true,"stream":\#(stream),"stop":[\#(String(data: try JSONEncoder().encode(stop), encoding: .utf8)!)]}"#)
      let text: String
      let finish: String?
      if stream {
        let chunks = events(body)
        text = chunks.compactMap { (($0["choices"] as? [[String: Any]])?.first?["delta"] as? [String: Any])?["content"] as? String }.joined()
        finish = (chunks.last?["choices"] as? [[String: Any]])?.first?["finish_reason"] as? String
      } else {
        let choice = (json(body)["choices"] as? [[String: Any]])?.first
        text = (choice?["message"] as? [String: Any])?["content"] as? String ?? ""
        finish = choice?["finish_reason"] as? String
      }
      #expect(text == cut, "stream=\(stream): \(text.debugDescription) vs \(cut.debugDescription)")
      #expect(finish == "stop")
    }
    // The stopped request was cancelled in the engine: all its blocks are free again.
    for _ in 0..<100 where f.engine.freeBlocks != 256 { usleep(10000) }
    #expect(f.engine.freeBlocks == 256)
  }

  @Test func concurrentRequestsAreBatchedAndEachCorrect() async throws {
    let f = try Fixture()
    let want = f.bundle.tokenizer.decode(f.reference.generated)
    let port = f.server.port
    let results = try await withThrowingTaskGroup(of: String.self) { group in
      for i in 0..<12 {
        group.addTask {
          let (_, body) = try await request(port: port, path: "/v1/chat/completions",
                                      body: #"{"messages":\#(conversation),"max_tokens":8,"temperature":0,"ignore_eos":true,"stream":\#(i % 2 == 0)}"#)
          if i % 2 == 0 {
            return events(body).compactMap { (($0["choices"] as? [[String: Any]])?.first?["delta"] as? [String: Any])?["content"] as? String }.joined()
          }
          return ((json(body)["choices"] as? [[String: Any]])?.first?["message"] as? [String: Any])?["content"] as? String ?? ""
        }
      }
      var all: [String] = []
      for try await r in group { all.append(r) }
      return all
    }
    #expect(results.count == 12)
    #expect(results.allSatisfy { $0 == want })
  }

  @Test func aClientThatHangsUpFreesItsRequest() async throws {
    let f = try Fixture()
    // Ask for many tokens, read the first bytes of the stream, hang up.
    _ = try await request(port: f.server.port, path: "/v1/completions",
                    body: #"{"prompt":"Hello world","max_tokens":400,"temperature":0,"ignore_eos":true,"stream":true}"#, readLimit: 400)
    for _ in 0..<300 where f.engine.freeBlocks != 256 { usleep(10000) }
    #expect(f.engine.freeBlocks == 256)
  }

  @Test func badRequestsGetErrors() async throws {
    let f = try Fixture()
    #expect(try await request(port: f.server.port, path: "/v1/chat/completions", body: "{not json").0 == 400)
    #expect(try await request(port: f.server.port, path: "/v1/completions", body: #"{"prompt":"hi","n":2}"#).0 == 400)
    #expect(try await request(port: f.server.port, path: "/v1/completions", body: #"{"prompt":""}"#).0 == 400)
    #expect(try await request(port: f.server.port, method: "GET", path: "/nowhere").0 == 404)
    #expect(try await request(port: f.server.port, method: "GET", path: "/v1/completions").0 == 405)
    let (s, models) = try await request(port: f.server.port, method: "GET", path: "/v1/models")
    #expect(s == 200 && models.contains("chat-tiny"))
  }
}
