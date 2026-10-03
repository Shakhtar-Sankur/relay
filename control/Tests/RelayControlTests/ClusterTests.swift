// The router's decisions, then the whole system: relay-worker processes (prefill, decode,
// colocated) behind the Swift control plane and its OpenAI API, checked against
// transformers' output on the chat-tiny fixture.
import Foundation
import Testing

@testable import RelayControl

#if canImport(Glibc)
  import Glibc
#endif

// ---- router -------------------------------------------------------------------

@Test func sharedPrefixesStickToOneWorker() {
  let r = Router(blockSize: 16)
  let workers = (0..<4).map { WorkerSnapshot(index: $0, freeBlocks: 100, totalBlocks: 100) }
  let system = Array(0..<40)
  let picks = Set((0..<20).map { r.choosePrefill(prompt: system + [100 + $0, 7, 8], workers: workers)! })
  #expect(picks.count == 1)
}

@Test func differentPrefixesSpreadOut() {
  let r = Router(blockSize: 16)
  let workers = (0..<4).map { WorkerSnapshot(index: $0, freeBlocks: 100, totalBlocks: 100) }
  var counts = [Int: Int]()
  for i in 0..<400 { counts[r.choosePrefill(prompt: Array(repeating: i, count: 20), workers: workers)!, default: 0] += 1 }
  #expect(counts.count == 4)
  #expect(counts.values.allSatisfy { $0 > 60 })
}

@Test func aBusyWorkerLosesItsAffinity() {
  let r = Router(blockSize: 16, slack: 100)
  let prompt = Array(0..<40)
  var workers = (0..<3).map { WorkerSnapshot(index: $0, freeBlocks: 100, totalBlocks: 100) }
  let favourite = r.choosePrefill(prompt: prompt, workers: workers)!
  workers[favourite].queuedPromptTokens = 500
  let next = r.choosePrefill(prompt: prompt, workers: workers)!
  #expect(next != favourite)
  workers[favourite].healthy = true
  workers[favourite].queuedPromptTokens = 50  // within the slack: affinity wins again
  #expect(r.choosePrefill(prompt: prompt, workers: workers) == favourite)
}

@Test func decodeGoesToTheMostFreeMemoryAndSkipsDeadWorkers() {
  let r = Router(blockSize: 16)
  var workers = [WorkerSnapshot(index: 3, freeBlocks: 50, totalBlocks: 100),
                 WorkerSnapshot(index: 4, freeBlocks: 90, totalBlocks: 100, assignedBlocks: 60),
                 WorkerSnapshot(index: 5, freeBlocks: 70, totalBlocks: 100)]
  #expect(r.chooseDecode(workers: workers) == 5)
  workers[2].healthy = false
  #expect(r.chooseDecode(workers: workers) == 3)
  #expect(r.chooseDecode(workers: []) == nil)
}

// The watchdog reads the clock, then a reader thread stamps a newer heartbeat: the
// difference must be 0, not an unsigned wrap to 584 years (a bug the chaos test found).
@Test func heartbeatAgeNeverWraps() {
  #expect(secondsSince(1_000, now: 3_000_000_000) == 2.999999)
  #expect(secondsSince(5_000, now: 4_000) == 0)
}

// ---- worker processes -----------------------------------------------------------

final class WorkerProcess: @unchecked Sendable {
  let process = Process()
  let address: WorkerAddress

  let role: WorkerRoleName
  let stepDelayMs: Int

  // controlPort 0: any free port; give the old one to restart a worker in its place.
  init(role: WorkerRoleName, model: URL = chatTiny, blocks: Int = 128, controlPort: Int = 0, stepDelayMs: Int = 0) throws {
    self.role = role
    self.stepDelayMs = stepDelayMs
    let bin = URL(fileURLWithPath: CommandLine.arguments[0]).deletingLastPathComponent().appendingPathComponent("relay-worker")
    process.executableURL = bin
    process.arguments = ["--model", model.path, "--role", role.rawValue, "--blocks", "\(blocks)", "--block-size", "16",
                         "--control-port", "\(controlPort)", "--step-delay-ms", "\(stepDelayMs)"]
    // Extra worker flags for experiments, e.g. RELAY_TEST_WORKER_ARGS="--no-prefix-cache".
    if let extra = ProcessInfo.processInfo.environment["RELAY_TEST_WORKER_ARGS"] {
      process.arguments! += extra.split(separator: " ").map(String.init)
    }
    let out = Pipe()
    process.standardOutput = out
    process.standardError = FileHandle.standardError
    try process.run()
    // "relay-worker <role> control=<port> kv=<port>"
    var line = Data()
    while !line.contains(UInt8(ascii: "\n")) {
      let d = out.fileHandleForReading.availableData
      if d.isEmpty { throw ClusterError.connect("relay-worker exited") }
      line.append(d)
    }
    let text = String(decoding: line, as: UTF8.self)
    guard let port = text.firstMatch(of: /control=(\d+)/).flatMap({ Int($0.1) }) else {
      throw ClusterError.connect("unexpected worker output: \(text)")
    }
    address = WorkerAddress(role: role, host: "127.0.0.1", port: port)
  }

  func signal(_ sig: Int32) { Glibc.kill(process.processIdentifier, sig) }

  // A new process in this one's place (same role and control port), once this one is gone.
  func restart() throws -> WorkerProcess {
    try WorkerProcess(role: role, controlPort: address.port, stepDelayMs: stepDelayMs)
  }

  // Not waitUntilExit(): off the main thread on Linux it spins a run loop at 100% CPU.
  func kill() {
    guard process.isRunning else { return }
    process.terminate()
    for _ in 0..<100 where process.isRunning { usleep(10_000) }
    if process.isRunning { Glibc.kill(process.processIdentifier, SIGKILL) }
    for _ in 0..<100 where process.isRunning { usleep(10_000) }
  }

  deinit { kill() }
}

final class ClusterFixture: @unchecked Sendable {
  let workers: [WorkerProcess]
  let cluster: Cluster
  let server: HTTPServer
  let bundle: ModelBundle
  let reference: Reference

  init(roles: [WorkerRoleName], maxInFlight: Int = 256) throws {
    workers = try roles.map { try WorkerProcess(role: $0) }
    cluster = try Cluster(addresses: workers.map(\.address), maxInFlight: maxInFlight)
    bundle = try ModelBundle(directory: chatTiny)
    let api = OpenAIServer(bundle: bundle, backend: cluster)
    server = try HTTPServer(port: 0) { req, w in await api.handle(req, w) }
    server.start()
    reference = try JSONDecoder().decode(Reference.self, from: Data(contentsOf: chatTiny.appendingPathComponent("relay-reference.json")))
  }

  deinit {
    server.stop()
    cluster.shutdown()
    for w in workers { w.kill() }
  }

  func chat(_ body: String) async throws -> (Int, String) {
    try await request(port: server.port, path: "/v1/chat/completions", body: body)
  }

  // Waits until every worker reports all its KV blocks free (requests finished or cancelled).
  func waitAllFree(seconds: Double = 10) -> Bool {
    let deadline = Date().addingTimeInterval(seconds)
    while Date() < deadline {
      let s = cluster.stats()
      if s.inFlight == 0 && s.workers.allSatisfy({ $0.freeBlocks == $0.totalBlocks }) { return true }
      usleep(10_000)
    }
    return false
  }
}

func content(_ body: String, stream: Bool) -> String {
  if stream {
    return events(body).compactMap { (($0["choices"] as? [[String: Any]])?.first?["delta"] as? [String: Any])?["content"] as? String }.joined()
  }
  return ((json(body)["choices"] as? [[String: Any]])?.first?["message"] as? [String: Any])?["content"] as? String ?? ""
}

let greedy = #"{"messages":[{"role":"system","content":"Be brief."},{"role":"user","content":"What is 2+2?"}],"max_tokens":8,"temperature":0,"ignore_eos":true"#

@Suite(.serialized) struct ClusterTests {
  @Test func disaggregatedClusterMatchesTransformers() async throws {
    let f = try ClusterFixture(roles: [.prefill, .decode, .decode])
    let want = f.bundle.tokenizer.decode(f.reference.generated)
    for stream in [false, true] {
      let (status, body) = try await f.chat(greedy + #","stream":\#(stream)}"#)
      #expect(status == 200)
      #expect(content(body, stream: stream) == want)
    }
    #expect(f.waitAllFree())
  }

  @Test func colocatedClusterMatchesTransformers() async throws {
    let f = try ClusterFixture(roles: [.both, .both])
    let want = f.bundle.tokenizer.decode(f.reference.generated)
    let (status, body) = try await f.chat(greedy + "}")
    #expect(status == 200)
    #expect(content(body, stream: false) == want)
  }

  @Test func manyConcurrentRequestsAcrossTheCluster() async throws {
    let f = try ClusterFixture(roles: [.prefill, .prefill, .decode, .decode])
    let want = f.bundle.tokenizer.decode(f.reference.generated)
    let results = try await withThrowingTaskGroup(of: String.self) { group in
      for i in 0..<16 {
        group.addTask { content(try await f.chat(greedy + #","stream":\#(i % 2 == 0)}"#).1, stream: i % 2 == 0) }
      }
      var all: [String] = []
      for try await r in group { all.append(r) }
      return all
    }
    #expect(results.count == 16 && results.allSatisfy { $0 == want })
    for r in results where r != want { print("MISMATCH got \(r.debugDescription) want \(want.debugDescription)") }
    #expect(f.waitAllFree())
    // Tokens 2..8 of every request were decoded on the decode workers. (Which decode worker
    // got each request depends on timing; the router's balancing is unit-tested above.)
    // (Load reports arrive every 20 ms: wait for the one that counts the last step.)
    var decoded: UInt64 = 0
    for _ in 0..<500 {
      decoded = f.cluster.stats().workers.filter { $0.role == "decode" }.map(\.forwardTokens).reduce(0, +)
      if decoded >= 16 * 7 { break }
      usleep(10_000)
    }
    #expect(decoded >= 16 * 7)
  }

  @Test func sharedSystemPromptsHitOnePrefillWorkersCache() async throws {
    let f = try ClusterFixture(roles: [.prefill, .prefill, .decode])
    let system = String(repeating: "You are a careful assistant who answers in one short sentence. ", count: 3)
    for q in ["What is 2+2?", "Name a colour.", "What is the capital of France?", "Say hi.", "Count to three."] {
      let body = #"{"messages":[{"role":"system","content":"\#(system)"},{"role":"user","content":"\#(q)"}],"max_tokens":4,"temperature":0}"#
      let (status, _) = try await f.chat(body)
      #expect(status == 200)
    }
    #expect(f.waitAllFree())
    var prefill = f.cluster.stats().workers.filter { $0.role == "prefill" }
    for _ in 0..<500 where prefill.map(\.prefixHitTokens).reduce(0, +) < 4 * 16 {  // load reports lag up to 20 ms
      usleep(10_000)
      prefill = f.cluster.stats().workers.filter { $0.role == "prefill" }
    }
    // All five went to the same prefill worker, and four found the system prompt cached.
    #expect(prefill.filter { $0.promptTokens > 0 }.count == 1)
    #expect(prefill.map(\.prefixHitTokens).reduce(0, +) >= 4 * 16)
  }

  @Test func clientsThatLeaveFreeMemoryOnEveryWorker() async throws {
    let f = try ClusterFixture(roles: [.prefill, .decode])
    _ = try await request(port: f.server.port, path: "/v1/completions",
                          body: #"{"prompt":"Hello world","max_tokens":300,"temperature":0,"ignore_eos":true,"stream":true}"#,
                          readLimit: 400)
    #expect(f.waitAllFree())
  }

  @Test func admissionControlAnswers503WhenFull() async throws {
    let f = try ClusterFixture(roles: [.both], maxInFlight: 2)
    let body = #"{"prompt":"Hello world","max_tokens":200,"temperature":0,"ignore_eos":true}"#
    let statuses = try await withThrowingTaskGroup(of: Int.self) { group in
      for _ in 0..<8 { group.addTask { try await request(port: f.server.port, path: "/v1/completions", body: body).0 } }
      var all: [Int] = []
      for try await s in group { all.append(s) }
      return all
    }
    #expect(statuses.contains(200))
    #expect(statuses.contains(503))
    #expect(statuses.allSatisfy { $0 == 200 || $0 == 503 })
  }
}
