// Fault tolerance with real relay-worker processes: workers are killed (SIGKILL), frozen
// (SIGSTOP) or stalled while requests run through them, and every request must still
// produce exactly the tokens of a fault-free run: same prompt, same seed, same output.
// The reference is relay's engine in this process (LocalEngine), with no faults at all.
import Foundation
import Synchronization
import Testing

@testable import RelayControl

#if canImport(Glibc)
  import Glibc
#endif

enum Oracle {
  static let engine: LocalEngine = try! LocalEngine(modelDirectory: chatTiny.path, blocks: 512, blockSize: 16, maxBatchTokens: 256)
}

func reference(_ prompt: [Int], _ s: Sampling) async throws -> [Int] {
  var out: [Int] = []
  for try await t in Oracle.engine.generate(prompt: prompt, sampling: s) { out.append(t.token) }
  return out
}

func randomPrompt(_ seed: UInt64, length: Int) -> [Int] {
  var x = seed &* 0x9E37_79B9_7F4A_7C15 &+ 1
  return (0..<length).map { _ in
    x = x &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
    return 3 + Int((x >> 33) % 400)
  }
}

func sampled(seed: UInt64, maxTokens: Int) -> Sampling {
  var s = Sampling()
  s.temperature = 0.8
  s.seed = seed
  s.maxTokens = maxTokens
  s.ignoreEOS = true
  return s
}

// Reads a stream to the end, running `action` once `after` tokens have arrived. Checks the
// client saw every index exactly once, in order: no gap, no repeat across a recovery.
func collect(_ stream: AsyncThrowingStream<GeneratedToken, Error>, after: Int = -1, _ action: () -> Void = {}) async throws
  -> [Int]
{
  var tokens: [Int] = []
  var indices: [Int] = []
  for try await t in stream {
    tokens.append(t.token)
    indices.append(t.index)
    if tokens.count == after { action() }
  }
  #expect(indices == Array(0..<tokens.count))
  return tokens
}

final class FaultCluster: @unchecked Sendable {
  private let lock = NSLock()
  private var processes: [WorkerProcess]  // index = the worker's index in the cluster
  let cluster: Cluster

  init(_ roles: [WorkerRoleName], stepDelayMs: [Int]? = nil, blocks: [Int]? = nil, options: ClusterOptions = ClusterOptions())
    throws
  {
    processes = try roles.enumerated().map { i, role in
      try WorkerProcess(role: role, blocks: blocks?[i] ?? 128, stepDelayMs: stepDelayMs?[i] ?? 0)
    }
    cluster = try Cluster(addresses: processes.map(\.address), options: options)
  }

  deinit {
    cluster.shutdown()
    for p in processes { p.kill() }
  }

  func process(_ i: Int) -> WorkerProcess { lock.withLock { processes[i] } }
  func kill(_ i: Int) { process(i).signal(SIGKILL) }
  func restart(_ i: Int) throws {
    let old = process(i)
    old.kill()
    let new = try old.restart()
    lock.withLock { processes[i] = new }
  }

  // A worker of the (only) request in flight.
  func current(_ pick: (Cluster.Request) -> Int?) -> Int? {
    cluster.state.withLock { s in s.requests.values.first.flatMap(pick) }
  }

  func waitUntil(seconds: Double = 10, _ condition: () -> Bool) -> Bool {
    let deadline = Date().addingTimeInterval(seconds)
    while Date() < deadline {
      if condition() { return true }
      usleep(10_000)
    }
    return false
  }
}

@Suite(.serialized) struct FaultToleranceTests {
  @Test func aDecodeWorkerKilledMidStreamChangesNothing() async throws {
    let f = try FaultCluster([.prefill, .decode, .decode], stepDelayMs: [0, 5, 5])
    let p = randomPrompt(1, length: 30), s = sampled(seed: 11, maxTokens: 60)
    let want = try await reference(p, s)
    var killed: Int?
    let got = try await collect(f.cluster.generate(prompt: p, sampling: s), after: 10) {
      killed = f.current { $0.decode }
      if let k = killed { f.kill(k) }
    }
    #expect(killed != nil)
    #expect(got == want)
    let st = f.cluster.stats()
    #expect(st.workersLost == 1)
    #expect(st.recovered >= 1)
  }

  @Test func aPrefillWorkerKilledBeforeTheFirstTokenChangesNothing() async throws {
    let f = try FaultCluster([.prefill, .prefill, .decode], stepDelayMs: [300, 300, 0])
    let p = randomPrompt(2, length: 40), s = sampled(seed: 12, maxTokens: 8)
    let want = try await reference(p, s)
    let stream = f.cluster.generate(prompt: p, sampling: s)  // placed at once
    let killed = try #require(f.current { $0.prefill })
    f.kill(killed)  // well inside its 300 ms before the first step
    let got = try await collect(stream)
    #expect(got == want)
    #expect(f.cluster.stats().workersLost == 1)
  }

  @Test func withNoPrefillWorkerLeftADecodeWorkerServesTheWholeRequest() async throws {
    let f = try FaultCluster([.prefill, .decode])
    f.kill(0)
    #expect(f.waitUntil { f.cluster.stats().workersLost == 1 })
    let p = randomPrompt(3, length: 25), s = sampled(seed: 13, maxTokens: 20)
    #expect(try await collect(f.cluster.generate(prompt: p, sampling: s)) == (try await reference(p, s)))

    // The prefill worker comes back in its old place and is used again.
    try f.restart(0)
    #expect(f.waitUntil { f.cluster.stats().workers[0].alive })
    let p2 = randomPrompt(4, length: 25), s2 = sampled(seed: 14, maxTokens: 20)
    #expect(try await collect(f.cluster.generate(prompt: p2, sampling: s2)) == (try await reference(p2, s2)))
    let st = f.cluster.stats()
    #expect(st.workersRejoined == 1)
    // The restarted process did the prefill (its load reports arrive every 20 ms).
    #expect(f.waitUntil { f.cluster.stats().workers[0].promptTokens > 0 })
  }

  @Test func aColocatedWorkerKilledMidStreamChangesNothing() async throws {
    let f = try FaultCluster([.both, .both], stepDelayMs: [5, 5])
    let p = randomPrompt(5, length: 20), s = sampled(seed: 15, maxTokens: 50)
    let want = try await reference(p, s)
    var killed: Int?
    let got = try await collect(f.cluster.generate(prompt: p, sampling: s), after: 7) {
      killed = f.current { $0.target }
      if let k = killed { f.kill(k) }
    }
    #expect(killed != nil)
    #expect(got == want)
  }

  @Test func aFrozenWorkerIsCaughtByTheHeartbeatAndRejoinsWhenItThaws() async throws {
    var o = ClusterOptions()
    o.heartbeatTimeout = 0.75
    let f = try FaultCluster([.prefill, .decode, .decode], stepDelayMs: [0, 5, 5], options: o)
    let p = randomPrompt(6, length: 30), s = sampled(seed: 16, maxTokens: 60)
    let want = try await reference(p, s)
    var frozen: Int?
    let start = Date()
    let got = try await collect(f.cluster.generate(prompt: p, sampling: s), after: 10) {
      frozen = f.current { $0.decode }
      if let k = frozen { f.process(k).signal(SIGSTOP) }  // alive, but silent
    }
    let elapsed = Date().timeIntervalSince(start)
    #expect(got == want)
    #expect(elapsed < 5)  // caught after ~0.75 s, not hung
    let k = try #require(frozen)
    f.process(k).signal(SIGCONT)
    #expect(f.waitUntil { f.cluster.stats().workers[k].alive })
    #expect(f.cluster.stats().workersRejoined == 1)
    let p2 = randomPrompt(7, length: 30), s2 = sampled(seed: 17, maxTokens: 30)
    #expect(try await collect(f.cluster.generate(prompt: p2, sampling: s2)) == (try await reference(p2, s2)))
  }

  @Test func aStalledWorkerIsCaughtByTheStepWatchdog() async throws {
    // Worker 2 answers heartbeats but takes 20 s before every step. It has the most memory,
    // so the router sends the request there.
    var o = ClusterOptions()
    o.stallTimeout = 1
    let f = try FaultCluster([.prefill, .decode, .decode], stepDelayMs: [0, 0, 20_000], blocks: [128, 128, 256], options: o)
    let p = randomPrompt(8, length: 20), s = sampled(seed: 18, maxTokens: 10)
    let want = try await reference(p, s)
    let start = Date()
    let stream = f.cluster.generate(prompt: p, sampling: s)
    #expect(f.current { $0.decode } == 2)
    let got = try await collect(stream)
    #expect(got == want)
    #expect(Date().timeIntervalSince(start) < 10)
    #expect(f.cluster.stats().workersLost == 1)
  }

  @Test func whenEveryWorkerIsDownRequestsWaitForOneToReturnOrFailCleanly() async throws {
    var o = ClusterOptions()
    o.placementTimeout = 1.5
    let f = try FaultCluster([.both], options: o)
    f.kill(0)
    #expect(f.waitUntil { f.cluster.stats().workersLost == 1 })
    let p = randomPrompt(9, length: 10), s = sampled(seed: 19, maxTokens: 5)
    // Nobody comes back: after the timeout, a clean error rather than a hang.
    let start = Date()
    await #expect(throws: ClusterError.self) { _ = try await collect(f.cluster.generate(prompt: p, sampling: s)) }
    #expect(Date().timeIntervalSince(start) > 1.4)
    // A request that arrives while the worker is down waits, and is served once it is back.
    let stream = f.cluster.generate(prompt: p, sampling: s)
    #expect(f.cluster.stats().waiting == 1)
    try f.restart(0)
    #expect(try await collect(stream) == (try await reference(p, s)))
    #expect(f.cluster.stats().waiting == 0)
  }

  // Many concurrent requests while workers are killed and restarted at seeded random
  // moments: every output must equal the fault-free run.
  @Test func chaosManyRequestsThroughRepeatedKillsAndRestarts() async throws {
    var o = ClusterOptions()
    o.maxPlacements = 50
    o.heartbeatTimeout = 1
    let roles: [WorkerRoleName] = [.prefill, .prefill, .decode, .decode]
    let f = try FaultCluster(roles, stepDelayMs: [2, 2, 2, 2], options: o)
    let n = 24
    let work = (0..<n).map { i in (randomPrompt(100 + UInt64(i), length: 10 + i * 2), sampled(seed: 200 + UInt64(i), maxTokens: 40 + i)) }
    var want: [[Int]] = []
    for (p, s) in work { want.append(try await reference(p, s)) }

    let done = Atomic<Bool>(false)
    let kills = Atomic<Int>(0)
    let chaos = Thread {
      var x: UInt64 = 0xC0FFEE
      func next(_ n: Int) -> Int {
        x = x &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
        return Int((x >> 33) % UInt64(n))
      }
      while !done.load(ordering: .relaxed) {
        usleep(UInt32(60_000 + next(80_000)))
        let i = next(roles.count)
        let alive = f.cluster.stats().workers
        // Keep at least one worker of each role up.
        guard alive.enumerated().contains(where: { $0.offset != i && $0.element.role == roles[i].rawValue && $0.element.alive }),
          alive[i].alive
        else { continue }
        f.kill(i)
        kills.add(1, ordering: .relaxed)
        usleep(100_000)
        try? f.restart(i)
      }
    }
    chaos.start()
    let got = try await withThrowingTaskGroup(of: (Int, [Int]).self) { group in
      for (i, (p, s)) in work.enumerated() {
        group.addTask {
          usleep(UInt32(i * 20_000))  // arrivals spread over half a second
          return (i, try await collect(f.cluster.generate(prompt: p, sampling: s)))
        }
      }
      var out = [[Int]](repeating: [], count: n)
      for try await (i, t) in group { out[i] = t }
      return out
    }
    done.store(true, ordering: .relaxed)
    while chaos.isExecuting { usleep(10_000) }
    #expect(got == want)
    let st = f.cluster.stats()
    print("chaos: \(n) requests, \(kills.load(ordering: .relaxed)) worker kills, \(st.recovered) recoveries, \(st.workersRejoined) rejoins; every output identical to the fault-free run")
    #expect(kills.load(ordering: .relaxed) >= 2)
  }
}

// Identical prompts arriving together share prefix-cache blocks on both prefill and decode
// workers. At the token level (no HTTP), every one must still equal the reference.
@Test func identicalConcurrentPromptsAtTheTokenLevel() async throws {
  let f = try FaultCluster([.prefill, .prefill, .decode, .decode])
  let ref = try JSONDecoder().decode(Reference.self, from: Data(contentsOf: chatTiny.appendingPathComponent("relay-reference.json")))
  var s = Sampling()
  s.temperature = 0
  s.maxTokens = ref.generated.count
  s.ignoreEOS = true
  let sampling = s
  for round in 0..<10 {
    let got = try await withThrowingTaskGroup(of: [Int].self) { group in
      for _ in 0..<16 { group.addTask { try await collect(f.cluster.generate(prompt: ref.prompt, sampling: sampling)) } }
      var all: [[Int]] = []
      for try await r in group { all.append(r) }
      return all
    }
    for g in got where g != ref.generated { print("TOKEN MISMATCH round \(round): got \(g) want \(ref.generated)") }
    #expect(got.allSatisfy { $0 == ref.generated })
  }
}
