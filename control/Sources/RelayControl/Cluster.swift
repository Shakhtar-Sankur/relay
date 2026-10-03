// The control plane's view of a cluster of relay-worker processes, as a GenerationBackend:
// the OpenAI API runs unchanged on top of it.
//
// Two shapes:
//   colocated      workers with role "both": each request runs entirely on one worker
//   disaggregated  prefill workers and decode workers: the router picks one of each, the
//                  prefill worker produces the first token and streams the KV cache to the
//                  decode worker, which produces the rest
// Each worker has one control connection (WorkerConnection) with a reader thread. Tokens for
// a request may arrive from two workers; they are delivered to the request's stream in
// index order.
#if canImport(Glibc)
  import Glibc
#endif
import Foundation
import Synchronization

public enum WorkerRoleName: String, Sendable {
  case both, prefill, decode
}

public struct WorkerAddress: Sendable, Equatable {
  public var role: WorkerRoleName
  public var host: String
  public var port: Int
  public init(role: WorkerRoleName, host: String, port: Int) {
    self.role = role
    self.host = host
    self.port = port
  }
  // "prefill=127.0.0.1:7001"
  public init?(_ spec: String) {
    let parts = spec.split(separator: "=", maxSplits: 1)
    guard parts.count == 2, let role = WorkerRoleName(rawValue: String(parts[0])),
      let colon = parts[1].lastIndex(of: ":"), let port = Int(parts[1][parts[1].index(after: colon)...])
    else { return nil }
    self.init(role: role, host: String(parts[1][..<colon]), port: port)
  }
}

enum Ctl: UInt16 {
  case hello = 1, token = 2, error = 3, load = 4, submit = 5, resume = 6, cancel = 7, peer = 8
}

struct FrameHeader {
  static let size = 32
  static let magic: UInt32 = 0x4359_4C52  // "RLYC"
  var type: UInt16
  var flags: UInt16
  var request: UInt64
  var a: UInt32
  var b: UInt32
  var payload: UInt64

  func encode() -> [UInt8] {
    var out = [UInt8]()
    out.reserveCapacity(FrameHeader.size)
    func put<T: FixedWidthInteger>(_ v: T) { withUnsafeBytes(of: v.littleEndian) { out += $0 } }
    put(FrameHeader.magic); put(type); put(flags); put(request); put(a); put(b); put(payload)
    return out
  }

  static func decode(_ b: [UInt8]) throws -> FrameHeader {
    func get<T: FixedWidthInteger>(_ at: Int, _: T.Type) -> T {
      b[at..<(at + MemoryLayout<T>.size)].withUnsafeBytes { T(littleEndian: $0.loadUnaligned(as: T.self)) }
    }
    guard get(0, UInt32.self) == magic else { throw ClusterError.protocolError("bad frame magic") }
    return FrameHeader(type: get(4, UInt16.self), flags: get(6, UInt16.self), request: get(8, UInt64.self),
                       a: get(16, UInt32.self), b: get(20, UInt32.self), payload: get(24, UInt64.self))
  }
}

public enum ClusterError: Error, CustomStringConvertible {
  case connect(String)
  case protocolError(String)
  case worker(String)
  case noWorker(String)
  public var description: String {
    switch self {
    case .connect(let s): return "cannot connect: \(s)"
    case .protocolError(let s): return "protocol error: \(s)"
    case .worker(let s): return "worker error: \(s)"
    case .noWorker(let s): return "no worker available: \(s)"
    }
  }
}

struct WorkerHello {
  var role: UInt32
  var numBlocks: Int32, blockSize: Int32, vocab: Int32, layers: Int32, kvPort: Int32
}

struct WorkerLoad {
  var freeBlocks: Int32, totalBlocks: Int32
  var promptTokens: UInt64, prefixHitTokens: UInt64, forwardTokens: UInt64
}

// One worker's control connection.
final class WorkerConnection: @unchecked Sendable {
  let address: WorkerAddress
  let fd: Int32
  let hello: WorkerHello
  private let writeLock = NSLock()
  let load = Mutex<WorkerLoad>(WorkerLoad(freeBlocks: 0, totalBlocks: 0, promptTokens: 0, prefixHitTokens: 0, forwardTokens: 0))
  let alive = Atomic<Bool>(true)
  private var reader: Thread?

  init(_ address: WorkerAddress, timeout: Double = 10) throws {
    self.address = address
    let deadline = Date().addingTimeInterval(timeout)
    var fd: Int32 = -1
    while true {
      fd = socket(AF_INET, Int32(SOCK_STREAM.rawValue), 0)
      var addr = sockaddr_in()
      addr.sin_family = sa_family_t(AF_INET)
      addr.sin_port = in_port_t(UInt16(address.port).bigEndian)
      inet_pton(AF_INET, address.host, &addr.sin_addr)
      let r = withUnsafePointer(to: &addr) {
        $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { connect(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size)) }
      }
      if r == 0 { break }
      close(fd)
      if Date() > deadline { throw ClusterError.connect("\(address.host):\(address.port)") }
      usleep(20_000)
    }
    var one: Int32 = 1
    setsockopt(fd, Int32(IPPROTO_TCP), TCP_NODELAY, &one, socklen_t(MemoryLayout<Int32>.size))
    self.fd = fd
    let h = try FrameHeader.decode(try WorkerConnection.recvExact(fd, FrameHeader.size))
    guard h.type == Ctl.hello.rawValue, h.payload == 24 else { throw ClusterError.protocolError("expected Hello") }
    let p = try WorkerConnection.recvExact(fd, 24)
    func i32(_ at: Int) -> Int32 { p[at..<(at + 4)].withUnsafeBytes { Int32(littleEndian: $0.loadUnaligned(as: Int32.self)) } }
    hello = WorkerHello(role: UInt32(bitPattern: i32(0)), numBlocks: i32(4), blockSize: i32(8), vocab: i32(12),
                        layers: i32(16), kvPort: i32(20))
    load.withLock { $0.totalBlocks = hello.numBlocks; $0.freeBlocks = hello.numBlocks }
  }

  static func recvExact(_ fd: Int32, _ n: Int) throws -> [UInt8] {
    var out = [UInt8](repeating: 0, count: n)
    var got = 0
    while got < n {
      let r = out.withUnsafeMutableBytes { recv(fd, $0.baseAddress! + got, n - got, 0) }
      if r <= 0 {
        if r < 0 && errno == EINTR { continue }
        throw ClusterError.connect("connection closed")
      }
      got += r
    }
    return out
  }

  func start(onFrame: @escaping @Sendable (FrameHeader, [UInt8]) -> Void, onClose: @escaping @Sendable () -> Void) {
    let t = Thread { [self] in
      do {
        while true {
          let h = try FrameHeader.decode(try WorkerConnection.recvExact(fd, FrameHeader.size))
          let p = h.payload > 0 ? try WorkerConnection.recvExact(fd, Int(h.payload)) : []
          if h.type == Ctl.load.rawValue, p.count >= 48 {
            func i32(_ at: Int) -> Int32 { p[at..<(at + 4)].withUnsafeBytes { Int32(littleEndian: $0.loadUnaligned(as: Int32.self)) } }
            func u64(_ at: Int) -> UInt64 { p[at..<(at + 8)].withUnsafeBytes { UInt64(littleEndian: $0.loadUnaligned(as: UInt64.self)) } }
            load.withLock { $0 = WorkerLoad(freeBlocks: i32(0), totalBlocks: i32(4), promptTokens: u64(16), prefixHitTokens: u64(24), forwardTokens: u64(32)) }
          } else {
            onFrame(h, p)
          }
        }
      } catch {}
      alive.store(false, ordering: .releasing)
      onClose()
    }
    t.name = "relay-worker-\(address.port)"
    reader = t
    t.start()
  }

  func send(_ type: Ctl, request: UInt64 = 0, a: UInt32 = 0, b: UInt32 = 0, payload: [UInt8] = []) throws {
    let bytes = FrameHeader(type: type.rawValue, flags: 0, request: request, a: a, b: b, payload: UInt64(payload.count)).encode() + payload
    writeLock.lock()
    defer { writeLock.unlock() }
    var off = 0
    while off < bytes.count {
      let n = bytes.withUnsafeBytes { Glibc.send(fd, $0.baseAddress! + off, bytes.count - off, Int32(MSG_NOSIGNAL)) }
      if n <= 0 {
        if n < 0 && errno == EINTR { continue }
        throw ClusterError.connect("send to \(address.host):\(address.port) failed")
      }
      off += n
    }
  }

  func shutdownConnection() { Glibc.shutdown(fd, Int32(SHUT_RDWR)) }

  deinit { close(fd) }
}

// Encodes a request the way the workers' BeginInfo expects.
func beginPayload(prompt: [Int], sampling: Sampling, generated: [Int] = []) -> [UInt8] {
  var out = [UInt8]()
  func put<T: FixedWidthInteger>(_ v: T) { withUnsafeBytes(of: v.littleEndian) { out += $0 } }
  put(UInt32(prompt.count)); put(UInt32(0)); put(Int32(sampling.maxTokens))
  put(sampling.temperature.bitPattern); put(sampling.topP.bitPattern); put(Int32(sampling.topK))
  put(sampling.seed); put(UInt32(sampling.ignoreEOS ? 1 : 0)); put(UInt32(0))
  for t in prompt + generated { put(Int32(t)) }
  return out
}

public struct ClusterStats: Sendable {
  public var workers: [(address: String, role: String, freeBlocks: Int, totalBlocks: Int, promptTokens: UInt64,
                        prefixHitTokens: UInt64, forwardTokens: UInt64, alive: Bool)]
  public var inFlight: Int
}

public final class Cluster: GenerationBackend, @unchecked Sendable {
  final class Request: @unchecked Sendable {
    let id: UInt64
    let prompt: [Int]
    let sampling: Sampling
    let continuation: AsyncThrowingStream<GeneratedToken, Error>.Continuation
    var prefill: Int?  // worker indices
    var decode: Int?
    var nextIndex = 0
    var pending: [Int: GeneratedToken] = [:]  // tokens that arrived ahead of their turn
    var generated: [Int] = []
    var finished = false
    init(id: UInt64, prompt: [Int], sampling: Sampling, continuation: AsyncThrowingStream<GeneratedToken, Error>.Continuation) {
      self.id = id
      self.prompt = prompt
      self.sampling = sampling
      self.continuation = continuation
    }
  }

  struct State {
    var requests: [UInt64: Request] = [:]
    var queuedPrompt: [Int: Int] = [:]   // worker -> prompt tokens routed and not prefilled yet
    var assignedBlocks: [Int: Int] = [:]  // worker -> blocks needed by requests routed there
  }

  let workers: [WorkerConnection]
  let router: Router
  public let maxInFlight: Int
  let state = Mutex(State())
  private let nextID = Atomic<UInt64>(1)

  public var disaggregated: Bool { workers.contains { $0.address.role == .prefill } }

  public init(addresses: [WorkerAddress], maxInFlight: Int = 256, slack: Int = 2048) throws {
    let roles = Set(addresses.map(\.role))
    guard roles == [.both] || roles == [.prefill, .decode] else {
      throw ClusterError.noWorker("give only role=both workers, or both prefill and decode workers")
    }
    workers = try addresses.map { try WorkerConnection($0) }
    guard let first = workers.first else { throw ClusterError.noWorker("no workers") }
    for w in workers where w.hello.vocab != first.hello.vocab || w.hello.layers != first.hello.layers
      || w.hello.blockSize != first.hello.blockSize
    {
      throw ClusterError.protocolError("worker \(w.address.port) runs a different model or block size")
    }
    router = Router(blockSize: Int(first.hello.blockSize), slack: slack)
    self.maxInFlight = maxInFlight
    for (i, w) in workers.enumerated() {
      w.start(onFrame: { [weak self] h, p in self?.frame(from: i, h, p) },
              onClose: { [weak self] in self?.workerLost(i) })
    }
    // Every prefill worker connects to every decode worker's KV port; peer index = worker index.
    for p in workers where p.address.role == .prefill {
      for (i, d) in workers.enumerated() where d.address.role == .decode {
        try p.send(.peer, a: UInt32(i), payload: Array("\(d.address.host):\(d.hello.kvPort)".utf8))
      }
    }
  }

  public func shutdown() { for w in workers { w.shutdownConnection() } }

  public func stats() -> ClusterStats {
    let inFlight = state.withLock { $0.requests.count }
    return ClusterStats(workers: workers.map { w in
      let l = w.load.withLock { $0 }
      return ("\(w.address.host):\(w.address.port)", w.address.role.rawValue, Int(l.freeBlocks), Int(l.totalBlocks),
              l.promptTokens, l.prefixHitTokens, l.forwardTokens, w.alive.load(ordering: .acquiring))
    }, inFlight: inFlight)
  }

  func snapshots(role: WorkerRoleName, _ s: State) -> [WorkerSnapshot] {
    workers.enumerated().filter { $0.element.address.role == role }.map { i, w in
      let l = w.load.withLock { $0 }
      return WorkerSnapshot(index: i, healthy: w.alive.load(ordering: .acquiring), freeBlocks: Int(l.freeBlocks),
                            totalBlocks: Int(l.totalBlocks), queuedPromptTokens: s.queuedPrompt[i, default: 0],
                            assignedBlocks: s.assignedBlocks[i, default: 0])
    }
  }

  public func generate(prompt: [Int], sampling: Sampling) -> AsyncThrowingStream<GeneratedToken, Error> {
    let id = nextID.add(1, ordering: .relaxed).oldValue
    return AsyncThrowingStream { continuation in
      let r = Request(id: id, prompt: prompt, sampling: sampling, continuation: continuation)
      let blocks = (prompt.count + sampling.maxTokens + router.blockSize - 1) / router.blockSize
      let choice: Result<(Int, Int?), Error> = state.withLock { s in
        if s.requests.count >= maxInFlight { return .failure(AdmissionError(inFlight: s.requests.count)) }
        let p: Int?, d: Int?
        if disaggregated {
          p = router.choosePrefill(prompt: prompt, workers: snapshots(role: .prefill, s))
          d = router.chooseDecode(workers: snapshots(role: .decode, s))
        } else {
          p = router.choosePrefill(prompt: prompt, workers: snapshots(role: .both, s))
          d = nil
        }
        guard let p, disaggregated == false || d != nil else { return .failure(ClusterError.noWorker("all workers down")) }
        r.prefill = p
        r.decode = d
        s.requests[id] = r
        s.queuedPrompt[p, default: 0] += prompt.count
        s.assignedBlocks[d ?? p, default: 0] += blocks
        return .success((p, d))
      }
      switch choice {
      case .failure(let e):
        continuation.finish(throwing: e)
        return
      case .success(let (p, d)):
        continuation.onTermination = { [weak self] reason in
          guard let self, case .cancelled = reason else { return }
          self.cancel(id)
        }
        do {
          try workers[p].send(.submit, request: id, a: d.map(UInt32.init) ?? 0xFFFF_FFFF,
                              payload: beginPayload(prompt: prompt, sampling: sampling))
        } catch {
          fail(id, error)
        }
      }
    }
  }

  private func release(_ r: Request, _ s: inout State) {
    if let p = r.prefill, r.nextIndex == 0 { s.queuedPrompt[p, default: 0] -= r.prompt.count }
    let blocks = (r.prompt.count + r.sampling.maxTokens + router.blockSize - 1) / router.blockSize
    if let w = r.decode ?? r.prefill { s.assignedBlocks[w, default: 0] -= blocks }
    s.requests[r.id] = nil
  }

  func cancel(_ id: UInt64) {
    let r = state.withLock { s -> Request? in
      guard let r = s.requests[id], !r.finished else { return nil }
      r.finished = true
      release(r, &s)
      return r
    }
    guard let r else { return }
    for w in Set([r.prefill, r.decode].compactMap { $0 }) { try? workers[w].send(.cancel, request: id) }
  }

  func fail(_ id: UInt64, _ error: Error) {
    let r = state.withLock { s -> Request? in
      guard let r = s.requests[id], !r.finished else { return nil }
      r.finished = true
      release(r, &s)
      return r
    }
    r?.continuation.finish(throwing: error)
  }

  private func frame(from worker: Int, _ h: FrameHeader, _ p: [UInt8]) {
    switch Ctl(rawValue: h.type) {
    case .token:
      let finish: FinishReason? = h.flags == 0 ? nil : (h.flags == 1 ? .length : .stop)
      deliver(h.request, GeneratedToken(token: Int(h.a), index: Int(h.b), finish: finish))
    case .error:
      let message = String(decoding: p, as: UTF8.self)
      if h.request != 0 { fail(h.request, ClusterError.worker(message)) }
    default:
      break
    }
  }

  // Delivers tokens to the request's stream in index order (the first may come from the
  // prefill worker after the decode worker's second, in principle).
  private func deliver(_ id: UInt64, _ t: GeneratedToken) {
    let ready: (Request, [GeneratedToken])? = state.withLock { s in
      guard let r = s.requests[id], !r.finished else { return nil }
      if t.index == 0, let p = r.prefill { s.queuedPrompt[p, default: 0] -= r.prompt.count }
      r.pending[t.index] = t
      var out: [GeneratedToken] = []
      while let next = r.pending.removeValue(forKey: r.nextIndex) {
        out.append(next)
        r.generated.append(next.token)
        r.nextIndex += 1
        if next.finish != nil {
          r.finished = true
          release(r, &s)
          break
        }
      }
      return (r, out)
    }
    guard let (r, tokens) = ready else { return }
    for t in tokens { r.continuation.yield(t) }
    if tokens.last?.finish != nil { r.continuation.finish() }
  }

  private func workerLost(_ worker: Int) {
    let affected = state.withLock { s in s.requests.values.filter { $0.prefill == worker || $0.decode == worker } }
    for r in affected { fail(r.id, ClusterError.worker("worker \(workers[worker].address.port) lost")) }
  }
}
