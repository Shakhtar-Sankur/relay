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
//
// Fault tolerance. A worker is lost when its connection closes (the process died), when it
// sends nothing for `heartbeatTimeout` (it hangs, or the network does), or when it has
// requests but its step counter stops for `stallTimeout`. Each request it held is placed
// again, depending on how far it got:
//   no token yet         a fresh prefill worker (and decode worker), or, with no prefill
//                        worker left, a decode worker that does the whole request
//   some tokens          a decode (or colocated) worker resumes it: it recomputes the
//                        prompt and the tokens already generated, then continues
// Sampling is a function of (seed, token index), so the client sees one uninterrupted
// stream, with the tokens it would have seen anyway. When no worker of a needed role is up
// (all of them lost at once, or still restarting), a request waits up to
// `placementTimeout` for one to come back rather than failing at once. Every placement gets a fresh wire id:
// tokens still arriving from an abandoned placement name an id that no longer exists and
// are dropped. Lost workers are reconnected in the background; a worker that comes back
// (restarted, or unfrozen) rejoins the cluster, and the prefill workers are pointed at
// decode workers' new KV ports.
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
  var promptTokens: UInt64, prefixHitTokens: UInt64, forwardTokens: UInt64, steps: UInt64
}

func monotonicNanos() -> UInt64 { DispatchTime.now().uptimeNanoseconds }

// RELAY_DEBUG=1 logs the cluster's view of workers coming and going.
let debugEnabled = ProcessInfo.processInfo.environment["RELAY_DEBUG"] != nil
func debugLog(_ message: @autoclosure () -> String) {
  if debugEnabled { FileHandle.standardError.write(Data("[cluster] \(message())\n".utf8)) }
}

// Seconds from `then` to `now`, or 0 if `then` is later: another thread may have stamped
// it after `now` was read, and an unsigned difference would wrap to centuries.
func secondsSince(_ then: UInt64, now: UInt64) -> Double { now > then ? Double(now - then) / 1e9 : 0 }

// One worker's control connection.
final class WorkerConnection: @unchecked Sendable {
  let address: WorkerAddress
  let fd: Int32
  let hello: WorkerHello
  private let writeLock = NSLock()
  let load = Mutex<WorkerLoad>(WorkerLoad(freeBlocks: 0, totalBlocks: 0, promptTokens: 0, prefixHitTokens: 0, forwardTokens: 0, steps: 0))
  let alive = Atomic<Bool>(true)
  let lastHeard = Atomic<UInt64>(monotonicNanos())  // any frame counts: Load arrives every 20 ms
  private var reader: Thread?

  // timeout: how long to keep trying to connect; helloTimeout: how long the worker may take
  // to greet us once connected (a frozen process accepts connections but never answers).
  init(_ address: WorkerAddress, timeout: Double = 10, helloTimeout: Double = 10) throws {
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
    WorkerConnection.setReceiveTimeout(fd, helloTimeout)
    do {
      let h = try FrameHeader.decode(try WorkerConnection.recvExact(fd, FrameHeader.size))
      guard h.type == Ctl.hello.rawValue, h.payload == 24 else { throw ClusterError.protocolError("expected Hello") }
      let p = try WorkerConnection.recvExact(fd, 24)
      func i32(_ at: Int) -> Int32 { p[at..<(at + 4)].withUnsafeBytes { Int32(littleEndian: $0.loadUnaligned(as: Int32.self)) } }
      hello = WorkerHello(role: UInt32(bitPattern: i32(0)), numBlocks: i32(4), blockSize: i32(8), vocab: i32(12),
                          layers: i32(16), kvPort: i32(20))
    } catch {
      close(fd)
      throw error
    }
    WorkerConnection.setReceiveTimeout(fd, 0)  // from here on the heartbeat watchdog decides
    self.fd = fd
    load.withLock { $0.totalBlocks = hello.numBlocks; $0.freeBlocks = hello.numBlocks }
  }

  static func setReceiveTimeout(_ fd: Int32, _ seconds: Double) {
    var tv = timeval(tv_sec: Int(seconds), tv_usec: Int((seconds - Double(Int(seconds))) * 1e6))
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, socklen_t(MemoryLayout<timeval>.size))
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
          lastHeard.store(monotonicNanos(), ordering: .relaxed)
          if h.type == Ctl.load.rawValue, p.count >= 48 {
            func i32(_ at: Int) -> Int32 { p[at..<(at + 4)].withUnsafeBytes { Int32(littleEndian: $0.loadUnaligned(as: Int32.self)) } }
            func u64(_ at: Int) -> UInt64 { p[at..<(at + 8)].withUnsafeBytes { UInt64(littleEndian: $0.loadUnaligned(as: UInt64.self)) } }
            load.withLock {
              $0 = WorkerLoad(freeBlocks: i32(0), totalBlocks: i32(4), promptTokens: u64(16), prefixHitTokens: u64(24),
                              forwardTokens: u64(32), steps: u64(40))
            }
          } else {
            onFrame(h, p)
          }
        }
      } catch {
        debugLog("reader for \(address.port): \(error)")
      }
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

public struct ClusterOptions: Sendable {
  public var maxInFlight = 256
  public var slack = 2048
  public var heartbeatTimeout = 2.0     // seconds without any frame from a worker: lost
  public var stallTimeout = 30.0        // seconds a worker holds requests without stepping: lost
  public var reconnectInterval = 0.5    // how often a lost worker is tried again
  public var maxPlacements = 16         // a request is placed at most this often (first + retries)
  public var placementTimeout = 30.0    // seconds a request may wait for a worker of a needed role
  public init() {}
}

public struct ClusterStats: Sendable {
  public var workers: [(address: String, role: String, freeBlocks: Int, totalBlocks: Int, promptTokens: UInt64,
                        prefixHitTokens: UInt64, forwardTokens: UInt64, alive: Bool)]
  public var inFlight: Int
  public var waiting: Int       // requests waiting for a worker of a needed role to come back
  public var recovered: Int     // requests placed again after losing a worker or a transfer
  public var workersLost: Int   // times a worker was declared lost
  public var workersRejoined: Int
}

// A worker's place in the cluster: fixed address and role, a connection that comes and goes.
final class WorkerSlot: @unchecked Sendable {
  let index: Int
  let address: WorkerAddress
  private let lock = NSLock()
  private var _conn: WorkerConnection?
  // Watchdog thread only.
  var watched: ObjectIdentifier?  // the connection the two below belong to
  var lastSteps: UInt64 = 0
  var lastProgress: UInt64 = 0
  // Reconnect thread only.
  var nextAttempt: UInt64 = 0
  let connecting = Atomic<Bool>(false)  // an attempt to reach it is under way

  init(index: Int, address: WorkerAddress, conn: WorkerConnection) {
    self.index = index
    self.address = address
    _conn = conn
  }
  var conn: WorkerConnection? { lock.withLock { _conn } }
  var alive: Bool { conn?.alive.load(ordering: .acquiring) ?? false }
  // Replaces the connection only if it is still `expected` (a late close of an old
  // connection must not unseat a new one).
  @discardableResult
  func replace(_ expected: WorkerConnection?, with new: WorkerConnection?) -> Bool {
    lock.withLock {
      guard _conn === expected else { return false }
      _conn = new
      return true
    }
  }
}

public final class Cluster: GenerationBackend, @unchecked Sendable {
  final class Request: @unchecked Sendable {
    let id: UInt64
    let prompt: [Int]
    let sampling: Sampling
    let continuation: AsyncThrowingStream<GeneratedToken, Error>.Continuation
    var wire: UInt64 = 0         // the id the workers know this placement by
    var prefill: Int?            // worker indices of the current placement
    var decode: Int?
    var target: Int?             // the worker whose cache the placement's blocks are counted on
    var queued = false           // its prompt is counted in queuedPrompt[prefill]
    var placements = 0
    var nextIndex = 0
    var pending: [Int: GeneratedToken] = [:]  // tokens that arrived ahead of their turn
    var generated: [Int] = []
    var finished = false
    var waitingSince: UInt64?    // set while no worker can take it
    init(id: UInt64, prompt: [Int], sampling: Sampling, continuation: AsyncThrowingStream<GeneratedToken, Error>.Continuation) {
      self.id = id
      self.prompt = prompt
      self.sampling = sampling
      self.continuation = continuation
    }
  }

  struct State {
    var requests: [UInt64: Request] = [:]
    var wire: [UInt64: UInt64] = [:]     // wire id -> request id, for current placements only
    var waiting: Set<UInt64> = []        // requests with no placement, waiting for a worker
    var queuedPrompt: [Int: Int] = [:]   // worker -> prompt tokens routed and not prefilled yet
    var assignedBlocks: [Int: Int] = [:]  // worker -> blocks needed by requests routed there
    var recovered = 0
    var workersLost = 0
    var workersRejoined = 0
  }

  // What to send to place a request.
  enum Placement {
    case submit(worker: Int, peer: Int?, wire: UInt64)
    case resume(worker: Int, wire: UInt64, generated: [Int])
  }

  let slots: [WorkerSlot]
  let router: Router
  public let options: ClusterOptions
  public var maxInFlight: Int { options.maxInFlight }
  let state = Mutex(State())
  private let nextID = Atomic<UInt64>(1)
  private let stopping = Atomic<Bool>(false)
  private var watchdog: Thread?
  private var reconnector: Thread?

  public let disaggregated: Bool
  let model: WorkerHello  // what every worker must run (the first one's, at startup)

  public convenience init(addresses: [WorkerAddress], maxInFlight: Int = 256, slack: Int = 2048) throws {
    var o = ClusterOptions()
    o.maxInFlight = maxInFlight
    o.slack = slack
    try self.init(addresses: addresses, options: o)
  }

  public init(addresses: [WorkerAddress], options: ClusterOptions) throws {
    let roles = Set(addresses.map(\.role))
    guard roles == [.both] || roles == [.prefill, .decode] else {
      throw ClusterError.noWorker("give only role=both workers, or both prefill and decode workers")
    }
    disaggregated = roles.contains(.prefill)
    let conns = try addresses.map { try WorkerConnection($0) }
    guard let first = conns.first else { throw ClusterError.noWorker("no workers") }
    for w in conns where Cluster.role(w.hello) != w.address.role {
      throw ClusterError.protocolError("worker \(w.address.port) is not a \(w.address.role.rawValue) worker")
    }
    for w in conns where !Cluster.sameModel(w.hello, first.hello) {
      throw ClusterError.protocolError("worker \(w.address.port) runs a different model or block size")
    }
    model = first.hello
    slots = conns.enumerated().map { WorkerSlot(index: $0, address: $1.address, conn: $1) }
    router = Router(blockSize: Int(first.hello.blockSize), slack: options.slack)
    self.options = options
    for slot in slots { attach(slot, slot.conn!) }
    // Every prefill worker connects to every decode worker's KV port; peer index = worker index.
    for slot in slots where slot.address.role == .decode { announce(decode: slot.index) }

    let w = Thread { [weak self] in
      while let self, !self.stopping.load(ordering: .relaxed) {
        self.watch()
        self.placeWaiting()
        usleep(50_000)
      }
    }
    w.name = "relay-cluster-watchdog"
    watchdog = w
    w.start()
    let r = Thread { [weak self] in
      while let self, !self.stopping.load(ordering: .relaxed) {
        self.reconnect()
        usleep(50_000)
      }
    }
    r.name = "relay-cluster-reconnect"
    reconnector = r
    r.start()
  }

  static func role(_ h: WorkerHello) -> WorkerRoleName? {
    switch h.role {
    case 0: return .both
    case 1: return .prefill
    case 2: return .decode
    default: return nil
    }
  }

  static func sameModel(_ a: WorkerHello, _ b: WorkerHello) -> Bool {
    a.vocab == b.vocab && a.layers == b.layers && a.blockSize == b.blockSize
  }

  public func shutdown() {
    stopping.store(true, ordering: .relaxed)
    for slot in slots { slot.conn?.shutdownConnection() }
  }

  public func stats() -> ClusterStats {
    let s = state.withLock { ($0.requests.count, $0.recovered, $0.workersLost, $0.workersRejoined, $0.waiting.count) }
    return ClusterStats(workers: slots.map { slot in
      let l = slot.conn?.load.withLock { $0 } ?? WorkerLoad(freeBlocks: 0, totalBlocks: 0, promptTokens: 0, prefixHitTokens: 0, forwardTokens: 0, steps: 0)
      return ("\(slot.address.host):\(slot.address.port)", slot.address.role.rawValue, Int(l.freeBlocks), Int(l.totalBlocks),
              l.promptTokens, l.prefixHitTokens, l.forwardTokens, slot.alive)
    }, inFlight: s.0, waiting: s.4, recovered: s.1, workersLost: s.2, workersRejoined: s.3)
  }

  // ---- connections ------------------------------------------------------------

  private func attach(_ slot: WorkerSlot, _ conn: WorkerConnection) {
    conn.start(onFrame: { [weak self] h, p in self?.frame(from: slot.index, h, p) },
               onClose: { [weak self] in self?.lost(slot, conn) })
  }

  // Sends to worker i; a failed send means the worker is gone.
  @discardableResult
  private func send(_ i: Int, _ type: Ctl, request: UInt64 = 0, a: UInt32 = 0, b: UInt32 = 0, payload: [UInt8] = []) -> Bool {
    guard let conn = slots[i].conn, conn.alive.load(ordering: .acquiring) else { return false }
    do {
      try conn.send(type, request: request, a: a, b: b, payload: payload)
      return true
    } catch {
      debugLog("send \(type) to worker \(i) failed")
      conn.shutdownConnection()  // the reader sees it and declares the worker lost
      return false
    }
  }

  // Points every live prefill worker at decode worker d's KV port.
  private func announce(decode d: Int) {
    guard let conn = slots[d].conn else { return }
    let addr = Array("\(slots[d].address.host):\(conn.hello.kvPort)".utf8)
    for p in slots where p.address.role == .prefill { send(p.index, .peer, a: UInt32(d), payload: addr) }
  }

  private func announcePeers(toPrefill p: Int) {
    for d in slots where d.address.role == .decode {
      guard let conn = d.conn, d.alive else { continue }
      send(p, .peer, a: UInt32(d.index), payload: Array("\(d.address.host):\(conn.hello.kvPort)".utf8))
    }
  }

  // Heartbeat and stall watchdog.
  private func watch() {
    let now = monotonicNanos()
    let busy: Set<Int> = state.withLock { s in
      var b = Set<Int>()
      for r in s.requests.values { for w in [r.prefill, r.decode, r.target].compactMap({ $0 }) { b.insert(w) } }
      return b
    }
    for slot in slots {
      guard let conn = slot.conn, conn.alive.load(ordering: .acquiring) else { continue }
      let silent = secondsSince(conn.lastHeard.load(ordering: .relaxed), now: now)
      if silent > options.heartbeatTimeout {
        debugLog("watchdog: worker \(slot.index) silent for \(silent)s")
        conn.shutdownConnection()
        continue
      }
      let steps = conn.load.withLock { $0.steps }
      if slot.watched != ObjectIdentifier(conn) || steps != slot.lastSteps || !busy.contains(slot.index) {
        slot.watched = ObjectIdentifier(conn)  // a new connection (a rejoined worker) starts afresh
        slot.lastSteps = steps
        slot.lastProgress = now
      } else if secondsSince(slot.lastProgress, now: now) > options.stallTimeout {
        debugLog("watchdog: worker \(slot.index) stalled at step \(steps)")
        conn.shutdownConnection()
      }
    }
  }

  // Brings lost workers back. Each attempt runs on its own thread: a frozen worker accepts
  // the connection but never greets, and must not hold up the others.
  private func reconnect() {
    let now = monotonicNanos()
    for slot in slots where slot.conn == nil && now >= slot.nextAttempt {
      guard slot.connecting.compareExchange(expected: false, desired: true, ordering: .acquiring).exchanged else { continue }
      slot.nextAttempt = now + UInt64(options.reconnectInterval * 1e9)
      Thread { [weak self] in
        defer { slot.connecting.store(false, ordering: .releasing) }
        guard let self, !self.stopping.load(ordering: .relaxed) else { return }
        guard let conn = try? WorkerConnection(slot.address, timeout: 0.2, helloTimeout: 2.0) else { return }
        // Whoever answers on this address must still run the same model in the same role.
        guard Cluster.sameModel(conn.hello, self.model), Cluster.role(conn.hello) == slot.address.role,
          slot.replace(nil, with: conn)
        else {
          conn.shutdownConnection()
          return
        }
        self.attach(slot, conn)
        self.state.withLock { $0.workersRejoined += 1 }
        debugLog("worker \(slot.index) rejoined")
        switch slot.address.role {
        case .decode: self.announce(decode: slot.index)
        case .prefill: self.announcePeers(toPrefill: slot.index)
        case .both: break
        }
      }.start()
    }
  }

  // ---- placement --------------------------------------------------------------

  func snapshots(role: WorkerRoleName, _ s: State, avoiding: Set<Int> = []) -> [WorkerSnapshot] {
    slots.filter { $0.address.role == role }.map { slot in
      let l = slot.conn?.load.withLock { $0 }
      return WorkerSnapshot(index: slot.index, healthy: slot.alive && !avoiding.contains(slot.index),
                            freeBlocks: Int(l?.freeBlocks ?? 0), totalBlocks: Int(l?.totalBlocks ?? 0),
                            queuedPromptTokens: s.queuedPrompt[slot.index, default: 0],
                            assignedBlocks: s.assignedBlocks[slot.index, default: 0])
    }
  }

  private func blocks(_ r: Request) -> Int { (r.prompt.count + r.sampling.maxTokens + router.blockSize - 1) / router.blockSize }

  // Chooses workers for the request from where it stands and records the placement.
  // `avoiding` are workers to pass over if any other will do.
  private func place(_ r: Request, _ s: inout State, avoiding: Set<Int>) -> Result<Placement, Error> {
    func pick(_ choose: (Set<Int>) -> Int?) -> Int? { choose(avoiding) ?? choose([]) }
    let wire = nextID.add(1, ordering: .relaxed).oldValue
    var placement: Placement
    if disaggregated {
      let d = pick { router.chooseDecode(workers: snapshots(role: .decode, s, avoiding: $0)) }
      let p = r.nextIndex == 0 ? pick { router.choosePrefill(prompt: r.prompt, workers: snapshots(role: .prefill, s, avoiding: $0)) } : nil
      guard let d else { return .failure(ClusterError.noWorker("no decode worker is up")) }
      if let p {
        placement = .submit(worker: p, peer: d, wire: wire)
        r.prefill = p
        r.queued = true
        s.queuedPrompt[p, default: 0] += r.prompt.count
      } else {
        // Tokens already generated (or no prefill worker left): the decode worker
        // recomputes the prompt and continues the request on its own.
        placement = .resume(worker: d, wire: wire, generated: r.generated)
        r.prefill = nil
      }
      r.decode = d
      r.target = d
    } else {
      guard let w = pick({ router.choosePrefill(prompt: r.prompt, workers: snapshots(role: .both, s, avoiding: $0)) }) else {
        return .failure(ClusterError.noWorker("all workers are down"))
      }
      placement = r.nextIndex == 0 ? .submit(worker: w, peer: nil, wire: wire) : .resume(worker: w, wire: wire, generated: r.generated)
      r.prefill = w
      r.decode = nil
      r.target = w
      if r.nextIndex == 0 {
        r.queued = true
        s.queuedPrompt[w, default: 0] += r.prompt.count
      }
    }
    s.assignedBlocks[r.target!, default: 0] += blocks(r)
    r.wire = wire
    r.placements += 1
    s.wire[wire] = r.id
    return .success(placement)
  }

  // Undoes a placement's bookkeeping.
  private func unplace(_ r: Request, _ s: inout State) {
    if r.queued, let p = r.prefill { s.queuedPrompt[p, default: 0] -= r.prompt.count }
    r.queued = false
    if let t = r.target { s.assignedBlocks[t, default: 0] -= blocks(r) }
    r.target = nil
    s.wire[r.wire] = nil
  }

  private func execute(_ r: Request, _ p: Placement) {
    switch p {
    case .submit(let w, let peer, let wire):
      if !send(w, .submit, request: wire, a: peer.map(UInt32.init) ?? 0xFFFF_FFFF,
               payload: beginPayload(prompt: r.prompt, sampling: r.sampling)) {
        replace(r.id, expected: wire, avoiding: [w])
      }
    case .resume(let w, let wire, let generated):
      if !send(w, .resume, request: wire, b: UInt32(generated.count),
               payload: beginPayload(prompt: r.prompt, sampling: r.sampling, generated: generated)) {
        replace(r.id, expected: wire, avoiding: [w])
      }
    }
  }

  public func generate(prompt: [Int], sampling: Sampling) -> AsyncThrowingStream<GeneratedToken, Error> {
    let id = nextID.add(1, ordering: .relaxed).oldValue
    return AsyncThrowingStream { continuation in
      let r = Request(id: id, prompt: prompt, sampling: sampling, continuation: continuation)
      let placed: Result<Placement?, Error> = state.withLock { s in
        if s.requests.count >= options.maxInFlight { return .failure(AdmissionError(inFlight: s.requests.count)) }
        switch place(r, &s, avoiding: []) {
        case .success(let p):
          s.requests[id] = r
          return .success(p)
        case .failure(let e):
          guard wait(r, &s, e) else { return .failure(e) }
          s.requests[id] = r
          return .success(nil)  // placed later, by placeWaiting()
        }
      }
      switch placed {
      case .failure(let e):
        continuation.finish(throwing: e)
      case .success(let p):
        continuation.onTermination = { [weak self] reason in
          guard let self, case .cancelled = reason else { return }
          self.cancel(id)
        }
        if let p { execute(r, p) }
      }
    }
  }

  // Moves a request to new workers, if its placement is still `expected` (the same loss
  // can be noticed twice: a closed connection and a failed send).
  private func replace(_ id: UInt64, expected wire: UInt64, avoiding: Set<Int>) {
    enum Outcome { case placed(Request, Placement, [Int], UInt64), waiting([Int], UInt64), failed(Request, Error), none }
    let outcome: Outcome = state.withLock { s in
      guard let r = s.requests[id], !r.finished, r.wire == wire else { return .none }
      let old = [r.prefill, r.decode].compactMap { $0 }
      unplace(r, &s)
      r.pending.removeAll()
      if r.placements >= options.maxPlacements {
        r.finished = true
        s.requests[id] = nil
        return .failed(r, ClusterError.worker("request lost its workers \(r.placements) times"))
      }
      switch place(r, &s, avoiding: avoiding) {
      case .success(let p):
        s.recovered += 1
        return .placed(r, p, old, wire)
      case .failure(let e):
        if wait(r, &s, e) {
          s.recovered += 1
          return .waiting(old, wire)
        }
        r.finished = true
        s.requests[id] = nil
        return .failed(r, e)
      }
    }
    switch outcome {
    case .none: break
    case .failed(let r, let e): r.continuation.finish(throwing: e)
    case .placed(let r, let p, let old, let oldWire):
      for w in Set(old) { send(w, .cancel, request: oldWire) }  // a worker still running it lets go
      execute(r, p)
    case .waiting(let old, let oldWire):
      for w in Set(old) { send(w, .cancel, request: oldWire) }
    }
  }

  // Parks a request that no worker can take right now; false if waiting is not an option
  // (another kind of error, or waiting is turned off).
  private func wait(_ r: Request, _ s: inout State, _ error: Error) -> Bool {
    guard case ClusterError.noWorker = error, options.placementTimeout > 0 else { return false }
    if r.waitingSince == nil { r.waitingSince = monotonicNanos() }
    debugLog("request \(r.id) waits (\(error)); alive: \(slots.map { "\($0.index):\($0.alive)" }.joined(separator: " "))")
    r.prefill = nil
    r.decode = nil
    s.waiting.insert(r.id)
    return true
  }

  // Places waiting requests on workers that have come back; fails those that waited too long.
  private func placeWaiting() {
    let now = monotonicNanos()
    var ready: [(Request, Placement)] = []
    var expired: [(Request, Error)] = []
    state.withLock { s in
      for id in s.waiting.sorted() {
        guard let r = s.requests[id], !r.finished else {
          s.waiting.remove(id)
          continue
        }
        switch place(r, &s, avoiding: []) {
        case .success(let p):
          s.waiting.remove(id)
          debugLog("request \(id) placed after waiting \(secondsSince(r.waitingSince ?? now, now: now))s")
          r.waitingSince = nil
          ready.append((r, p))
        case .failure(let e):
          if secondsSince(r.waitingSince ?? now, now: now) > options.placementTimeout {
            debugLog("request \(id) gives up waiting; alive: \(slots.map { "\($0.index):\($0.alive)" }.joined(separator: " "))")
            s.waiting.remove(id)
            r.finished = true
            s.requests[id] = nil
            expired.append((r, e))
          }
        }
      }
    }
    for (r, p) in ready { execute(r, p) }
    for (r, e) in expired { r.continuation.finish(throwing: e) }
  }

  private func lost(_ slot: WorkerSlot, _ conn: WorkerConnection) {
    guard slot.replace(conn, with: nil) else { return }
    debugLog("lost worker \(slot.index) (\(slot.address.role.rawValue)), last heard \(secondsSince(conn.lastHeard.load(ordering: .relaxed), now: monotonicNanos()))s ago")
    conn.shutdownConnection()
    if stopping.load(ordering: .relaxed) { return }
    let i = slot.index
    // Prefill workers drop their connection to a lost decode worker at once: transfers
    // stuck behind a frozen peer fail instead of holding prefill memory.
    if slot.address.role == .decode {
      for p in slots where p.address.role == .prefill { send(p.index, .peer, a: UInt32(i)) }
    }
    let affected: [(UInt64, UInt64)] = state.withLock { s in
      s.workersLost += 1
      return s.requests.values.filter { r in
        if r.finished { return false }
        if r.decode == i || r.target == i { return true }
        // A lost prefill worker matters until the decode worker has shown it has the
        // request: before its first own token, the KV cache may never have arrived.
        return r.prefill == i && r.nextIndex < 2
      }.map { ($0.id, $0.wire) }
    }
    for (id, wire) in affected { replace(id, expected: wire, avoiding: [i]) }
  }

  func cancel(_ id: UInt64) {
    let r = state.withLock { s -> Request? in
      guard let r = s.requests[id], !r.finished else { return nil }
      r.finished = true
      unplace(r, &s)
      s.requests[id] = nil
      s.waiting.remove(id)
      return r
    }
    guard let r else { return }
    for w in Set([r.prefill, r.decode].compactMap { $0 }) { send(w, .cancel, request: r.wire) }
  }

  func fail(_ id: UInt64, _ error: Error) {
    let r = state.withLock { s -> Request? in
      guard let r = s.requests[id], !r.finished else { return nil }
      r.finished = true
      unplace(r, &s)
      s.requests[id] = nil
      return r
    }
    r?.continuation.finish(throwing: error)
  }

  private func frame(from worker: Int, _ h: FrameHeader, _ p: [UInt8]) {
    switch Ctl(rawValue: h.type) {
    case .token:
      let finish: FinishReason? = h.flags == 0 ? nil : (h.flags == 1 ? .length : .stop)
      deliver(wire: h.request, GeneratedToken(token: Int(h.a), index: Int(h.b), finish: finish))
    case .error:
      let message = String(decoding: p, as: UTF8.self)
      debugLog("error from worker \(worker) for wire \(h.request) (flags \(h.flags)): \(message)")
      let target: (UInt64, Int?)? = state.withLock { s in
        guard let id = s.wire[h.request], let r = s.requests[id] else { return nil }
        return (id, r.decode)
      }
      guard let (id, decode) = target else { return }  // an abandoned placement
      if h.flags & 1 != 0 {
        // The KV cache did not reach the decode worker: place the request again,
        // preferring another decode worker.
        replace(id, expected: h.request, avoiding: Set([decode].compactMap { $0 }))
      } else {
        fail(id, ClusterError.worker(message))
      }
    default:
      break
    }
  }

  // Delivers tokens to the request's stream in index order (the first may come from the
  // prefill worker after the decode worker's second, in principle). Tokens the client has
  // already seen are dropped.
  //
  // Tokens from different workers arrive on different reader threads, so they are yielded
  // to the stream while the lock is held: ordering them under the lock and yielding after
  // it would let one thread overtake another, reordering the stream or ending it before an
  // earlier token was yielded. (yield only enqueues; it never runs the consumer inline.)
  private func deliver(wire: UInt64, _ t: GeneratedToken) {
    state.withLock { s in
      guard let id = s.wire[wire], let r = s.requests[id], !r.finished else { return }
      if t.index == 0, r.queued, let p = r.prefill {
        s.queuedPrompt[p, default: 0] -= r.prompt.count
        r.queued = false
      }
      guard t.index >= r.nextIndex else { return }
      r.pending[t.index] = t
      while let next = r.pending.removeValue(forKey: r.nextIndex) {
        r.generated.append(next.token)
        r.nextIndex += 1
        r.continuation.yield(next)
        if next.finish != nil {
          r.finished = true
          unplace(r, &s)
          s.requests[id] = nil
          r.continuation.finish()
          break
        }
      }
    }
  }
}
