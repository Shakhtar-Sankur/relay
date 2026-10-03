// Generation as the control plane sees it: send token ids, get a stream of tokens back.
// LocalEngine runs relay's C++ engine in this process through Swift's C++ interop;
// the cluster (M4) implements the same protocol with remote prefill and decode workers.
import Foundation
import CxxStdlib
import RelayBridge
import Synchronization

public struct Sampling: Sendable, Equatable {
  public var temperature: Float = 1.0
  public var topP: Float = 1.0
  public var topK: Int = 0
  public var maxTokens: Int = 16
  public var seed: UInt64 = 0
  public var ignoreEOS: Bool = false
  public init() {}
}

public enum FinishReason: String, Sendable, Codable {
  case length
  case stop
}

public struct GeneratedToken: Sendable, Equatable {
  public var token: Int
  public var index: Int
  public var finish: FinishReason?
}

public enum GenerationError: Error, CustomStringConvertible {
  case rejected(String)
  case failed(String)
  public var description: String {
    switch self {
    case .rejected(let s): return "request rejected: \(s)"
    case .failed(let s): return "generation failed: \(s)"
    }
  }
}

public protocol GenerationBackend: Sendable {
  // The stream ends after the token whose `finish` is set. Ending the stream early
  // (the consumer stops iterating) cancels the request and frees its cache.
  func generate(prompt: [Int], sampling: Sampling) -> AsyncThrowingStream<GeneratedToken, Error>
}

// The C++ engine in this process. One thread runs the engine's step loop; requests are
// added from any task, and each request's tokens are delivered to its own stream.
public final class LocalEngine: GenerationBackend, @unchecked Sendable {
  private let worker: relaybridge.LocalWorker
  private let wake = NSCondition()
  private let streams = Mutex<[UInt64: AsyncThrowingStream<GeneratedToken, Error>.Continuation]>([:])
  private let nextID = Atomic<UInt64>(1)
  private let stopping = Atomic<Bool>(false)
  private var thread: Thread?
  public let layers: Int
  public let vocab: Int

  public init(modelDirectory: String, blocks: Int = 512, blockSize: Int = 16, maxBatchTokens: Int = 512,
              maxSeqs: Int = 64) throws {
    var o = relaybridge.WorkerOptions()
    o.modelDir = std.string(modelDirectory)
    o.blocks = Int32(blocks)
    o.blockSize = Int32(blockSize)
    o.maxBatchTokens = Int32(maxBatchTokens)
    o.maxSeqs = Int32(maxSeqs)
    guard let w = relaybridge.LocalWorker.create(o) else {
      throw GenerationError.failed(String(relaybridge.LocalWorker.lastError()))
    }
    worker = w
    let info = w.info()
    layers = Int(info.layers)
    vocab = Int(info.vocab)
    let t = Thread { [unowned self] in self.loop() }
    t.name = "relay-engine"
    thread = t
    t.start()
  }

  deinit { shutdown() }

  public func shutdown() {
    if stopping.exchange(true, ordering: .acquiringAndReleasing) { return }
    wake.lock()
    wake.broadcast()
    wake.unlock()
    while thread?.isFinished == false { usleep(1000) }
  }

  public var freeBlocks: Int { Int(worker.freeBlocks()) }

  public func generate(prompt: [Int], sampling: Sampling) -> AsyncThrowingStream<GeneratedToken, Error> {
    let id = nextID.add(1, ordering: .relaxed).oldValue
    return AsyncThrowingStream { continuation in
      streams.withLock { $0[id] = continuation }
      continuation.onTermination = { [weak self] reason in
        guard let self else { return }
        if case .cancelled = reason { _ = self.worker.cancel(id) }
        self.streams.withLock { _ = $0.removeValue(forKey: id) }
      }
      var s = relaybridge.SamplingOptions()
      s.temperature = sampling.temperature
      s.topP = sampling.topP
      s.topK = Int32(sampling.topK)
      s.maxNewTokens = Int32(sampling.maxTokens)
      s.seed = sampling.seed
      s.ignoreEos = sampling.ignoreEOS
      let error = String(worker.add(id, relaybridge.TokenVector(prompt.map { Int32($0) }), s))
      if !error.isEmpty {
        streams.withLock { _ = $0.removeValue(forKey: id) }
        continuation.finish(throwing: GenerationError.rejected(error))
        return
      }
      wake.lock()
      wake.signal()
      wake.unlock()
    }
  }

  private func loop() {
    while !stopping.load(ordering: .acquiring) {
      if !worker.hasWork() {
        wake.lock()
        if !worker.hasWork() && !stopping.load(ordering: .acquiring) { _ = wake.wait(until: Date().addingTimeInterval(0.05)) }
        wake.unlock()
        continue
      }
      let r = worker.step()
      if !String(r.error).isEmpty {
        let message = String(r.error)
        let all = streams.withLock { s -> [AsyncThrowingStream<GeneratedToken, Error>.Continuation] in
          defer { s.removeAll() }
          return Array(s.values)
        }
        for c in all { c.finish(throwing: GenerationError.failed(message)) }
        continue
      }
      for e in r.events {
        let finish: FinishReason? = e.finish == 0 ? nil : (e.finish == 1 ? .length : .stop)
        guard let c = streams.withLock({ $0[e.request] }) else { continue }
        c.yield(GeneratedToken(token: Int(e.token), index: Int(e.index), finish: finish))
        if finish != nil {
          streams.withLock { _ = $0.removeValue(forKey: e.request) }
          c.finish()
        }
      }
    }
    let all = streams.withLock { s -> [AsyncThrowingStream<GeneratedToken, Error>.Continuation] in
      defer { s.removeAll() }
      return Array(s.values)
    }
    for c in all { c.finish(throwing: GenerationError.failed("engine shut down")) }
  }
}
