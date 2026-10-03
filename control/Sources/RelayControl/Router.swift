// Routing and admission for a cluster of workers. Pure decisions over a snapshot of the
// workers' state, so they are unit-tested without any workers.
//
// Prefill: requests that share a prompt prefix should land on the same prefill worker, whose
// prefix cache then already holds that prefix. The key is the prompt's first full KV block
// (where system prompts and few-shot examples live); each worker gets a score
// hash(key, worker) and the highest score wins (rendezvous hashing: adding or removing a
// worker moves only the keys that belonged to it). Affinity is abandoned when it would pile
// work on a busy worker: only workers whose queued prompt tokens are within `slack` of the
// least loaded one are candidates.
//
// Decode: the worker with the most free KV blocks after the requests already sent its way,
// since decode capacity is cache memory.
//
// Admission: at most `maxInFlight` requests in the cluster; beyond that the API answers 503
// at once instead of letting queues and latency grow without bound.
import Foundation

public struct WorkerSnapshot: Sendable, Equatable {
  public var index: Int
  public var healthy: Bool
  public var freeBlocks: Int
  public var totalBlocks: Int
  public var queuedPromptTokens: Int  // prompt tokens sent and not yet prefilled
  public var assignedBlocks: Int      // blocks the requests routed here will need
  public init(index: Int, healthy: Bool = true, freeBlocks: Int, totalBlocks: Int, queuedPromptTokens: Int = 0,
              assignedBlocks: Int = 0) {
    self.index = index
    self.healthy = healthy
    self.freeBlocks = freeBlocks
    self.totalBlocks = totalBlocks
    self.queuedPromptTokens = queuedPromptTokens
    self.assignedBlocks = assignedBlocks
  }
}

public struct Router: Sendable {
  public var blockSize: Int
  public var slack: Int  // prompt tokens of extra queue a worker may have and still win on affinity

  public init(blockSize: Int, slack: Int = 2048) {
    self.blockSize = blockSize
    self.slack = slack
  }

  static func mix(_ a: UInt64, _ b: UInt64) -> UInt64 {
    var z = a &+ 0x9E37_79B9_7F4A_7C15 &* (b &+ 1)
    z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
    z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
    return z ^ (z >> 31)
  }

  // The affinity key: a hash of the prompt's first full block, or nil for shorter prompts.
  public func prefixKey(_ prompt: [Int]) -> UInt64? {
    guard prompt.count > blockSize else { return nil }
    var h: UInt64 = 0x6A09_E667_F3BC_C908
    for t in prompt.prefix(blockSize) { h = Router.mix(h, UInt64(UInt32(truncatingIfNeeded: t))) }
    return h
  }

  public func choosePrefill(prompt: [Int], workers: [WorkerSnapshot]) -> Int? {
    let healthy = workers.filter(\.healthy)
    guard let least = healthy.map(\.queuedPromptTokens).min() else { return nil }
    let candidates = healthy.filter { $0.queuedPromptTokens <= least + slack }
    guard let key = prefixKey(prompt) else {
      return candidates.min { ($0.queuedPromptTokens, $0.index) < ($1.queuedPromptTokens, $1.index) }?.index
    }
    return candidates.max { Router.mix(key, UInt64($0.index)) < Router.mix(key, UInt64($1.index)) }?.index
  }

  public func chooseDecode(workers: [WorkerSnapshot]) -> Int? {
    workers.filter(\.healthy).max {
      ($0.freeBlocks - $0.assignedBlocks, -$0.index) < ($1.freeBlocks - $1.assignedBlocks, -$1.index)
    }?.index
  }
}

public struct AdmissionError: Error, CustomStringConvertible {
  public var inFlight: Int
  public var description: String { "overloaded: \(inFlight) requests in flight" }
}
