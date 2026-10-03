// relay-server: the OpenAI-compatible API in front of a model.
//
//   relay-server --model DIR [--host 127.0.0.1] [--port 8000] [--blocks 512] [--block-size 16]
//                [--max-batch-tokens 512]
//       runs the engine in this process (CPU)
//
//   relay-server --model DIR --workers prefill=H:P,decode=H:P,... [--max-in-flight 256]
//                [--heartbeat-timeout 2] [--stall-timeout 30]
//       fronts a cluster of relay-worker processes (role=both workers for colocated
//       serving; prefill and decode workers for disaggregated serving). The model directory
//       is still read here for the tokenizer and chat template.
import Foundation
import RelayControl

func usage() -> Never {
  FileHandle.standardError.write(Data("""
    usage: relay-server --model DIR [--host 127.0.0.1] [--port 8000] [--blocks 512] [--block-size 16] \
    [--max-batch-tokens 512]\n
    """.utf8))
  exit(2)
}

var options: [String: String] = [:]
var args = Array(CommandLine.arguments.dropFirst())
while !args.isEmpty {
  let k = args.removeFirst()
  guard k.hasPrefix("--"), !args.isEmpty else { usage() }
  options[String(k.dropFirst(2))] = args.removeFirst()
}
guard let model = options["model"] else { usage() }

do {
  let bundle = try ModelBundle(directory: URL(fileURLWithPath: model))
  let backend: GenerationBackend
  let description: String
  if let spec = options["workers"] {
    let addresses = spec.split(separator: ",").map { WorkerAddress(String($0)) }
    guard addresses.allSatisfy({ $0 != nil }) else { usage() }
    var o = ClusterOptions()
    o.maxInFlight = Int(options["max-in-flight"] ?? "") ?? o.maxInFlight
    o.heartbeatTimeout = Double(options["heartbeat-timeout"] ?? "") ?? o.heartbeatTimeout
    o.stallTimeout = Double(options["stall-timeout"] ?? "") ?? o.stallTimeout
    let cluster = try Cluster(addresses: addresses.compactMap { $0 }, options: o)
    backend = cluster
    description = "\(cluster.disaggregated ? "disaggregated" : "colocated") cluster of \(addresses.count) workers"
  } else {
    let engine = try LocalEngine(modelDirectory: model, blocks: Int(options["blocks"] ?? "512") ?? 512,
                                 blockSize: Int(options["block-size"] ?? "16") ?? 16,
                                 maxBatchTokens: Int(options["max-batch-tokens"] ?? "512") ?? 512)
    backend = engine
    description = "in-process engine, \(engine.layers) layers"
  }
  let api = OpenAIServer(bundle: bundle, backend: backend)
  let host = options["host"] ?? "127.0.0.1"
  let server = try HTTPServer(host: host, port: Int(options["port"] ?? "8000") ?? 8000) { req, w in
    await api.handle(req, w)
  }
  server.start()
  // FileHandle writes are unbuffered: the line appears at once, even when piped.
  FileHandle.standardOutput.write(Data(("relay-server: \(bundle.name) (\(description)) on http://\(host):\(server.port)\n").utf8))
  while true { sleep(3600) }
} catch {
  FileHandle.standardError.write(Data("relay-server: \(error)\n".utf8))
  exit(1)
}
