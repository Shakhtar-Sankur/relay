// relay-server: the OpenAI-compatible API in front of a model.
//
//   relay-server --model DIR [--host 127.0.0.1] [--port 8000] [--blocks 512] [--block-size 16]
//                [--max-batch-tokens 512]
//
// This runs the engine in this process (CPU). For prefill and decode on separate worker
// processes, see relay-cluster.
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
  let engine = try LocalEngine(modelDirectory: model, blocks: Int(options["blocks"] ?? "512") ?? 512,
                               blockSize: Int(options["block-size"] ?? "16") ?? 16,
                               maxBatchTokens: Int(options["max-batch-tokens"] ?? "512") ?? 512)
  let api = OpenAIServer(bundle: bundle, backend: engine)
  let host = options["host"] ?? "127.0.0.1"
  let server = try HTTPServer(host: host, port: Int(options["port"] ?? "8000") ?? 8000) { req, w in
    await api.handle(req, w)
  }
  server.start()
  // FileHandle writes are unbuffered: the line appears at once, even when piped.
  FileHandle.standardOutput.write(Data(("relay-server: \(bundle.name) (\(engine.layers) layers, vocab \(engine.vocab)) on http://\(host):\(server.port)\n").utf8))
  while true { sleep(3600) }
} catch {
  FileHandle.standardError.write(Data("relay-server: \(error)\n".utf8))
  exit(1)
}
