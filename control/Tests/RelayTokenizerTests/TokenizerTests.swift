// The Swift tokenizer against Hugging Face `tokenizers`: same ids for every case, same
// text back. Fixtures from scripts/make_tokenizer_fixtures.py; real models are checked
// too when RELAY_MODELS_DIR points at a directory of model folders.
import Foundation
import Testing

@testable import RelayTokenizer

struct Case: Decodable {
  let text: String
  let ids: [Int]
  let decoded: String
  let decoded_with_special: String
}

let repoRoot = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
  .deletingLastPathComponent().deletingLastPathComponent()
let fixtures = repoRoot.appendingPathComponent("tests/fixtures/tokenizers")

func check(tokenizer: URL, cases: URL) throws -> (passed: Int, failed: [String]) {
  let tok = try Tokenizer(contentsOf: tokenizer)
  let all = try JSONDecoder().decode([Case].self, from: Data(contentsOf: cases))
  var failed: [String] = []
  for c in all {
    let ids = tok.encode(c.text)
    if ids != c.ids { failed.append("encode \(c.text.debugDescription): got \(ids), want \(c.ids)") }
    let text = tok.decode(c.ids)
    if text != c.decoded { failed.append("decode \(c.ids): got \(text.debugDescription), want \(c.decoded.debugDescription)") }
    let withSpecial = tok.decode(c.ids, skipSpecialTokens: false)
    if withSpecial != c.decoded_with_special {
      failed.append("decode+special \(c.ids): got \(withSpecial.debugDescription), want \(c.decoded_with_special.debugDescription)")
    }
  }
  return (all.count - failed.count, failed)
}

@Test(arguments: ["gpt2-digits", "split-bytelevel", "sentencepiece"])
func matchesHuggingFace(kind: String) throws {
  let r = try check(tokenizer: fixtures.appendingPathComponent("\(kind).json"),
                    cases: fixtures.appendingPathComponent("\(kind).cases.json"))
  for f in r.failed.prefix(5) { Issue.record(Comment(rawValue: f)) }
  #expect(r.failed.isEmpty)
}

@Test func matchesHuggingFaceOnRealModels() throws {
  guard let dir = ProcessInfo.processInfo.environment["RELAY_MODELS_DIR"] else { return }
  let models = try FileManager.default.contentsOfDirectory(atPath: dir).sorted()
  for m in models {
    let base = URL(fileURLWithPath: dir).appendingPathComponent(m)
    let cases = base.appendingPathComponent("relay-tokenizer-cases.json")
    guard FileManager.default.fileExists(atPath: cases.path) else { continue }
    let r = try check(tokenizer: base.appendingPathComponent("tokenizer.json"), cases: cases)
    print("  \(m): \(r.passed) cases match")
    for f in r.failed.prefix(5) { Issue.record(Comment(rawValue: "\(m): \(f)")) }
    #expect(r.failed.isEmpty)
  }
}

@Test func streamingDecodeEqualsWholeDecode() throws {
  for kind in ["gpt2-digits", "sentencepiece"] {
    let tok = try Tokenizer(contentsOf: fixtures.appendingPathComponent("\(kind).json"))
    let text = "Streaming: café 東京 🚀 x² done."
    let ids = tok.encode(text, addSpecialTokens: false)
    var d = StreamingDecoder(tok)
    var out = ""
    var pieces = 0
    for id in ids {
      let s = d.push(id)
      if !s.isEmpty { pieces += 1 }
      #expect(!s.contains("\u{FFFD}"), "a partial character escaped: \(s.debugDescription)")
      out += s
    }
    out += d.finish()
    #expect(out == tok.decode(ids), "\(kind): \(out.debugDescription)")
    #expect(pieces > 1)
  }
}

@Test func rejectsUnsupportedPipelines() throws {
  var json = try JSONSerialization.jsonObject(
    with: Data(contentsOf: fixtures.appendingPathComponent("gpt2-digits.json"))) as! [String: Any]
  json["pre_tokenizer"] = ["type": "Whitespace"]
  #expect(throws: TokenizerError.self) { try Tokenizer(json: json) }
}
