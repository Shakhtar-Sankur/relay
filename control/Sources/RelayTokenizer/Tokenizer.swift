// A byte-pair-encoding tokenizer that reads Hugging Face's tokenizer.json and produces
// the same token ids as the `tokenizers` library for the pipelines relay's models use:
//
//   - GPT-2 style byte-level BPE (SmolLM2, Llama 3): optional per-digit split, the GPT-2
//     regex, bytes mapped to printable characters, then BPE.
//   - Split-then-byte-level BPE (Qwen2): NFC, a model-specific regex, bytes, BPE.
//   - SentencePiece-style BPE (Llama 2, TinyLlama): spaces become "▁", the whole text is
//     one word, characters missing from the vocabulary fall back to <0xNN> byte tokens.
//
// Anything else in a tokenizer.json is rejected at load time rather than silently
// tokenized differently.
import Foundation

public enum TokenizerError: Error, CustomStringConvertible {
  case unsupported(String)
  case malformed(String)

  public var description: String {
    switch self {
    case .unsupported(let s): return "unsupported tokenizer feature: \(s)"
    case .malformed(let s): return "malformed tokenizer.json: \(s)"
    }
  }
}

public final class Tokenizer: @unchecked Sendable {
  // Pre-tokenization steps, applied in order; each splits every piece further.
  enum PreStep {
    case digits                                  // each numeric character on its own
    case regex(Regex<AnyRegexOutput>)            // matches and gaps both become pieces
    case byteLevel(regex: Regex<AnyRegexOutput>?, addPrefixSpace: Bool)  // optional split, then map bytes
  }

  enum Normalizer {
    case nfc
    case prepend(String)
    case replace(String, String)
  }

  enum DecoderKind {
    case byteLevel
    case sentencePiece(stripLeading: Int)  // ▁ -> space, <0xNN> bytes, strip N leading spaces
  }

  public let vocabSize: Int
  let vocab: [String: Int]
  let idToToken: [String]
  let mergeRank: [Int64: (rank: Int, merged: Int)]  // (left id, right id) -> rank and result
  let byteFallback: Bool
  let ignoreMerges: Bool
  let unkId: Int?
  let normalizers: [Normalizer]
  let preSteps: [PreStep]
  let decoder: DecoderKind
  let byteLevelModel: Bool
  public let specialIds: Set<Int>
  let addedTokens: [(content: String, id: Int)]  // longest first
  let prefixTokens: [Int]  // added by the post-processor when add_special_tokens
  let suffixTokens: [Int]
  let byteTokenIds: [Int]  // <0x00>..<0xFF> ids for byte fallback

  public convenience init(contentsOf url: URL) throws {
    let data = try Data(contentsOf: url)
    guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
      throw TokenizerError.malformed("not a JSON object")
    }
    try self.init(json: json)
  }

  public init(json: [String: Any]) throws {
    guard let model = json["model"] as? [String: Any] else { throw TokenizerError.malformed("no model") }
    guard (model["type"] as? String) == "BPE" else {
      throw TokenizerError.unsupported("model type \(model["type"] ?? "nil") (only BPE)")
    }
    guard let v = model["vocab"] as? [String: Any] else { throw TokenizerError.malformed("no vocab") }
    var vocab: [String: Int] = [:]
    for (k, x) in v { vocab[k] = (x as? NSNumber)?.intValue ?? (x as! Int) }

    var added: [(String, Int)] = []
    var special = Set<Int>()
    for a in json["added_tokens"] as? [[String: Any]] ?? [] {
      guard let content = a["content"] as? String, let id = (a["id"] as? NSNumber)?.intValue else { continue }
      added.append((content, id))
      vocab[content] = id
      if (a["special"] as? Bool) ?? false { special.insert(id) }
    }
    let size = (vocab.values.max() ?? -1) + 1
    var idToToken = [String](repeating: "", count: size)
    for (k, id) in vocab { idToToken[id] = k }

    var ranks: [Int64: (Int, Int)] = [:]
    let merges = model["merges"] as? [Any] ?? []
    for (rank, m) in merges.enumerated() {
      let pair: [String]
      if let s = m as? String {
        pair = s.split(separator: " ", maxSplits: 1, omittingEmptySubsequences: false).map(String.init)
      } else if let a = m as? [String] {
        pair = a
      } else {
        throw TokenizerError.malformed("merge \(rank)")
      }
      guard pair.count == 2, let l = vocab[pair[0]], let r = vocab[pair[1]], let merged = vocab[pair[0] + pair[1]] else {
        continue  // a merge whose parts or result are not in the vocabulary can never apply
      }
      let key = Int64(l) << 32 | Int64(r)
      if ranks[key] == nil { ranks[key] = (rank, merged) }
    }

    if let p = model["continuing_subword_prefix"] as? String, !p.isEmpty {
      throw TokenizerError.unsupported("continuing_subword_prefix")
    }
    if let p = model["end_of_word_suffix"] as? String, !p.isEmpty { throw TokenizerError.unsupported("end_of_word_suffix") }
    if let d = model["dropout"] as? NSNumber, d.doubleValue > 0 { throw TokenizerError.unsupported("BPE dropout") }

    self.vocab = vocab
    self.idToToken = idToToken
    self.vocabSize = size
    self.mergeRank = ranks.mapValues { (rank: $0.0, merged: $0.1) }
    self.byteFallback = (model["byte_fallback"] as? Bool) ?? false
    self.ignoreMerges = (model["ignore_merges"] as? Bool) ?? false
    self.unkId = (model["unk_token"] as? String).flatMap { vocab[$0] }
    self.specialIds = special
    self.addedTokens = added.sorted { $0.0.count > $1.0.count }
    self.byteTokenIds = (0..<256).map { vocab[String(format: "<0x%02X>", $0)] ?? -1 }

    self.normalizers = try Tokenizer.parseNormalizer(json["normalizer"])
    let steps = try Tokenizer.parsePreTokenizer(json["pre_tokenizer"])
    self.preSteps = steps
    self.byteLevelModel = steps.contains { if case .byteLevel = $0 { return true } else { return false } }
    self.decoder = try Tokenizer.parseDecoder(json["decoder"])
    (self.prefixTokens, self.suffixTokens) = try Tokenizer.parsePostProcessor(json["post_processor"], vocab: vocab)
    if byteFallback && byteTokenIds.contains(-1) { throw TokenizerError.malformed("byte_fallback without all <0xNN> tokens") }
  }

  // MARK: - parsing the pipeline

  static func parseNormalizer(_ n: Any?) throws -> [Normalizer] {
    guard let n = n as? [String: Any] else { return [] }
    switch n["type"] as? String {
    case "Sequence":
      return try (n["normalizers"] as? [Any] ?? []).flatMap { try parseNormalizer($0) }
    case "NFC": return [.nfc]
    case "Prepend": return [.prepend(n["prepend"] as? String ?? "")]
    case "Replace":
      guard let pat = n["pattern"] as? [String: Any], let s = pat["String"] as? String else {
        throw TokenizerError.unsupported("Replace normalizer with a regex")
      }
      return [.replace(s, n["content"] as? String ?? "")]
    default: throw TokenizerError.unsupported("normalizer \(n["type"] ?? "nil")")
    }
  }

  // GPT-2's pre-tokenization regex, used by ByteLevel(use_regex: true).
  static let gpt2Pattern =
    #"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"#

  static func makeRegex(_ pattern: String) throws -> Regex<AnyRegexOutput> {
    // Hugging Face matches Unicode scalars, not grapheme clusters ("\r\n" is two characters there).
    return try Regex(pattern).matchingSemantics(.unicodeScalar)
  }

  static func parsePreTokenizer(_ p: Any?) throws -> [PreStep] {
    guard let p = p as? [String: Any] else { return [] }
    switch p["type"] as? String {
    case "Sequence":
      return try (p["pretokenizers"] as? [Any] ?? []).flatMap { try parsePreTokenizer($0) }
    case "Digits":
      guard (p["individual_digits"] as? Bool) == true else { throw TokenizerError.unsupported("Digits without individual_digits") }
      return [.digits]
    case "ByteLevel":
      let useRegex = (p["use_regex"] as? Bool) ?? true
      return [.byteLevel(regex: useRegex ? try makeRegex(gpt2Pattern) : nil,
                         addPrefixSpace: (p["add_prefix_space"] as? Bool) ?? false)]
    case "Split":
      guard (p["behavior"] as? String) == "Isolated", (p["invert"] as? Bool) != true else {
        throw TokenizerError.unsupported("Split behavior \(p["behavior"] ?? "nil")")
      }
      guard let pat = p["pattern"] as? [String: Any], let re = pat["Regex"] as? String else {
        throw TokenizerError.unsupported("Split with a string pattern")
      }
      return [.regex(try makeRegex(re))]
    default:
      throw TokenizerError.unsupported("pre-tokenizer \(p["type"] ?? "nil")")
    }
  }

  static func parseDecoder(_ d: Any?) throws -> DecoderKind {
    guard let d = d as? [String: Any] else { return .byteLevel }
    switch d["type"] as? String {
    case "ByteLevel": return .byteLevel
    case "Sequence":
      // The SentencePiece-style sequence: Replace(▁, " "), ByteFallback, Fuse, Strip.
      let parts = d["decoders"] as? [[String: Any]] ?? []
      let types = parts.compactMap { $0["type"] as? String }
      guard types.contains("ByteFallback") || types.contains("Replace") else {
        throw TokenizerError.unsupported("decoder sequence \(types)")
      }
      let strip = parts.first { ($0["type"] as? String) == "Strip" }.flatMap { ($0["start"] as? NSNumber)?.intValue } ?? 0
      return .sentencePiece(stripLeading: strip)
    default:
      throw TokenizerError.unsupported("decoder \(d["type"] ?? "nil")")
    }
  }

  static func parsePostProcessor(_ p: Any?, vocab: [String: Int]) throws -> ([Int], [Int]) {
    guard let p = p as? [String: Any] else { return ([], []) }
    switch p["type"] as? String {
    case "ByteLevel": return ([], [])  // only adjusts offsets
    case "TemplateProcessing":
      var prefix: [Int] = [], suffix: [Int] = []
      var seenSequence = false
      for item in p["single"] as? [[String: Any]] ?? [] {
        if item["Sequence"] != nil {
          seenSequence = true
        } else if let s = item["SpecialToken"] as? [String: Any], let name = s["id"] as? String {
          guard let id = vocab[name] else { throw TokenizerError.malformed("template token \(name)") }
          if seenSequence { suffix.append(id) } else { prefix.append(id) }
        }
      }
      return (prefix, suffix)
    default:
      throw TokenizerError.unsupported("post-processor \(p["type"] ?? "nil")")
    }
  }

  // MARK: - encoding

  public func encode(_ text: String, addSpecialTokens: Bool = true) -> [Int] {
    var ids: [Int] = []
    if addSpecialTokens { ids += prefixTokens }
    for segment in splitAddedTokens(text) {
      switch segment {
      case .added(let id): ids.append(id)
      case .text(let s):
        let normalized = normalize(s)
        for piece in preTokenize(normalized) { ids += bpe(piece) }
      }
    }
    if addSpecialTokens { ids += suffixTokens }
    return ids
  }

  enum Segment {
    case text(String)
    case added(Int)
  }

  // Added tokens (special ones like <|im_start|>) are found in the raw text first and never
  // go through normalization or BPE.
  func splitAddedTokens(_ text: String) -> [Segment] {
    guard !addedTokens.isEmpty else { return [.text(text)] }
    var out: [Segment] = []
    let scalars = Array(text.unicodeScalars)
    let tokens = addedTokens.map { (Array($0.content.unicodeScalars), $0.id) }
    var start = 0, i = 0
    while i < scalars.count {
      var matched: (len: Int, id: Int)? = nil
      for (t, id) in tokens where !t.isEmpty && i + t.count <= scalars.count {
        if scalars[i] == t[0] && Array(scalars[i..<(i + t.count)]) == t {
          matched = (t.count, id)
          break  // longest first
        }
      }
      if let m = matched {
        if start < i { out.append(.text(String(String.UnicodeScalarView(scalars[start..<i])))) }
        out.append(.added(m.id))
        i += m.len
        start = i
      } else {
        i += 1
      }
    }
    if start < scalars.count { out.append(.text(String(String.UnicodeScalarView(scalars[start...])))) }
    return out
  }

  func normalize(_ s: String) -> String {
    var s = s
    for n in normalizers {
      switch n {
      case .nfc: s = s.precomposedStringWithCanonicalMapping
      case .prepend(let p): if !s.isEmpty { s = p + s }
      case .replace(let a, let b): s = s.replacingOccurrences(of: a, with: b)
      }
    }
    return s
  }

  static func split(_ piece: String, _ regex: Regex<AnyRegexOutput>) -> [String] {
    var out: [String] = []
    let scalars = piece.unicodeScalars
    var last = scalars.startIndex
    for m in piece.matches(of: regex) {
      let r = m.range
      if r.isEmpty { continue }
      if last < r.lowerBound { out.append(String(scalars[last..<r.lowerBound])) }
      out.append(String(scalars[r]))
      last = r.upperBound
    }
    if last < scalars.endIndex { out.append(String(scalars[last...])) }
    return out
  }

  func preTokenize(_ s: String) -> [String] {
    if s.isEmpty { return [] }
    var pieces = [s]
    for step in preSteps {
      var next: [String] = []
      for p in pieces {
        switch step {
        case .digits:
          var cur = String.UnicodeScalarView()
          for c in p.unicodeScalars {
            switch c.properties.generalCategory {
            case .decimalNumber, .letterNumber, .otherNumber:
              if !cur.isEmpty { next.append(String(cur)); cur = String.UnicodeScalarView() }
              next.append(String(c))
            default:
              cur.append(c)
            }
          }
          if !cur.isEmpty { next.append(String(cur)) }
        case .regex(let re):
          next += Tokenizer.split(p, re)
        case .byteLevel(let re, _):
          let parts = re.map { Tokenizer.split(p, $0) } ?? [p]
          for part in parts { next.append(Tokenizer.byteEncode(part)) }
        }
      }
      pieces = next
    }
    if case .byteLevel(_, let addPrefixSpace)? = preSteps.last(where: { if case .byteLevel = $0 { return true }; return false }),
      addPrefixSpace, let first = pieces.first, !first.hasPrefix("Ġ")
    {
      pieces[0] = "Ġ" + first
    }
    return pieces
  }

  // GPT-2's reversible map from bytes to printable characters.
  static let byteToChar: [Character] = {
    var bs: [Int] = Array(33...126) + Array(161...172) + Array(174...255)
    var cs = bs
    var n = 0
    for b in 0..<256 where !bs.contains(b) {
      bs.append(b)
      cs.append(256 + n)
      n += 1
    }
    var table = [Character](repeating: " ", count: 256)
    for (b, c) in zip(bs, cs) { table[b] = Character(Unicode.Scalar(UInt32(c))!) }
    return table
  }()

  static let charToByte: [Character: UInt8] = {
    var m: [Character: UInt8] = [:]
    for (b, c) in byteToChar.enumerated() { m[c] = UInt8(b) }
    return m
  }()

  static func byteEncode(_ s: String) -> String {
    var out = ""
    for b in s.utf8 { out.append(byteToChar[Int(b)]) }
    return out
  }

  // MARK: - BPE

  func bpe(_ piece: String) -> [Int] {
    if piece.isEmpty { return [] }
    if ignoreMerges, let id = vocab[piece] { return [id] }
    // Initial symbols: one per character (bytes already mapped for byte-level models).
    var symbols: [Int] = []
    var unkRun = false
    for c in piece.unicodeScalars {
      if let id = vocab[String(c)] {
        symbols.append(id)
        unkRun = false
      } else if byteFallback {
        for b in String(c).utf8 { symbols.append(byteTokenIds[Int(b)]) }
        unkRun = false
      } else if let unk = unkId {
        if !unkRun { symbols.append(unk) }  // fuse_unk
        unkRun = true
      }
    }
    // Repeatedly merge the adjacent pair with the lowest rank.
    while symbols.count > 1 {
      var best = Int.max, at = -1, merged = 0
      for i in 0..<(symbols.count - 1) {
        if let m = mergeRank[Int64(symbols[i]) << 32 | Int64(symbols[i + 1])], m.rank < best {
          best = m.rank
          at = i
          merged = m.merged
        }
      }
      if at < 0 { break }
      symbols[at] = merged
      symbols.remove(at: at + 1)
    }
    return symbols
  }

  // MARK: - decoding

  public func token(_ id: Int) -> String? { id >= 0 && id < idToToken.count ? idToToken[id] : nil }
  public func id(_ token: String) -> Int? { vocab[token] }

  public func decode(_ ids: [Int], skipSpecialTokens: Bool = true) -> String {
    switch decoder {
    case .byteLevel:
      var bytes: [UInt8] = []
      for id in ids {
        guard let t = token(id) else { continue }
        if specialIds.contains(id) {
          if !skipSpecialTokens { bytes += Array(t.utf8) }
          continue
        }
        if addedTokens.contains(where: { $0.id == id }) && Tokenizer.charToByte[t.first ?? " "] == nil {
          bytes += Array(t.utf8)  // a non-special added token is plain text
          continue
        }
        for ch in t {
          if let b = Tokenizer.charToByte[ch] { bytes.append(b) } else { bytes += Array(String(ch).utf8) }
        }
      }
      return String(decoding: bytes, as: UTF8.self)
    case .sentencePiece(let strip):
      var out = ""
      var pending: [UInt8] = []
      func flush() {
        if !pending.isEmpty {
          out += String(decoding: pending, as: UTF8.self)
          pending.removeAll()
        }
      }
      for id in ids {
        guard let t = token(id) else { continue }
        if specialIds.contains(id) {
          if !skipSpecialTokens { flush(); out += t }
          continue
        }
        if t.count == 6, t.hasPrefix("<0x"), t.hasSuffix(">"), let b = UInt8(t.dropFirst(3).prefix(2), radix: 16) {
          pending.append(b)
        } else {
          flush()
          out += t.replacingOccurrences(of: "▁", with: " ")
        }
      }
      flush()
      var n = strip
      while n > 0 && out.hasPrefix(" ") {
        out.removeFirst()
        n -= 1
      }
      return out
    }
  }
}

// Turns a stream of token ids into a stream of text, emitting text only once it is
// complete: a character split across byte tokens is held back until all its bytes have
// arrived. Works for every decoder by decoding a short window and taking the difference
// (the method text-generation-inference uses).
public struct StreamingDecoder {
  let tokenizer: Tokenizer
  var ids: [Int] = []
  var prefixOffset = 0
  var readOffset = 0

  public init(_ tokenizer: Tokenizer) { self.tokenizer = tokenizer }

  public mutating func push(_ id: Int) -> String {
    ids.append(id)
    let before = tokenizer.decode(Array(ids[prefixOffset..<readOffset]))
    let after = tokenizer.decode(Array(ids[prefixOffset...]))
    if after.count > before.count, !after.hasSuffix("\u{FFFD}") {
      let new = String(after.dropFirst(before.count))
      prefixOffset = readOffset
      readOffset = ids.count
      return new
    }
    return ""
  }

  // Whatever is left, complete or not.
  public mutating func finish() -> String {
    let before = tokenizer.decode(Array(ids[prefixOffset..<readOffset]))
    let after = tokenizer.decode(Array(ids[prefixOffset...]))
    readOffset = ids.count
    prefixOffset = readOffset
    return after.count > before.count ? String(after.dropFirst(before.count)) : ""
  }
}
