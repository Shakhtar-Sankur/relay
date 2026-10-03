// What the control plane needs from a model directory besides the weights: the tokenizer,
// the chat template, and which token ids end a generation.
import Foundation
import RelayTokenizer

public struct ChatMessage: Codable, Sendable, Equatable {
  public var role: String
  public var content: String
  public init(role: String, content: String) {
    self.role = role
    self.content = content
  }
}

public enum ChatTemplateError: Error, CustomStringConvertible {
  case noTemplate
  case unsupported(String)
  public var description: String {
    switch self {
    case .noTemplate: return "the model has no chat_template in tokenizer_config.json"
    case .unsupported(let s): return "unsupported chat template: \(s)"
    }
  }
}

// relay renders the chat formats its models use rather than running a Jinja interpreter:
// the template text is matched to a known format, and the tests compare every rendering
// with transformers' apply_chat_template.
public struct ChatTemplate: Sendable {
  public enum Format: Sendable, Equatable {
    case chatML(defaultSystem: String?)  // SmolLM2, Qwen2
    case zephyr(eos: String)             // TinyLlama
    case llama3(bos: String)             // Llama 3.x
  }

  public let format: Format

  public init(format: Format) { self.format = format }

  public init(template: String, eosToken: String?, bosToken: String?) throws {
    if template.contains("<|im_start|>") {
      format = .chatML(defaultSystem: ChatTemplate.defaultSystem(in: template))
    } else if template.contains("<|user|>") && template.contains("<|assistant|>") {
      guard let eos = eosToken else { throw ChatTemplateError.unsupported("zephyr format without an eos_token") }
      format = .zephyr(eos: eos)
    } else if template.contains("<|start_header_id|>") {
      format = .llama3(bos: bosToken ?? "<|begin_of_text|>")
    } else {
      throw ChatTemplateError.unsupported("not ChatML, Zephyr or Llama 3")
    }
  }

  // The system prompt a ChatML template inserts when the conversation has none: written
  // either inside the system block itself or as a literal starting with "You are".
  static func defaultSystem(in template: String) -> String? {
    if let r = template.firstMatch(of: /<\|im_start\|>system\n([^'"<{}]+)<\|im_end\|>/) { return String(r.1) }
    if let r = template.firstMatch(of: /'(You are [^']+)'/) { return String(r.1) }
    return nil
  }

  public func render(_ messages: [ChatMessage], addGenerationPrompt: Bool = true) -> String {
    var out = ""
    switch format {
    case .chatML(let defaultSystem):
      if let d = defaultSystem, messages.first?.role != "system" {
        out += "<|im_start|>system\n\(d)<|im_end|>\n"
      }
      for m in messages { out += "<|im_start|>\(m.role)\n\(m.content)<|im_end|>\n" }
      if addGenerationPrompt { out += "<|im_start|>assistant\n" }
    case .zephyr(let eos):
      for m in messages where ["system", "user", "assistant"].contains(m.role) {
        out += "<|\(m.role)|>\n\(m.content)\(eos)\n"
      }
      if addGenerationPrompt { out += "<|assistant|>\n" }
    case .llama3(let bos):
      out += bos
      for m in messages {
        out += "<|start_header_id|>\(m.role)<|end_header_id|>\n\n\(m.content.trimmingCharacters(in: .whitespacesAndNewlines))<|eot_id|>"
      }
      if addGenerationPrompt { out += "<|start_header_id|>assistant<|end_header_id|>\n\n" }
    }
    return out
  }
}

public struct ModelBundle: Sendable {
  public let directory: URL
  public let name: String
  public let tokenizer: Tokenizer
  public let chatTemplate: ChatTemplate?
  // Token ids that end a generation: config.json, generation_config.json and the
  // tokenizer's eos_token together (chat models often end turns with a token the base
  // config does not list).
  public let stopTokenIDs: Set<Int>
  public let maxContext: Int

  public init(directory: URL) throws {
    self.directory = directory
    self.name = directory.lastPathComponent
    tokenizer = try Tokenizer(contentsOf: directory.appendingPathComponent("tokenizer.json"))

    func json(_ file: String) -> [String: Any] {
      guard let d = try? Data(contentsOf: directory.appendingPathComponent(file)),
        let j = try? JSONSerialization.jsonObject(with: d) as? [String: Any]
      else { return [:] }
      return j
    }
    func ids(_ v: Any?) -> [Int] {
      if let n = v as? NSNumber { return [n.intValue] }
      if let a = v as? [NSNumber] { return a.map(\.intValue) }
      return []
    }
    func tokenString(_ v: Any?) -> String? {
      if let s = v as? String { return s }
      if let d = v as? [String: Any] { return d["content"] as? String }
      return nil
    }
    let config = json("config.json"), generation = json("generation_config.json"), tc = json("tokenizer_config.json")
    var stops = Set(ids(config["eos_token_id"]) + ids(generation["eos_token_id"]))
    let eosToken = tokenString(tc["eos_token"]), bosToken = tokenString(tc["bos_token"])
    if let e = eosToken, let id = tokenizer.id(e) { stops.insert(id) }
    stopTokenIDs = stops
    maxContext = (config["max_position_embeddings"] as? NSNumber)?.intValue ?? 2048
    if let t = tc["chat_template"] as? String {
      chatTemplate = try ChatTemplate(template: t, eosToken: eosToken, bosToken: bosToken)
    } else {
      chatTemplate = nil
    }
  }
}
