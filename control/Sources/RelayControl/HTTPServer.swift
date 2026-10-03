// A small HTTP/1.1 server on POSIX sockets: enough for an OpenAI-compatible API with
// server-sent events. Each connection gets a thread that reads one request, runs the async
// handler, and writes the response (a body, or a stream of SSE events) before closing.
// No keep-alive: every response ends with the connection.
#if canImport(Glibc)
  import Glibc
#elseif canImport(Darwin)
  import Darwin
#endif
import Foundation

public struct HTTPRequest: Sendable {
  public var method: String
  public var path: String
  public var headers: [String: String]  // lower-cased names
  public var body: Data
}

public enum HTTPServerError: Error {
  case socket(String)
}

// Writes the response. `start` sends the status line and headers; with no Content-Length
// the body runs until the connection closes, which is how SSE is sent here.
public final class ResponseWriter: @unchecked Sendable {
  let fd: Int32
  private let lock = NSLock()
  private var _started = false
  public private(set) var closedByPeer = false
  public var started: Bool {
    lock.lock()
    defer { lock.unlock() }
    return _started
  }

  init(fd: Int32) { self.fd = fd }

  public func start(status: Int, headers: [(String, String)]) {
    lock.lock()
    defer { lock.unlock() }
    guard !_started else { return }
    _started = true
    var head = "HTTP/1.1 \(status) \(ResponseWriter.reason(status))\r\n"
    for (k, v) in headers { head += "\(k): \(v)\r\n" }
    head += "Connection: close\r\n\r\n"
    writeLocked(Data(head.utf8))
  }

  // false once the client has gone away
  @discardableResult
  public func write(_ data: Data) -> Bool {
    lock.lock()
    defer { lock.unlock() }
    return writeLocked(data)
  }

  @discardableResult
  private func writeLocked(_ data: Data) -> Bool {
    if closedByPeer { return false }
    return data.withUnsafeBytes { raw -> Bool in
      var off = 0
      while off < raw.count {
        #if canImport(Glibc)
          let n = send(fd, raw.baseAddress! + off, raw.count - off, Int32(MSG_NOSIGNAL))
        #else
          let n = send(fd, raw.baseAddress! + off, raw.count - off, 0)
        #endif
        if n <= 0 {
          if n < 0 && errno == EINTR { continue }
          closedByPeer = true
          return false
        }
        off += n
      }
      return true
    }
  }

  public func respond(status: Int, contentType: String, body: Data) {
    start(status: status, headers: [("Content-Type", contentType), ("Content-Length", "\(body.count)")])
    write(body)
  }

  public func respondJSON(status: Int = 200, _ object: some Encodable) {
    let enc = JSONEncoder()
    enc.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
    respond(status: status, contentType: "application/json", body: (try? enc.encode(object)) ?? Data("{}".utf8))
  }

  static func reason(_ status: Int) -> String {
    switch status {
    case 200: return "OK"
    case 400: return "Bad Request"
    case 404: return "Not Found"
    case 405: return "Method Not Allowed"
    case 413: return "Payload Too Large"
    case 500: return "Internal Server Error"
    case 503: return "Service Unavailable"
    default: return "Status"
    }
  }
}

public final class HTTPServer: @unchecked Sendable {
  public typealias Handler = @Sendable (HTTPRequest, ResponseWriter) async -> Void
  private let listenFD: Int32
  public let port: Int
  private let handler: Handler
  private var acceptThread: Thread?
  private let maxBody = 16 << 20

  public init(host: String = "127.0.0.1", port: Int, handler: @escaping Handler) throws {
    signal(SIGPIPE, SIG_IGN)
    #if canImport(Glibc)
      let fd = socket(AF_INET, Int32(SOCK_STREAM.rawValue), 0)
    #else
      let fd = socket(AF_INET, SOCK_STREAM, 0)
    #endif
    guard fd >= 0 else { throw HTTPServerError.socket("socket: \(String(cString: strerror(errno)))") }
    var one: Int32 = 1
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, socklen_t(MemoryLayout<Int32>.size))
    var addr = sockaddr_in()
    addr.sin_family = sa_family_t(AF_INET)
    addr.sin_port = in_port_t(UInt16(port).bigEndian)
    inet_pton(AF_INET, host, &addr.sin_addr)
    let bound = withUnsafePointer(to: &addr) {
      $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { bind(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size)) }
    }
    guard bound == 0, listen(fd, 128) == 0 else {
      close(fd)
      throw HTTPServerError.socket("bind/listen on \(host):\(port): \(String(cString: strerror(errno)))")
    }
    var actual = sockaddr_in()
    var len = socklen_t(MemoryLayout<sockaddr_in>.size)
    _ = withUnsafeMutablePointer(to: &actual) {
      $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { getsockname(fd, $0, &len) }
    }
    self.listenFD = fd
    self.port = Int(UInt16(bigEndian: actual.sin_port))
    self.handler = handler
  }

  public func start() {
    let t = Thread { [self] in acceptLoop() }
    t.name = "http-accept"
    acceptThread = t
    t.start()
  }

  public func stop() {
    shutdown(listenFD, Int32(SHUT_RDWR))
    close(listenFD)
  }

  private func acceptLoop() {
    while true {
      let c = accept(listenFD, nil, nil)
      if c < 0 {
        if errno == EINTR { continue }
        return  // listening socket closed
      }
      let t = Thread { [self] in serve(c) }
      t.start()
    }
  }

  private func serve(_ fd: Int32) {
    defer { close(fd) }
    let writer = ResponseWriter(fd: fd)
    guard let request = readRequest(fd) else {
      writer.respond(status: 400, contentType: "text/plain", body: Data("bad request\n".utf8))
      return
    }
    let done = DispatchSemaphore(value: 0)
    let handler = self.handler
    Task {
      await handler(request, writer)
      done.signal()
    }
    done.wait()
  }

  private func readRequest(_ fd: Int32) -> HTTPRequest? {
    var buf = [UInt8]()
    var chunk = [UInt8](repeating: 0, count: 65536)
    var headerEnd: Int? = nil
    while headerEnd == nil {
      let n = recv(fd, &chunk, chunk.count, 0)
      if n <= 0 { return nil }
      buf += chunk[0..<n]
      if let r = HTTPServer.find([13, 10, 13, 10], in: buf) { headerEnd = r }
      if buf.count > 64 << 10 && headerEnd == nil { return nil }
    }
    let head = String(decoding: buf[0..<headerEnd!], as: UTF8.self)
    var lines = head.components(separatedBy: "\r\n")
    let requestLine = lines.removeFirst().split(separator: " ")
    guard requestLine.count >= 2 else { return nil }
    var headers: [String: String] = [:]
    for l in lines {
      guard let colon = l.firstIndex(of: ":") else { continue }
      headers[l[..<colon].lowercased()] = l[l.index(after: colon)...].trimmingCharacters(in: .whitespaces)
    }
    var body = Array(buf[(headerEnd! + 4)...])
    let length = Int(headers["content-length"] ?? "0") ?? 0
    if length > maxBody { return nil }
    while body.count < length {
      let n = recv(fd, &chunk, min(chunk.count, length - body.count), 0)
      if n <= 0 { return nil }
      body += chunk[0..<n]
    }
    let path = String(requestLine[1].split(separator: "?", maxSplits: 1).first ?? "")
    return HTTPRequest(method: String(requestLine[0]), path: path, headers: headers, body: Data(body))
  }

  static func find(_ needle: [UInt8], in hay: [UInt8]) -> Int? {
    guard hay.count >= needle.count else { return nil }
    for i in 0...(hay.count - needle.count) where hay[i] == needle[0] {
      if Array(hay[i..<(i + needle.count)]) == needle { return i }
    }
    return nil
  }
}
