// Byte-stream connections between workers.
//
// Two kinds:
//   - TCP, for workers on different machines (and on one machine when convenient).
//     send() hands the kernel a list of slices in one sendmsg call, so KV blocks go
//     from the cache to the socket without being copied into a buffer first.
//   - A shared-memory ring, for workers on the same machine: one single-producer /
//     single-consumer byte ring per direction in a POSIX shared memory segment, with
//     acquire/release head and tail counters. No system call per message.
//
// Both are reliable, ordered streams; a closed or broken peer makes recv() throw
// ConnectionClosed, which is how the transfer engine detects a lost worker.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

namespace relay {

struct ConnectionClosed : std::runtime_error {
  using std::runtime_error::runtime_error;
};

struct Slice {
  const void* data;
  std::size_t len;
};

class Connection {
 public:
  virtual ~Connection() = default;
  // Sends every slice, in order, or throws.
  virtual void send(const Slice* slices, int n) = 0;
  void send(const void* data, std::size_t len) {
    Slice s{data, len};
    send(&s, 1);
  }
  // Receives exactly len bytes, or throws ConnectionClosed.
  virtual void recv(void* dst, std::size_t len) = 0;
  // Ends the connection for both sides; a recv() blocked in another thread returns
  // by throwing ConnectionClosed.
  virtual void close() = 0;
  virtual std::string describe() const = 0;
};

// ---- TCP ----

class TcpListener {
 public:
  // port 0 picks a free port.
  explicit TcpListener(int port = 0, const std::string& host = "127.0.0.1");
  ~TcpListener();
  TcpListener(const TcpListener&) = delete;
  TcpListener& operator=(const TcpListener&) = delete;
  int port() const { return port_; }
  std::unique_ptr<Connection> accept();

 private:
  int fd_ = -1;
  int port_ = 0;
};

std::unique_ptr<Connection> tcp_connect(const std::string& host, int port, double timeout_s = 10.0);

// ---- shared memory ----

// The creating side makes the segment (and removes its name once the peer has
// opened it); the other side opens it by name. capacity is per direction.
std::unique_ptr<Connection> shm_create(const std::string& name, std::size_t capacity = 64 << 20);
std::unique_ptr<Connection> shm_open(const std::string& name, double timeout_s = 10.0);

}  // namespace relay
