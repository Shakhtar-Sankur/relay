#include "relay/transport.h"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/uio.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <thread>
#include <vector>

namespace relay {
namespace {

std::string errno_text(const char* what) { return std::string(what) + ": " + std::strerror(errno); }

// ---- TCP ----

class TcpConnection final : public Connection {
 public:
  TcpConnection(int fd, std::string peer) : fd_(fd), peer_(std::move(peer)) {
    int one = 1;
    ::setsockopt(fd_, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
    int buf = 8 << 20;
    ::setsockopt(fd_, SOL_SOCKET, SO_SNDBUF, &buf, sizeof buf);
    ::setsockopt(fd_, SOL_SOCKET, SO_RCVBUF, &buf, sizeof buf);
  }
  ~TcpConnection() override {
    if (fd_ >= 0) ::close(fd_);
  }

  void send(const Slice* slices, int n) override {
    std::vector<iovec> iov;
    iov.reserve(n);
    for (int i = 0; i < n; ++i)
      if (slices[i].len) iov.push_back({const_cast<void*>(slices[i].data), slices[i].len});
    std::size_t first = 0;
    while (first < iov.size()) {
      msghdr msg{};
      msg.msg_iov = &iov[first];
      msg.msg_iovlen = std::min<std::size_t>(iov.size() - first, IOV_MAX);
      ssize_t w = ::sendmsg(fd_, &msg, MSG_NOSIGNAL);
      if (w < 0) {
        if (errno == EINTR) continue;
        throw ConnectionClosed(errno_text("tcp send"));
      }
      // Skip what was written; a partly written slice is advanced in place.
      auto left = static_cast<std::size_t>(w);
      while (left > 0 && first < iov.size()) {
        if (left >= iov[first].iov_len) {
          left -= iov[first].iov_len;
          ++first;
        } else {
          iov[first].iov_base = static_cast<char*>(iov[first].iov_base) + left;
          iov[first].iov_len -= left;
          left = 0;
        }
      }
    }
  }

  void recv(void* dst, std::size_t len) override {
    auto* p = static_cast<char*>(dst);
    while (len > 0) {
      ssize_t r = ::recv(fd_, p, len, 0);
      if (r == 0) throw ConnectionClosed("tcp: peer closed the connection (" + peer_ + ")");
      if (r < 0) {
        if (errno == EINTR) continue;
        throw ConnectionClosed(errno_text("tcp recv"));
      }
      p += r;
      len -= static_cast<std::size_t>(r);
    }
  }

  void close() override { ::shutdown(fd_, SHUT_RDWR); }
  std::string describe() const override { return "tcp " + peer_; }

 private:
  int fd_;
  std::string peer_;
};

// ---- shared memory ----

// One direction: a byte ring. `head` counts bytes ever written, `tail` bytes ever read;
// both only grow, so head - tail is the amount buffered and index = counter % capacity.
struct RingHeader {
  std::uint64_t magic;
  std::uint64_t capacity;
  alignas(64) std::atomic<std::uint64_t> head;
  alignas(64) std::atomic<std::uint64_t> tail;
  alignas(64) std::atomic<std::uint32_t> closed;
  std::atomic<std::uint32_t> opened;  // the peer has mapped the segment
};
static_assert(std::atomic<std::uint64_t>::is_always_lock_free, "the ring needs lock-free 64-bit atomics");

constexpr std::uint64_t kRingMagic = 0x72656c6179726e67ULL;  // "relayrng"

struct Segment {
  void* base = nullptr;
  std::size_t size = 0;
  RingHeader* ring[2] = {nullptr, nullptr};
  unsigned char* data[2] = {nullptr, nullptr};
};

std::size_t header_bytes() { return (sizeof(RingHeader) + 63) / 64 * 64; }

// Waits for cond(): spin briefly, then yield, then sleep; false on timeout or close.
template <typename Cond>
bool wait_for(Cond cond, const std::atomic<std::uint32_t>& closed, double timeout_s) {
  auto start = std::chrono::steady_clock::now();
  for (int i = 0;; ++i) {
    if (cond()) return true;
    if (closed.load(std::memory_order_acquire)) return cond();
    if (i < 128) continue;
    if (i < 1024) {
      std::this_thread::yield();
      continue;
    }
    std::this_thread::sleep_for(std::chrono::microseconds(50));
    if (timeout_s > 0 && std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() > timeout_s)
      return false;
  }
}

class ShmConnection final : public Connection {
 public:
  // side 0 writes ring 0 and reads ring 1; side 1 the opposite.
  ShmConnection(Segment seg, int side, std::string name) : seg_(seg), side_(side), name_(std::move(name)) {}
  ~ShmConnection() override {
    close();
    ::munmap(seg_.base, seg_.size);
    if (side_ == 0) ::shm_unlink(name_.c_str());  // in case the peer never opened it
  }

  void send(const Slice* slices, int n) override {
    RingHeader* r = seg_.ring[side_];
    unsigned char* data = seg_.data[side_];
    const std::uint64_t cap = r->capacity;
    for (int i = 0; i < n; ++i) {
      const auto* p = static_cast<const unsigned char*>(slices[i].data);
      std::size_t left = slices[i].len;
      while (left > 0) {
        std::uint64_t head = r->head.load(std::memory_order_relaxed);
        std::uint64_t space = 0;
        bool ok = wait_for(
            [&] {
              space = cap - (head - r->tail.load(std::memory_order_acquire));
              return space > 0;
            },
            r->closed, 0);
        if (!ok || r->closed.load(std::memory_order_acquire)) throw ConnectionClosed("shm: connection closed (" + name_ + ")");
        std::size_t n_now = static_cast<std::size_t>(std::min<std::uint64_t>(space, left));
        std::size_t at = static_cast<std::size_t>(head % cap);
        std::size_t first = std::min(n_now, static_cast<std::size_t>(cap - at));
        std::memcpy(data + at, p, first);
        std::memcpy(data, p + first, n_now - first);
        r->head.store(head + n_now, std::memory_order_release);
        p += n_now;
        left -= n_now;
      }
    }
  }

  void recv(void* dst, std::size_t len) override {
    RingHeader* r = seg_.ring[1 - side_];
    unsigned char* data = seg_.data[1 - side_];
    const std::uint64_t cap = r->capacity;
    auto* p = static_cast<unsigned char*>(dst);
    while (len > 0) {
      std::uint64_t tail = r->tail.load(std::memory_order_relaxed);
      std::uint64_t avail = 0;
      wait_for(
          [&] {
            avail = r->head.load(std::memory_order_acquire) - tail;
            return avail > 0;
          },
          r->closed, 0);
      if (avail == 0) throw ConnectionClosed("shm: peer closed the connection (" + name_ + ")");
      std::size_t n_now = static_cast<std::size_t>(std::min<std::uint64_t>(avail, len));
      std::size_t at = static_cast<std::size_t>(tail % cap);
      std::size_t first = std::min(n_now, static_cast<std::size_t>(cap - at));
      std::memcpy(p, data + at, first);
      std::memcpy(p + first, data, n_now - first);
      r->tail.store(tail + n_now, std::memory_order_release);
      p += n_now;
      len -= n_now;
    }
  }

  void close() override {
    seg_.ring[0]->closed.store(1, std::memory_order_release);
    seg_.ring[1]->closed.store(1, std::memory_order_release);
  }

  std::string describe() const override { return "shm " + name_; }

 private:
  Segment seg_;
  int side_;
  std::string name_;
};

Segment map_segment(int fd, std::size_t size) {
  Segment s;
  s.size = size;
  s.base = ::mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (s.base == MAP_FAILED) throw std::runtime_error(errno_text("shm mmap"));
  auto* base = static_cast<unsigned char*>(s.base);
  std::size_t cap = (size - 2 * header_bytes()) / 2;
  for (int i = 0; i < 2; ++i) {
    unsigned char* h = base + i * (header_bytes() + cap);
    s.ring[i] = reinterpret_cast<RingHeader*>(h);
    s.data[i] = h + header_bytes();
  }
  return s;
}

std::string shm_path(const std::string& name) { return name.empty() || name[0] != '/' ? "/" + name : name; }

}  // namespace

TcpListener::TcpListener(int port, const std::string& host) {
  fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd_ < 0) throw std::runtime_error(errno_text("socket"));
  int one = 1;
  ::setsockopt(fd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
  sockaddr_in a{};
  a.sin_family = AF_INET;
  a.sin_port = htons(static_cast<std::uint16_t>(port));
  if (::inet_pton(AF_INET, host.c_str(), &a.sin_addr) != 1) throw std::runtime_error("bad listen address " + host);
  if (::bind(fd_, reinterpret_cast<sockaddr*>(&a), sizeof a) != 0) throw std::runtime_error(errno_text("bind"));
  if (::listen(fd_, 64) != 0) throw std::runtime_error(errno_text("listen"));
  socklen_t len = sizeof a;
  ::getsockname(fd_, reinterpret_cast<sockaddr*>(&a), &len);
  port_ = ntohs(a.sin_port);
}

TcpListener::~TcpListener() {
  if (fd_ >= 0) ::close(fd_);
}

std::unique_ptr<Connection> TcpListener::accept() {
  sockaddr_in a{};
  socklen_t len = sizeof a;
  int fd;
  do {
    fd = ::accept(fd_, reinterpret_cast<sockaddr*>(&a), &len);
  } while (fd < 0 && errno == EINTR);
  if (fd < 0) throw std::runtime_error(errno_text("accept"));
  char ip[64];
  ::inet_ntop(AF_INET, &a.sin_addr, ip, sizeof ip);
  return std::make_unique<TcpConnection>(fd, std::string(ip) + ":" + std::to_string(ntohs(a.sin_port)));
}

std::unique_ptr<Connection> tcp_connect(const std::string& host, int port, double timeout_s) {
  addrinfo hints{}, *res = nullptr;
  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_STREAM;
  if (::getaddrinfo(host.c_str(), std::to_string(port).c_str(), &hints, &res) != 0 || !res)
    throw std::runtime_error("cannot resolve " + host);
  auto deadline = std::chrono::steady_clock::now() + std::chrono::duration<double>(timeout_s);
  while (true) {
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
      ::freeaddrinfo(res);
      throw std::runtime_error(errno_text("socket"));
    }
    if (::connect(fd, res->ai_addr, res->ai_addrlen) == 0) {
      ::freeaddrinfo(res);
      return std::make_unique<TcpConnection>(fd, host + ":" + std::to_string(port));
    }
    ::close(fd);
    if (std::chrono::steady_clock::now() > deadline) {
      ::freeaddrinfo(res);
      throw std::runtime_error("cannot connect to " + host + ":" + std::to_string(port));
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));  // the listener may not be up yet
  }
}

std::unique_ptr<Connection> shm_create(const std::string& name, std::size_t capacity) {
  std::string path = shm_path(name);
  ::shm_unlink(path.c_str());
  int fd = ::shm_open(path.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
  if (fd < 0) throw std::runtime_error(errno_text("shm_open create"));
  std::size_t size = 2 * (header_bytes() + capacity);
  if (::ftruncate(fd, static_cast<off_t>(size)) != 0) {
    ::close(fd);
    ::shm_unlink(path.c_str());
    throw std::runtime_error(errno_text("ftruncate"));
  }
  Segment seg = map_segment(fd, size);
  ::close(fd);
  for (auto* r : seg.ring) {
    r->capacity = capacity;
    r->head.store(0);
    r->tail.store(0);
    r->closed.store(0);
    r->opened.store(0);
    std::atomic_thread_fence(std::memory_order_release);
    r->magic = kRingMagic;
  }
  return std::make_unique<ShmConnection>(seg, 0, path);
}

std::unique_ptr<Connection> shm_open(const std::string& name, double timeout_s) {
  std::string path = shm_path(name);
  auto deadline = std::chrono::steady_clock::now() + std::chrono::duration<double>(timeout_s);
  int fd;
  while ((fd = ::shm_open(path.c_str(), O_RDWR, 0600)) < 0) {
    if (std::chrono::steady_clock::now() > deadline) throw std::runtime_error(errno_text("shm_open"));
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  struct stat st;
  ::fstat(fd, &st);
  Segment seg = map_segment(fd, static_cast<std::size_t>(st.st_size));
  ::close(fd);
  while (seg.ring[1]->magic != kRingMagic) {
    if (std::chrono::steady_clock::now() > deadline) throw std::runtime_error("shm: segment never initialized");
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  // Both sides have it mapped now; the name is no longer needed.
  ::shm_unlink(path.c_str());
  seg.ring[0]->opened.store(1, std::memory_order_release);
  return std::make_unique<ShmConnection>(seg, 1, path);
}

}  // namespace relay
