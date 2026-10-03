#include "relay/kv_transfer.h"

#include <algorithm>
#include <chrono>
#include <memory>
#include <cstring>
#include <stdexcept>
#include <string>

#include "relay/sampler.h"

namespace relay {

KVFingerprint KVFingerprint::of(const Backend& b) {
  const KVLayout& l = b.kv_layout();
  return {l.layers, l.kv_heads, l.head_dim, l.block_size, static_cast<std::int32_t>(b.kv_block_bytes()), b.config().vocab};
}

namespace {

FrameHeader header(FrameType t, std::uint64_t request, std::uint32_t a, std::uint32_t b, std::uint64_t payload) {
  return FrameHeader{kFrameMagic, static_cast<std::uint16_t>(t), 0, request, a, b, payload};
}

}  // namespace

// ---- sender -----------------------------------------------------------------

KVSender::KVSender(Backend& backend, Connection& conn, std::function<void(std::uint64_t, int)> on_layer_sent)
    : on_layer_sent(std::move(on_layer_sent)), be_(backend), conn_(conn) {
  KVFingerprint fp = KVFingerprint::of(be_);
  Job hello{FrameType::Hello};
  hello.payload.resize(sizeof fp);
  std::memcpy(hello.payload.data(), &fp, sizeof fp);
  queue_.push_back(std::move(hello));
  thread_ = std::thread([this] { run(); });
}

KVSender::~KVSender() {
  {
    std::lock_guard<std::mutex> lock(mu_);
    stop_ = true;
  }
  cv_.notify_all();
  thread_.join();
}

void KVSender::push(Job j) {
  {
    std::lock_guard<std::mutex> lock(mu_);
    if (!error_) {
      queue_.push_back(std::move(j));
      cv_.notify_one();
      return;
    }
  }
  // The connection is gone: the frame is dropped, and its owner hears so at once.
  if (j.on_sent) j.on_sent(false);
}

bool KVSender::failed() const {
  std::lock_guard<std::mutex> lock(mu_);
  return error_ != nullptr;
}

void KVSender::begin(const Request& r, int num_blocks) {
  BeginInfo info{static_cast<std::uint32_t>(r.prompt.size()), static_cast<std::uint32_t>(num_blocks),
                 r.params.max_new_tokens, r.params.temperature, r.params.top_p, r.params.top_k, r.params.seed,
                 r.params.ignore_eos ? 1u : 0u, 0};
  Job j{FrameType::Begin, r.id};
  j.payload.resize(sizeof info + r.prompt.size() * sizeof(std::int32_t));
  std::memcpy(j.payload.data(), &info, sizeof info);
  std::memcpy(j.payload.data() + sizeof info, r.prompt.data(), r.prompt.size() * sizeof(std::int32_t));
  push(std::move(j));
}

void KVSender::layer(std::uint64_t request, int layer, int first_index, std::vector<int> blocks) {
  Job j{FrameType::Layer, request, layer, first_index};
  j.blocks = std::move(blocks);
  push(std::move(j));
}

void KVSender::end(std::uint64_t request, int first_token, bool finished, std::function<void(bool)> on_sent) {
  Job j{FrameType::End, request, first_token, finished ? 1 : 0};
  j.on_sent = std::move(on_sent);
  push(std::move(j));
}

void KVSender::abort(std::uint64_t request, std::function<void(bool)> on_sent) {
  Job j{FrameType::Abort, request};
  j.on_sent = std::move(on_sent);
  push(std::move(j));
}

void KVSender::flush() {
  std::unique_lock<std::mutex> lock(mu_);
  // After a failure, busy_ stays set until every dropped frame's callback has run.
  idle_cv_.wait(lock, [this] { return !busy_ && (queue_.empty() || error_); });
  if (error_) std::rethrow_exception(error_);
}

TransferStats KVSender::stats() const {
  std::lock_guard<std::mutex> lock(mu_);
  return stats_;
}

void KVSender::send_layer(Job& j) {
  const std::size_t bb = be_.kv_block_bytes(), n = j.blocks.size();
  FrameHeader h = header(FrameType::Layer, j.request, j.a, j.b, 2 * n * bb);
  be_.wait_kv_written(j.a);
  if (be_.kv_host_ptr(j.a, j.blocks[0], false)) {
    // The cache is in host memory: point the socket straight at the blocks.
    std::vector<Slice> s;
    s.reserve(1 + 2 * n);
    s.push_back({&h, sizeof h});
    for (int b : j.blocks) s.push_back({be_.kv_host_ptr(j.a, b, false), bb});
    for (int b : j.blocks) s.push_back({be_.kv_host_ptr(j.a, b, true), bb});
    conn_.send(s.data(), static_cast<int>(s.size()));
  } else {
    // GPU cache: gather the layer's blocks into one host buffer, then send.
    if (staging_.size() < 2 * n * bb) staging_.resize(2 * n * bb);
    be_.read_kv_layer(j.a, j.blocks, staging_.data());
    Slice s[2] = {{&h, sizeof h}, {staging_.data(), 2 * n * bb}};
    conn_.send(s, 2);
  }
  std::lock_guard<std::mutex> lock(mu_);
  stats_.kv_bytes += 2 * n * bb;
}

void KVSender::run() {
  while (true) {
    Job j;
    {
      std::unique_lock<std::mutex> lock(mu_);
      cv_.wait(lock, [this] { return stop_ || !queue_.empty(); });
      if (queue_.empty()) return;  // stop_ and nothing left
      j = std::move(queue_.front());
      queue_.pop_front();
      busy_ = true;
    }
    try {
      if (j.type == FrameType::Layer) {
        send_layer(j);
        if (on_layer_sent) on_layer_sent(j.request, j.a);
      } else {
        FrameHeader h = header(j.type, j.request, static_cast<std::uint32_t>(j.a), static_cast<std::uint32_t>(j.b),
                               j.payload.size());
        Slice s[2] = {{&h, sizeof h}, {j.payload.data(), j.payload.size()}};
        conn_.send(s, 2);
      }
      if (j.on_sent) j.on_sent(true);
      std::lock_guard<std::mutex> lock(mu_);
      ++stats_.frames;
      if (j.type == FrameType::End) ++stats_.requests;
    } catch (...) {
      // Everything still queued is dropped; its owners learn which requests did not arrive
      // (and get their blocks back). The callbacks run without the lock: they take the
      // prefill worker's.
      std::deque<Job> dropped;
      {
        std::lock_guard<std::mutex> lock(mu_);
        error_ = std::current_exception();
        dropped.swap(queue_);
      }
      if (j.on_sent) j.on_sent(false);
      for (auto& q : dropped)
        if (q.on_sent) q.on_sent(false);
      std::lock_guard<std::mutex> lock(mu_);
      busy_ = false;
      idle_cv_.notify_all();
      return;
    }
    std::lock_guard<std::mutex> lock(mu_);
    busy_ = false;
    if (queue_.empty()) idle_cv_.notify_all();
  }
}

// ---- receiver ---------------------------------------------------------------

KVReceiver::KVReceiver(Engine& engine, Backend& backend, Connection& conn,
                       std::function<void(std::uint64_t, int)> on_ready)
    : on_ready(std::move(on_ready)), engine_(engine), be_(backend), conn_(conn) {
  thread_ = std::thread([this] { run(); });
}

KVReceiver::~KVReceiver() {
  conn_.close();
  if (thread_.joinable()) thread_.join();
}

void KVReceiver::join() {
  if (thread_.joinable()) thread_.join();
}

TransferStats KVReceiver::stats() const {
  std::lock_guard<std::mutex> lock(mu_);
  return stats_;
}

std::exception_ptr KVReceiver::error() const {
  std::lock_guard<std::mutex> lock(mu_);
  return error_;
}

int KVReceiver::in_flight() const {
  std::lock_guard<std::mutex> lock(mu_);
  return static_cast<int>(incoming_.size());
}

void KVReceiver::run() {
  bool hello = false;
  try {
    while (true) {
      FrameHeader h;
      conn_.recv(&h, sizeof h);
      if (h.magic != kFrameMagic) throw std::runtime_error("kv transfer: bad frame magic");
      if (!hello && h.type != static_cast<std::uint16_t>(FrameType::Hello))
        throw std::runtime_error("kv transfer: the first frame must be Hello");
      if (h.type == static_cast<std::uint16_t>(FrameType::Hello)) {
        KVFingerprint theirs, ours = KVFingerprint::of(be_);
        if (h.payload != sizeof theirs) throw std::runtime_error("kv transfer: bad Hello");
        conn_.recv(&theirs, sizeof theirs);
        if (!(theirs == ours))
          throw std::runtime_error("kv transfer: the sender's model or KV cache layout differs from this worker's");
        hello = true;
        continue;
      }
      handle(h);
      std::lock_guard<std::mutex> lock(mu_);
      ++stats_.frames;
    }
  } catch (const ConnectionClosed&) {
    // The peer went away (or we closed): fall through to cleanup.
  } catch (...) {
    std::lock_guard<std::mutex> lock(mu_);
    error_ = std::current_exception();
  }
  // Requests whose End never arrived give their blocks back.
  std::lock_guard<std::mutex> lock(mu_);
  for (auto& [id, in] : incoming_)
    if (!in.blocks.empty()) engine_.release_blocks(in.blocks);
  incoming_.clear();
}

void KVReceiver::handle(const FrameHeader& h) {
  switch (static_cast<FrameType>(h.type)) {
    case FrameType::Begin: {
      BeginInfo info;
      if (h.payload < sizeof info) throw std::runtime_error("kv transfer: bad Begin");
      conn_.recv(&info, sizeof info);
      if (h.payload != sizeof info + info.prompt_len * sizeof(std::int32_t))
        throw std::runtime_error("kv transfer: Begin payload size does not match its prompt length");
      Incoming in;
      in.req.id = h.request;
      in.req.prompt.resize(info.prompt_len);
      conn_.recv(in.req.prompt.data(), info.prompt_len * sizeof(std::int32_t));
      in.req.params.max_new_tokens = info.max_new_tokens;
      in.req.params.temperature = info.temperature;
      in.req.params.top_p = info.top_p;
      in.req.params.top_k = info.top_k;
      in.req.params.seed = info.seed;
      in.req.params.ignore_eos = info.ignore_eos != 0;
      if (!engine_.try_reserve_for_transfer(static_cast<int>(info.num_blocks), in.blocks)) in.blocks.clear();
      std::lock_guard<std::mutex> lock(mu_);
      incoming_[h.request] = std::move(in);
      break;
    }
    case FrameType::Layer: {
      std::vector<int> dst;
      {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = incoming_.find(h.request);
        if (it == incoming_.end()) throw std::runtime_error("kv transfer: Layer for an unknown request");
        const std::size_t bb = be_.kv_block_bytes();
        if (h.payload % (2 * bb) != 0) throw std::runtime_error("kv transfer: Layer payload is not whole blocks");
        std::size_t n = h.payload / (2 * bb);
        if (!it->second.blocks.empty()) {
          if (h.b + n > it->second.blocks.size()) throw std::runtime_error("kv transfer: Layer beyond the request's blocks");
          dst.assign(it->second.blocks.begin() + h.b, it->second.blocks.begin() + h.b + n);
        }
      }
      const std::size_t bb = be_.kv_block_bytes();
      if (dst.empty()) {  // no room for this request: read the bytes and drop them
        if (staging_.size() < h.payload) staging_.resize(h.payload);
        conn_.recv(staging_.data(), h.payload);
        break;
      }
      if (be_.kv_host_ptr(static_cast<int>(h.a), dst[0], false)) {
        // Host cache: read each block straight into place.
        for (int b : dst) conn_.recv(be_.kv_host_ptr(static_cast<int>(h.a), b, false), bb);
        for (int b : dst) conn_.recv(be_.kv_host_ptr(static_cast<int>(h.a), b, true), bb);
      } else {
        if (staging_.size() < h.payload) staging_.resize(h.payload);
        conn_.recv(staging_.data(), h.payload);
        be_.write_kv_layer(static_cast<int>(h.a), dst, staging_.data());
      }
      std::lock_guard<std::mutex> lock(mu_);
      stats_.kv_bytes += h.payload;
      break;
    }
    case FrameType::End: {
      Incoming in;
      {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = incoming_.find(h.request);
        if (it == incoming_.end()) throw std::runtime_error("kv transfer: End for an unknown request");
        in = std::move(it->second);
        incoming_.erase(it);
        ++stats_.requests;
      }
      int first_token = static_cast<int>(h.a);
      if (h.b) {
        engine_.release_blocks(in.blocks);  // finished at its first token: nothing to decode
      } else if (in.blocks.empty()) {
        engine_.add_recompute(std::move(in.req), first_token);
        std::lock_guard<std::mutex> lock(mu_);
        ++stats_.recomputed;
      } else {
        engine_.add_prefilled(std::move(in.req), std::move(in.blocks), first_token);
      }
      if (on_ready) on_ready(h.request, first_token);
      break;
    }
    case FrameType::Abort: {
      std::lock_guard<std::mutex> lock(mu_);
      auto it = incoming_.find(h.request);
      if (it != incoming_.end()) {
        if (!it->second.blocks.empty()) engine_.release_blocks(it->second.blocks);
        incoming_.erase(it);
      }
      break;
    }
    default:
      throw std::runtime_error("kv transfer: unknown frame type " + std::to_string(h.type));
  }
}

// ---- prefill worker ---------------------------------------------------------

PrefillWorker::PrefillWorker(Backend& backend, KVSender* sender, int max_batch_tokens, bool prefix_caching)
    : be_(backend),
      sender_(sender),
      budget_(max_batch_tokens),
      alloc_(backend.kv_layout().num_blocks, backend.kv_layout().block_size, prefix_caching) {}

PrefillStats PrefillWorker::stats() const {
  std::lock_guard<std::mutex> lock(alloc_mu_);
  return stats_;
}

void PrefillWorker::add(Request r, KVSender* to) {
  if (r.prompt.empty()) throw std::invalid_argument("empty prompt");
  if (be_.kv_layout().blocks_for(static_cast<int>(r.prompt.size())) > be_.kv_layout().num_blocks)
    throw std::invalid_argument("prompt needs more KV blocks than the cache has");
  Pending p;
  p.req = std::move(r);
  p.sender = to ? to : sender_;
  if (!p.sender) throw std::invalid_argument("prefill: no destination for the request's KV cache");
  queue_.push_back(std::move(p));
}

bool PrefillWorker::cancel(std::uint64_t id) {
  for (auto it = queue_.begin(); it != queue_.end(); ++it) {
    if (it->req.id != id || it->done) continue;
    // Queued Layer frames may still read the blocks: free them once the Abort is sent.
    auto blocks = std::make_shared<std::vector<int>>(std::move(it->blocks));
    if (it->begun) {
      it->sender->abort(id, [this, blocks](bool) {
        std::lock_guard<std::mutex> lock(alloc_mu_);
        alloc_.release_all(*blocks);
      });
    } else {
      std::lock_guard<std::mutex> lock(alloc_mu_);
      alloc_.release_all(*blocks);
    }
    queue_.erase(it);
    return true;
  }
  return false;
}

void PrefillWorker::cancel_all() {
  while (!queue_.empty()) cancel(queue_.front().req.id);
}

void PrefillWorker::drop_sender(const KVSender* sender) {
  for (auto it = queue_.begin(); it != queue_.end();) {
    if (it->sender != sender) {
      ++it;
      continue;
    }
    std::lock_guard<std::mutex> lock(alloc_mu_);
    alloc_.release_all(it->blocks);
    failed_.push_back(it->req.id);
    it = queue_.erase(it);
  }
}

std::vector<std::uint64_t> PrefillWorker::take_failed() {
  std::lock_guard<std::mutex> lock(alloc_mu_);
  std::vector<std::uint64_t> out;
  out.swap(failed_);
  return out;
}

bool PrefillWorker::has_work() const { return !queue_.empty(); }

int PrefillWorker::free_blocks() const {
  std::lock_guard<std::mutex> lock(alloc_mu_);
  return alloc_.num_available();
}

std::vector<TokenEvent> PrefillWorker::step() {
  const KVLayout& kv = be_.kv_layout();
  // Requests headed to a decode worker whose connection failed cannot be delivered: drop
  // them (the sender's thread has stopped, so nothing reads their blocks any more).
  for (auto it = queue_.begin(); it != queue_.end();) {
    if (it->done || !it->sender->failed()) {
      ++it;
      continue;
    }
    std::lock_guard<std::mutex> lock(alloc_mu_);
    alloc_.release_all(it->blocks);
    failed_.push_back(it->req.id);
    it = queue_.erase(it);
  }
  ForwardBatch batch;
  struct Member {
    Pending* p;
    int first_index, n_blocks;
  };
  std::vector<Member> members;
  int budget = budget_;
  // Prompts in arrival order; the one at the front may be part-way through.
  for (Pending& p : queue_) {
    if (budget == 0) break;
    if (p.done) continue;
    if (p.blocks.empty()) {
      std::lock_guard<std::mutex> lock(alloc_mu_);
      const int len = static_cast<int>(p.req.prompt.size());
      // Cached prefix blocks first (at least one token is left to compute), then fresh
      // blocks for the rest of the prompt.
      p.computed = alloc_.match_prefix(p.req.prompt, len - 1, p.blocks, p.hashes);
      if (!alloc_.allocate(kv.blocks_for(len) - static_cast<int>(p.blocks.size()), p.blocks)) {
        alloc_.release_all(p.blocks);
        p.hashes.clear();
        p.computed = 0;
        break;
      }
      stats_.prompt_tokens += static_cast<std::uint64_t>(len);
      stats_.prefix_hit_tokens += static_cast<std::uint64_t>(p.computed);
    }
    if (!p.begun) {
      p.sender->begin(p.req, static_cast<int>(p.blocks.size()));
      p.begun = true;
      // Blocks that came from the prefix cache are complete already: send them now.
      int cached = p.computed / kv.block_size;
      if (cached > 0)
        for (int l = 0; l < kv.layers; ++l)
          p.sender->layer(p.req.id, l, 0, std::vector<int>(p.blocks.begin(), p.blocks.begin() + cached));
    }
    int n = std::min(budget, static_cast<int>(p.req.prompt.size()) - p.computed);
    std::vector<int> chunk(p.req.prompt.begin() + p.computed, p.req.prompt.begin() + p.computed + n);
    bool last = p.computed + n == static_cast<int>(p.req.prompt.size());
    batch.seqs.push_back({std::move(chunk), p.computed, p.blocks, last});
    int first_block = p.computed / kv.block_size, end_block = kv.blocks_for(p.computed + n);
    members.push_back({&p, first_block, end_block - first_block});
    budget -= n;
  }
  if (batch.seqs.empty()) return {};

  // Stream each layer's new blocks as soon as the backend reports them written.
  struct Streamer : LayerObserver {
    std::vector<Member>* members;
    void kv_written(int layer) override {
      for (const Member& m : *members) {
        std::vector<int> blocks(m.p->blocks.begin() + m.first_index, m.p->blocks.begin() + m.first_index + m.n_blocks);
        m.p->sender->layer(m.p->req.id, layer, m.first_index, std::move(blocks));
      }
    }
  } streamer;
  streamer.members = &members;
  std::vector<float> logits = be_.forward(batch, stream_layers ? &streamer : nullptr);
  if (!stream_layers)
    for (int l = 0; l < be_.config().layers; ++l) streamer.kv_written(l);

  std::vector<TokenEvent> events;
  const int V = be_.config().vocab;
  const auto& eos = be_.config().eos_ids;
  int row = 0;
  for (std::size_t i = 0; i < members.size(); ++i) {
    Pending* p = members[i].p;
    p->computed += static_cast<int>(batch.seqs[i].tokens.size());
    {
      std::lock_guard<std::mutex> lock(alloc_mu_);
      stats_.forward_tokens += batch.seqs[i].tokens.size();
      alloc_.register_full(p->req.prompt, p->computed, p->blocks, p->hashes);
    }
    if (!batch.seqs[i].want_logits) continue;
    const float* lg = logits.data() + static_cast<std::size_t>(row++) * V;
    if (on_logits) on_logits(p->req.id, lg);
    int tok = sample(lg, V, p->req.params, 0);
    bool stop = !p->req.params.ignore_eos && std::find(eos.begin(), eos.end(), tok) != eos.end();
    bool finished = stop || p->req.params.max_new_tokens <= 1;
    events.push_back({p->req.id, tok, 0, stop ? Finish::Stop : finished ? Finish::Length : Finish::None});
    // The blocks go back to the pool only once the sender has sent every frame that
    // reads them.
    auto blocks = std::make_shared<std::vector<int>>(std::move(p->blocks));
    const std::uint64_t id = p->req.id;
    p->sender->end(id, tok, finished, [this, blocks, id, finished](bool sent) {
      std::lock_guard<std::mutex> lock(alloc_mu_);
      alloc_.release_all(*blocks);
      // A request that ended at its first token needs nothing from the decode worker.
      if (!sent && !finished) failed_.push_back(id);
    });
    p->done = true;
  }
  for (auto it = queue_.begin(); it != queue_.end();) it = it->done ? queue_.erase(it) : it + 1;
  return events;
}

}  // namespace relay
