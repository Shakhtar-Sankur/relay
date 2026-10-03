"""Load generator for relay's OpenAI-compatible API (or any other: vLLM, etc.).

    python scripts/loadgen.py --url http://127.0.0.1:8000 --requests 64 --concurrency 8 \
        --prompt-words 200 --max-tokens 64 [--long-every 4 --long-words 1500] [--out run.json]

Each request streams a completion and records:
  ttft  time from sending the request to the first token
  itl   gaps between consecutive tokens (inter-token latency)
Prompts are random words, so the prefix cache only helps where --shared-prefix-words asks
it to. --long-every K makes every K-th prompt long: the mix where long prefills interfere
with running decodes on a colocated server, which is what disaggregation is meant to fix.
Standard library only.
"""

import argparse
import http.client
import json
import random
import statistics
import threading
import time
import urllib.parse

WORDS = ("system data model token cache layer prefill decode stream batch memory engine worker "
         "router latency request kernel tensor matrix vector block page queue network transfer "
         "cluster schedule replica shard node commit record value index search graph").split()


def percentile(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100
    f, c = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def one_request(url, prompt, max_tokens, seed, model):
    u = urllib.parse.urlparse(url)
    conn = http.client.HTTPConnection(u.hostname, u.port, timeout=600)
    body = json.dumps({"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0.7,
                       "seed": seed, "stream": True, "ignore_eos": True})
    t0 = time.perf_counter()
    conn.request("POST", "/v1/completions", body=body, headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    if resp.status != 200:
        return {"status": resp.status, "error": resp.read()[:200].decode(errors="replace")}
    times, buf = [], b""
    while True:
        chunk = resp.read1(65536) if hasattr(resp, "read1") else resp.read(1)
        if not chunk:
            break
        buf += chunk
        while b"\n\n" in buf:
            event, buf = buf.split(b"\n\n", 1)
            if not event.startswith(b"data: ") or event == b"data: [DONE]":
                continue
            d = json.loads(event[6:])
            if "error" in d:
                return {"status": 500, "error": d["error"]}
            text = d["choices"][0].get("text", "")
            if text:
                times.append(time.perf_counter())
    end = time.perf_counter()
    conn.close()
    if not times:
        return {"status": 200, "ttft": end - t0, "itl": [], "tokens": 0, "total": end - t0}
    return {"status": 200, "ttft": times[0] - t0, "itl": [b - a for a, b in zip(times, times[1:])],
            "tokens": len(times), "total": end - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="")
    ap.add_argument("--requests", type=int, default=32)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--prompt-words", type=int, default=100)
    ap.add_argument("--shared-prefix-words", type=int, default=0)
    ap.add_argument("--long-every", type=int, default=0)
    ap.add_argument("--long-words", type=int, default=1000)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label", default="")
    ap.add_argument("--out")
    a = ap.parse_args()

    rng = random.Random(a.seed)
    shared = " ".join(rng.choice(WORDS) for _ in range(a.shared_prefix_words))
    prompts = []
    for i in range(a.requests):
        n = a.long_words if a.long_every and i % a.long_every == a.long_every - 1 else a.prompt_words
        prompts.append((shared + " " if shared else "") + " ".join(rng.choice(WORDS) for _ in range(n)))

    results, lock, next_i = [None] * a.requests, threading.Lock(), [0]

    def worker():
        while True:
            with lock:
                i = next_i[0]
                next_i[0] += 1
            if i >= a.requests:
                return
            results[i] = one_request(a.url, prompts[i], a.max_tokens, a.seed * 100000 + i, a.model)
            results[i]["long"] = bool(a.long_every and i % a.long_every == a.long_every - 1)

    t0 = time.perf_counter()
    threads = [threading.Thread(target=worker) for _ in range(a.concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0

    ok = [r for r in results if r["status"] == 200]
    ttft = [r["ttft"] for r in ok]
    ttft_short = [r["ttft"] for r in ok if not r["long"]]
    itl = [x for r in ok for x in r["itl"]]
    tokens = sum(r["tokens"] for r in ok)
    summary = {
        "label": a.label, "url": a.url, "requests": a.requests, "concurrency": a.concurrency,
        "prompt_words": a.prompt_words, "long_every": a.long_every, "long_words": a.long_words,
        "max_tokens": a.max_tokens, "ok": len(ok), "errors": len(results) - len(ok),
        "wall_s": wall, "output_tokens_per_s": tokens / wall if wall else 0,
        "ttft_ms": {p: (percentile(ttft, p) or 0) * 1e3 for p in (50, 90, 99)},
        "ttft_short_prompts_ms": {p: (percentile(ttft_short, p) or 0) * 1e3 for p in (50, 90, 99)},
        "itl_ms": {p: (percentile(itl, p) or 0) * 1e3 for p in (50, 90, 99)},
        "itl_mean_ms": statistics.mean(itl) * 1e3 if itl else 0,
    }
    print(json.dumps(summary, indent=2))
    if a.out:
        with open(a.out, "a") as f:
            f.write(json.dumps(summary) + "\n")


if __name__ == "__main__":
    main()
