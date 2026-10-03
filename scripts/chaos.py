"""Chaos run: a real relay cluster serving a real model while its workers are killed.

    python scripts/chaos.py --model models/smollm2-135m [--prefill 2 --decode 2] \
        [--requests 64 --concurrency 8 --max-tokens 48 --kills 10 --seed 0]

Starts relay-worker processes and the Swift API server in front of them, then sends the
same requests twice through the OpenAI API (streaming, sampled with a fixed seed each):
  1. with no faults, to get each request's output;
  2. while a chaos thread SIGKILLs a random worker every so often and restarts it in its
     place (always leaving one worker of each role up).
Every request of run 2 must produce exactly the text of run 1. Prints a summary and exits
non-zero if any request differs or fails. Standard library only.
"""

import argparse
import http.client
import json
import os
import random
import signal
import socket
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class Worker:
    def __init__(self, binary, model, role, blocks, threads, control_port=0):
        self.binary, self.model, self.role, self.blocks, self.threads = binary, model, role, blocks, threads
        # The workers share the machine's cores instead of each spinning a thread per core.
        env = dict(os.environ, OMP_NUM_THREADS=str(threads))
        log = open(os.environ["RELAY_CHAOS_LOG"], "a") if os.environ.get("RELAY_CHAOS_LOG") else subprocess.DEVNULL
        self.proc = subprocess.Popen(
            [binary, "--model", model, "--role", role, "--blocks", str(blocks), "--control-port", str(control_port)],
            stdout=subprocess.PIPE, stderr=log, text=True, env=env)
        line = self.proc.stdout.readline()  # "relay-worker <role> control=<port> kv=<port>"
        if "control=" not in line:
            raise RuntimeError(f"worker did not start: {line!r}")
        self.port = int(line.split("control=")[1].split()[0])

    def kill(self):
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGKILL)
        self.proc.wait()

    def restart(self):
        self.kill()
        return Worker(self.binary, self.model, self.role, self.blocks, self.threads, self.port)


def complete(port, prompt, max_tokens, seed):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=600)
    body = json.dumps({"prompt": prompt, "max_tokens": max_tokens, "temperature": 0.8, "seed": seed,
                       "stream": True, "ignore_eos": True})
    conn.request("POST", "/v1/completions", body=body, headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    if resp.status != 200:
        return None, f"HTTP {resp.status}: {resp.read()[:200]!r}"
    text, buf = [], b""
    while True:
        chunk = resp.read1(65536)
        if not chunk:
            break
        buf += chunk
        while b"\n\n" in buf:
            event, buf = buf.split(b"\n\n", 1)
            if not event.startswith(b"data: ") or event == b"data: [DONE]":
                continue
            d = json.loads(event[6:])
            if "error" in d:
                return None, str(d["error"])
            text.append(d["choices"][0].get("text", ""))
    conn.close()
    return "".join(text), None


def run_all(port, work, concurrency):
    results = [None] * len(work)
    errors = [None] * len(work)
    lock, nxt = threading.Lock(), [0]

    def loop():
        while True:
            with lock:
                i = nxt[0]
                nxt[0] += 1
            if i >= len(work):
                return
            prompt, max_tokens, seed = work[i]
            results[i], errors[i] = complete(port, prompt, max_tokens, seed)

    t0 = time.perf_counter()
    threads = [threading.Thread(target=loop) for _ in range(concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results, errors, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prefill", type=int, default=2)
    ap.add_argument("--decode", type=int, default=2)
    ap.add_argument("--blocks", type=int, default=512)
    ap.add_argument("--requests", type=int, default=64)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--kills", type=int, default=10, help="stop the chaos after this many kills")
    ap.add_argument("--interval", type=float, default=0.5, help="mean seconds between kills")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--worker-bin", default=os.path.join(ROOT, "build", "relay-worker"))
    ap.add_argument("--server-bin", default=os.path.join(ROOT, ".build", "debug", "relay-server"))
    a = ap.parse_args()

    rng = random.Random(a.seed)
    words = ("the cluster keeps serving while workers fail and come back; each request resumes "
             "where it stopped, on another worker, with the same tokens").split()
    work = [(" ".join(rng.choice(words) for _ in range(rng.randint(8, 60))), rng.randint(a.max_tokens // 2, a.max_tokens),
             1000 + i) for i in range(a.requests)]

    roles = ["prefill"] * a.prefill + ["decode"] * a.decode if a.prefill else ["both"] * a.decode
    threads = max(1, (os.cpu_count() or 1) // len(roles))

    def cluster_run(chaos_on):
        """Starts a fresh cluster (so neither run inherits the other's caches), sends every
        request, and returns (texts, errors, seconds, kills)."""
        workers = [Worker(a.worker_bin, a.model, r, a.blocks, threads) for r in roles]
        port = free_port()
        spec = ",".join(f"{w.role}=127.0.0.1:{w.port}" for w in workers)
        log = open(os.environ["RELAY_CHAOS_LOG"], "a") if os.environ.get("RELAY_CHAOS_LOG") else subprocess.DEVNULL
        server = subprocess.Popen([a.server_bin, "--model", a.model, "--workers", spec, "--port", str(port)],
                                  stdout=subprocess.DEVNULL, stderr=log)
        kills, done = [], threading.Event()

        def chaos():
            crng = random.Random(a.seed + 1)
            while not done.is_set() and len(kills) < a.kills:
                time.sleep(crng.uniform(0.5, 1.5) * a.interval)
                i = crng.randrange(len(workers))
                # Keep one worker of each role up: kill i only if another of its role is running.
                others = [w for j, w in enumerate(workers)
                          if j != i and w.role == workers[i].role and w.proc.poll() is None]
                if not others or done.is_set():
                    continue
                workers[i].kill()
                kills.append(workers[i].role)
                if os.environ.get("RELAY_CHAOS_LOG"):
                    with open(os.environ["RELAY_CHAOS_LOG"], "a") as f:
                        f.write(f"[chaos] killed worker {i} ({workers[i].role}) port {workers[i].port}\n")
                time.sleep(crng.uniform(0.1, 0.5))
                workers[i] = workers[i].restart()

        try:
            for _ in range(600):
                try:
                    socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
                    break
                except OSError:
                    time.sleep(0.1)
            t = threading.Thread(target=chaos) if chaos_on else None
            if t:
                t.start()
            texts, errors, seconds = run_all(port, work, a.concurrency)
            done.set()
            if t:
                t.join()
            return texts, errors, seconds, kills
        finally:
            server.terminate()
            server.wait()
            for w in workers:
                w.kill()

    clean, clean_err, clean_s, _ = cluster_run(False)
    if any(clean_err):
        print("fault-free run had errors:", [e for e in clean_err if e][:3])
        return 2
    faulty, faulty_err, faulty_s, kills = cluster_run(True)

    same = sum(1 for x, y in zip(clean, faulty) if x is not None and x == y)
    failed = sum(1 for e in faulty_err if e)
    tokens = sum(m for _, m, _ in work)
    print(f"model {os.path.basename(a.model.rstrip('/'))}, {len(roles)} workers "
          f"({', '.join(f'{roles.count(r)} {r}' for r in sorted(set(roles)))}, {threads} threads each), "
          f"{a.requests} requests ({tokens} tokens), {a.concurrency} at a time, CPU")
    print(f"fault-free run: {clean_s:.1f} s")
    print(f"chaos run:      {faulty_s:.1f} s, {len(kills)} workers killed with SIGKILL and restarted "
          f"({', '.join(f'{kills.count(r)} {r}' for r in sorted(set(kills)))})")
    print(f"identical outputs: {same} of {a.requests}; failed requests: {failed}")
    for i, e in enumerate(faulty_err):
        if e:
            print(f"  request {i} failed: {e}")
    return 0 if same == a.requests else 1


if __name__ == "__main__":
    sys.exit(main())
