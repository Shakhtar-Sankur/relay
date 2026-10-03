%%bash
# relay M2 on a GPU: the KV transfer engine with the CUDA backend.
#  1. every test, including prefill-on-one-backend / decode-on-another, on the CUDA backend
#  2. raw transport speed (TCP and shared memory between two processes)
#  3. layer streaming against sending after the forward pass, TinyLlama-1.1B, 512 and 2048-token prompts
# Colab: Runtime -> Change runtime type -> T4 GPU. Paste this file into ONE cell (about 10 minutes).
set -e
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || echo "NO GPU: turn on a T4 GPU"
cd /content && rm -rf relay && git clone -q https://github.com/Shakhtar-Sankur/relay && cd relay
echo "relay commit: $(git log -1 --format='%h %s')"
pip install -q transformers safetensors huggingface_hub 2>&1 | tail -1 || true
nproc

echo "== build (CPU + CUDA backends)"
cmake -S . -B build -DRELAY_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 > build.log 2>&1 || { tail -30 build.log; exit 1; }
cmake --build build -j 4 >> build.log 2>&1 || { grep -E "error" build.log | head -30; exit 1; }

python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("TinyLlama/TinyLlama-1.1B-Chat-v1.0", local_dir="models/tinyllama-1.1b", allow_patterns=["*.json", "*.safetensors"])
print("downloaded TinyLlama-1.1B")
PY

for be in cpu cuda; do
  echo "== tests on the $be backend"
  for t in safetensors sampler kv_cache reference batching transfer; do
    RELAY_TEST_BACKEND=$be ./build/test_$t > test_${be}_$t.log 2>&1 && r=ok || r=FAILED
    echo "test_$t: $r  ($(tail -1 test_${be}_$t.log))"
    grep -E "frames|fell back|FAILED" test_${be}_$t.log | sed 's/^/    /' || true
  done
done

echo "== raw transport speed"
./build/relay-transfer-bench raw --transport tcp
./build/relay-transfer-bench raw --transport shm

echo "== layer streaming vs sending after the forward pass (CUDA backend)"
for p in 512 2048; do
  for tr in tcp shm; do
    ./build/relay-transfer-bench stream --model models/tinyllama-1.1b --backend cuda --transport $tr --prompt $p --repeats 5
    echo
  done
done
echo "== done: copy everything from the first line and send it back"
