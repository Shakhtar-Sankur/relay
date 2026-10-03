%%bash
# relay M0 on a GPU: build with CUDA, run every test on the CUDA backend, compare with
# Hugging Face on two real models, and time generation.
# Colab: Runtime -> Change runtime type -> T4 GPU. Paste this file into ONE cell (about 10 minutes).
set -e
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || echo "NO GPU: turn on a T4 GPU"
cd /content && rm -rf relay && git clone -q https://github.com/Shakhtar-Sankur/relay && cd relay
echo "relay commit: $(git log -1 --format='%h %s')"
pip install -q transformers safetensors huggingface_hub 2>&1 | tail -1 || true

echo "== build (CPU + CUDA backends)"
cmake -S . -B build -DRELAY_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 > build.log 2>&1 || { tail -30 build.log; exit 1; }
cmake --build build -j 4 >> build.log 2>&1 || { grep -E "error" build.log | head -30; exit 1; }
echo "built: $(ls build | grep -E '^(test_|relay-)' | tr '\n' ' ')"

echo "== real models and their Hugging Face reference outputs"
python - <<'PY'
from huggingface_hub import snapshot_download
for repo, name in [("HuggingFaceTB/SmolLM2-135M", "smollm2-135m"), ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "tinyllama-1.1b")]:
    snapshot_download(repo, local_dir=f"models/{name}", allow_patterns=["*.json", "*.safetensors"])
    print("downloaded", repo)
PY
for m in smollm2-135m tinyllama-1.1b; do python scripts/make_fixtures.py --model models/$m 2>&1 | grep wrote; done
export RELAY_REFERENCE_MODELS=$PWD/models/smollm2-135m:$PWD/models/tinyllama-1.1b

for be in cpu cuda; do
  echo "== tests on the $be backend"
  for t in safetensors sampler kv_cache reference batching; do
    RELAY_TEST_BACKEND=$be ./build/test_$t > test_${be}_$t.log 2>&1 && r=ok || r=FAILED
    echo "test_$t: $r  ($(tail -1 test_${be}_$t.log))"
    grep -E "relative error|preemptions|FAILED" test_${be}_$t.log | sed 's/^/    /' || true
  done
done

echo "== generation speed (M0 kernels: cuBLAS GEMMs, simple attention; M1 replaces the attention)"
IDS=$(python -c "from transformers import AutoTokenizer as A; t=A.from_pretrained('models/tinyllama-1.1b'); print(' '.join(map(str, t('The three most important ideas in distributed systems are')['input_ids'])))")
for n in 1 8 32; do
  ARGS=""; for i in $(seq 1 $n); do ARGS="$ARGS --ids \"$IDS\""; done
  eval ./build/relay-generate --model models/tinyllama-1.1b $ARGS --max-new 128 --ignore-eos --backend cuda 2>&1 >/dev/null | sed "s/^/  batch $n: /"
done
echo "== done: copy everything from the first line and send it back"
