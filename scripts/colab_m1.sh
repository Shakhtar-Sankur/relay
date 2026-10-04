%%bash
# relay M1 on a GPU: the new attention kernels (flash-style prefill, split-context decode)
# and weight-only int8, checked against the M0 kernel and Hugging Face, then timed.
# Colab: Runtime -> Change runtime type -> T4 GPU. Paste this file into ONE cell (about 15-20 minutes).
set -e
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || echo "NO GPU: turn on a T4 GPU"
BRANCH=${RELAY_BRANCH:-main}
cd /content && rm -rf relay && git clone -q -b $BRANCH https://github.com/Shakhtar-Sankur/relay && cd relay
echo "relay $BRANCH: $(git log -1 --format='%h %s')"
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

echo "== 1. M1 attention against M0 attention, and int8 against fp16 (same GPU, same inputs)"
RELAY_TEST_BACKEND=cuda ./build/test_attention 2>&1 | grep -vE "^\[ RUN" || echo "test_attention FAILED"

echo "== 2. every model test on the CUDA backend with the M1 kernels (logits against Hugging Face)"
for t in safetensors sampler kv_cache reference batching transfer; do
  RELAY_TEST_BACKEND=cuda ./build/test_$t > test_$t.log 2>&1 && r=ok || r=FAILED
  echo "test_$t: $r  ($(tail -1 test_$t.log))"
  grep -E "relative error|FAILED" test_$t.log | sed 's/^/    /' | head -12 || true
done

echo "== 3. forward-pass time: M0 kernel vs M1 kernels vs M1 + int8"
./build/relay-attention-bench --model models/tinyllama-1.1b --repeats 5
./build/relay-attention-bench --model models/smollm2-135m --repeats 5

echo "== 4. end-to-end generation speed (TinyLlama-1.1B, 128 new tokens per request; M0 was 92 / 535 / 1217 tok/s)"
IDS=$(python -c "from transformers import AutoTokenizer as A; t=A.from_pretrained('models/tinyllama-1.1b'); print(' '.join(map(str, t('The three most important ideas in distributed systems are')['input_ids'])))")
for n in 1 8 32; do
  ARGS=""; for i in $(seq 1 $n); do ARGS="$ARGS --ids \"$IDS\""; done
  eval ./build/relay-generate --model models/tinyllama-1.1b $ARGS --max-new 128 --ignore-eos --backend cuda 2>&1 >/dev/null | sed "s/^/  M1 batch $n: /"
  RELAY_CUDA_INT8=1 eval ./build/relay-generate --model models/tinyllama-1.1b $ARGS --max-new 128 --ignore-eos --backend cuda 2>&1 >/dev/null | sed "s/^/  M1+int8 batch $n: /"
done
echo "== done: copy everything from the first line and send it back"
