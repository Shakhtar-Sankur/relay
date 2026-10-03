"""Builds the test fixtures: tiny randomly initialized models in Hugging Face format,
and the logits Hugging Face transformers computes for them.

    python scripts/make_fixtures.py              # tiny models -> tests/fixtures (committed)
    python scripts/make_fixtures.py --model DIR  # a real model -> DIR/relay-reference.json

The reference file holds a prompt, the logits at every prompt position, then the
greedy continuation with the logits each generated token was picked from. The C++
tests load the same model, run the same prompt, and compare.
"""

import argparse
import json
import os

import torch
from transformers import AutoModelForCausalLM, LlamaConfig, Qwen2Config

FIXTURES = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures")

TINY = {
    # Multi-head attention, untied embeddings, float32 weights.
    "llama-mha": (LlamaConfig, dict(num_attention_heads=4, num_key_value_heads=4), "float32"),
    # Grouped-query attention (4 query heads per KV head), tied embeddings, bf16 weights.
    "llama-gqa-tied-bf16": (LlamaConfig, dict(num_attention_heads=8, num_key_value_heads=2, tie_word_embeddings=True), "bfloat16"),
    # Llama 3.x long-context RoPE scaling and a large rope_theta.
    "llama3-rope": (LlamaConfig, dict(num_attention_heads=4, num_key_value_heads=2, rope_theta=500000.0,
                                      rope_scaling={"rope_type": "llama3", "factor": 32.0, "low_freq_factor": 1.0,
                                                    "high_freq_factor": 4.0, "original_max_position_embeddings": 64}), "float32"),
    # Qwen2: biases on q/k/v, tied embeddings.
    "qwen2-bias": (Qwen2Config, dict(num_attention_heads=4, num_key_value_heads=2, tie_word_embeddings=True), "float32"),
}


def reference(model, prompt, new_tokens):
    model.eval()
    with torch.no_grad():
        ids = torch.tensor([prompt])
        prompt_logits = model(ids).logits[0].float()
        tokens, step_logits = list(prompt), []
        for _ in range(new_tokens):
            lg = model(torch.tensor([tokens])).logits[0, -1].float()
            step_logits.append(lg.tolist())
            tokens.append(int(lg.argmax()))
    return {
        "prompt": prompt,
        "prompt_logits": prompt_logits.tolist(),
        "generated": tokens[len(prompt):],
        "step_logits": step_logits,
    }


def tiny_models():
    os.makedirs(FIXTURES, exist_ok=True)
    for i, (name, (cls, extra, dtype)) in enumerate(TINY.items()):
        torch.manual_seed(1234 + i)
        cfg = cls(vocab_size=128, hidden_size=64, intermediate_size=160, num_hidden_layers=3,
                  max_position_embeddings=256, rms_norm_eps=1e-6, **extra)
        model = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.float32)
        # The default init is too small for the logits to differ much between tokens;
        # larger weights make the test sensitive to mistakes.
        with torch.no_grad():
            for p in model.parameters():
                p.normal_(0.0, 0.15)
        out = os.path.join(FIXTURES, name)
        model.to(getattr(torch, dtype)).save_pretrained(out, safe_serialization=True)
        # Reference computed in float32 from the saved (possibly bf16) weights.
        model = AutoModelForCausalLM.from_pretrained(out, torch_dtype=torch.float32)
        g = torch.Generator().manual_seed(99 + i)
        prompt = torch.randint(0, 128, (21,), generator=g).tolist()
        with open(os.path.join(out, "relay-reference.json"), "w") as f:
            json.dump(reference(model, prompt, 8), f)
        print("wrote", out)


def real_model(path, prompt_text, new_tokens):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32)
    prompt = tok(prompt_text)["input_ids"]
    ref = reference(model, prompt, new_tokens)
    ref["prompt_logits"] = ref["prompt_logits"][-1:]  # the full table is large; the last row is enough
    ref["prompt_logits_last_only"] = True
    ref["text"] = prompt_text
    with open(os.path.join(path, "relay-reference.json"), "w") as f:
        json.dump(ref, f)
    print("wrote", os.path.join(path, "relay-reference.json"), "prompt tokens:", len(prompt),
          "continuation:", repr(tok.decode(ref["generated"])))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--prompt", default="The three most important ideas in distributed systems are")
    ap.add_argument("--new-tokens", type=int, default=12)
    a = ap.parse_args()
    if a.model:
        real_model(a.model, a.prompt, a.new_tokens)
    else:
        tiny_models()
