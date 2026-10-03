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


CHATML = (
    "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}"
    "{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}"
    "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}")

CONVERSATIONS = [
    [{"role": "user", "content": "Hi there"}],
    [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "What is 2+2?"}],
    [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"},
     {"role": "assistant", "content": "Hello! How can I help?"}, {"role": "user", "content": "Tell me about café 東京 🚀"}],
]


def chat_cases(path):
    """How transformers renders a few conversations with the model's chat template."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    cases = [{"messages": m, "text": tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)}
             for m in CONVERSATIONS]
    with open(os.path.join(path, "relay-chat-cases.json"), "w") as f:
        json.dump(cases, f, ensure_ascii=False)


def chat_fixture():
    """A tiny model with a real tokenizer and chat template, for end-to-end server tests."""
    out = os.path.join(FIXTURES, "chat-tiny")
    os.makedirs(out, exist_ok=True)
    tok_src = os.path.join(FIXTURES, "tokenizers", "gpt2-digits.json")
    with open(tok_src) as f:
        tok_json = json.load(f)
    vocab = max(max(tok_json["model"]["vocab"].values()), max(a["id"] for a in tok_json["added_tokens"])) + 1
    im_end = next(a["id"] for a in tok_json["added_tokens"] if a["content"] == "<|im_end|>")
    torch.manual_seed(4321)
    cfg = LlamaConfig(vocab_size=vocab, hidden_size=64, intermediate_size=160, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
                      rms_norm_eps=1e-6, tie_word_embeddings=True, eos_token_id=im_end, bos_token_id=None)
    model = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.float32)
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0.0, 0.15)
    model.save_pretrained(out, safe_serialization=True)
    with open(os.path.join(out, "tokenizer.json"), "w") as f:
        json.dump(tok_json, f, ensure_ascii=False)
    with open(os.path.join(out, "tokenizer_config.json"), "w") as f:
        json.dump({"chat_template": CHATML, "eos_token": "<|im_end|>", "bos_token": None,
                   "tokenizer_class": "PreTrainedTokenizerFast", "model_max_length": 512}, f)
    model = AutoModelForCausalLM.from_pretrained(out, torch_dtype=torch.float32)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(out)
    text = tok.apply_chat_template(CONVERSATIONS[1], tokenize=False, add_generation_prompt=True)
    ref = reference(model, tok(text)["input_ids"], 8)
    ref["text"] = text
    with open(os.path.join(out, "relay-reference.json"), "w") as f:
        json.dump(ref, f)
    chat_cases(out)
    print("wrote", out, "vocab", vocab)


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
        if os.path.exists(os.path.join(a.model, "tokenizer_config.json")):
            chat_cases(a.model)
    else:
        tiny_models()
        chat_fixture()
