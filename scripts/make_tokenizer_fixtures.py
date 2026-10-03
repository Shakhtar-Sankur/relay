"""Tokenizer test fixtures: what Hugging Face `tokenizers` produces, for the Swift tokenizer
to match.

    python scripts/make_tokenizer_fixtures.py            # three small trained tokenizers
                                                         # -> tests/fixtures/tokenizers (committed)
    python scripts/make_tokenizer_fixtures.py --model DIR  # a real model's tokenizer.json
                                                         # -> DIR/relay-tokenizer-cases.json

The small tokenizers are trained here with the same pipeline as the real models:
  gpt2-digits     Digits(individual) + ByteLevel regex           (SmolLM2, Llama 3 style)
  split-bytelevel NFC + Split(Qwen2 regex) + ByteLevel            (Qwen2 style)
  sentencepiece   "▁" for spaces, BPE with byte fallback, <s>   (Llama 2, TinyLlama style)
"""

import argparse
import json
import os

from tokenizers import Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers, processors, trainers

OUT = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures", "tokenizers")

QWEN_PATTERN = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*"
                r"|\s*[\r\n]+|\s+(?!\S)|\s+")

CORPUS = [
    "The three most important ideas in distributed systems are consensus, replication and failure detection.",
    "Prefill is compute-bound; decode is memory-bandwidth-bound. Splitting them lets each pool be tuned.",
    "In 2024 we served 1,234,567 requests at 99.9% availability, with p99 latency under 250 ms.",
    "def forward(self, x):\n    return self.proj(x) + x  # residual\n",
    "Unicode: café, naïve, Zürich, 東京, 서울, Москва, emoji 🚀🔥, and math ∑ x² ≤ ∞.",
    "Don't stop; we'll see. It's what they've said, isn't it? I'm sure you'd agree.",
    "   leading spaces, trailing spaces   \n\nand\tTabs\r\nwindows line ends.",
] * 20

CASES = [
    "Hello world",
    "Hello  world   ",
    " leading space",
    "The three most important ideas in distributed systems are",
    "In 2024, revenue grew 12.5% to $3,456,789.",
    "phone: 555-0199, ratio 3/4, version v1.2.3",
    "café naïve Zürich",
    "東京と서울 and Москва",
    "emoji 🚀🔥👩‍💻 done",
    "x² + y³ = z⁴ ≤ ∞",
    "Don't, won't, they've, I'm, we'll, he'd, IT'S",
    "It's here and she's there; that's it.",
    "line one\nline two\n\nline four\r\nwindows",
    "\ttab\tseparated\tvalues",
    "    def f(x):\n        return x  # comment",
    "",
    " ",
    "\n",
    "a",
    "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n",
    "<s> literal special </s> tokens",
    "é combining accent vs é precomposed",
    "zero​width space and non breaking",
    "repeated !!!!!!!!!! ?????? ......",
    "MiXeD CaSe WoRdS and UPPER lower",
]


def train(kind):
    special = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    if kind == "gpt2-digits":
        tok = Tokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Digits(individual_digits=True),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)])
        tok.decoder = decoders.ByteLevel()
        tr = trainers.BpeTrainer(vocab_size=420, special_tokens=special,
                                 initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    elif kind == "split-bytelevel":
        tok = Tokenizer(models.BPE())
        tok.normalizer = normalizers.NFC()
        tok.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(Regex(QWEN_PATTERN), behavior="isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
        tok.decoder = decoders.ByteLevel()
        tok.post_processor = processors.ByteLevel(trim_offsets=False)
        tr = trainers.BpeTrainer(vocab_size=420, special_tokens=special,
                                 initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    else:  # sentencepiece
        special = ["<unk>", "<s>", "</s>"]
        tok = Tokenizer(models.BPE(unk_token="<unk>", byte_fallback=True, fuse_unk=True))
        tok.normalizer = normalizers.Sequence([normalizers.Prepend("▁"), normalizers.Replace(" ", "▁")])
        tok.decoder = decoders.Sequence([decoders.Replace("▁", " "), decoders.ByteFallback(), decoders.Fuse(),
                                         decoders.Strip(" ", 1, 0)])
        # Byte tokens are ordinary vocabulary entries in SentencePiece models.
        tr = trainers.BpeTrainer(vocab_size=420, special_tokens=special + [f"<0x{b:02X}>" for b in range(256)],
                                 limit_alphabet=60, show_progress=False)
    tok.train_from_iterator(CORPUS, tr)
    if kind == "sentencepiece":
        tok.post_processor = processors.TemplateProcessing(single="<s> $A", pair="<s> $A <s> $B",
                                                           special_tokens=[("<s>", tok.token_to_id("<s>"))])
        # The trainer registered the byte tokens as added tokens; make them plain vocabulary
        # (as in a converted SentencePiece model), keeping only the three real specials.
        j = json.loads(tok.to_str())
        j["added_tokens"] = [a for a in j["added_tokens"] if a["content"] in special]
        tok = Tokenizer.from_str(json.dumps(j))
    return tok


def cases_for(tok, add_special=True):
    out = []
    for text in CASES:
        ids = tok.encode(text, add_special_tokens=add_special).ids
        out.append({"text": text, "ids": ids,
                    "decoded": tok.decode(ids, skip_special_tokens=True),
                    "decoded_with_special": tok.decode(ids, skip_special_tokens=False)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    a = ap.parse_args()
    if a.model:
        tok = Tokenizer.from_file(os.path.join(a.model, "tokenizer.json"))
        with open(os.path.join(a.model, "relay-tokenizer-cases.json"), "w") as f:
            json.dump(cases_for(tok), f, ensure_ascii=False)
        print("wrote", os.path.join(a.model, "relay-tokenizer-cases.json"))
        return
    os.makedirs(OUT, exist_ok=True)
    for kind in ["gpt2-digits", "split-bytelevel", "sentencepiece"]:
        tok = train(kind)
        tok.save(os.path.join(OUT, f"{kind}.json"))
        with open(os.path.join(OUT, f"{kind}.cases.json"), "w") as f:
            json.dump(cases_for(tok), f, ensure_ascii=False)
        print("wrote", kind, "vocab", tok.get_vocab_size())


if __name__ == "__main__":
    main()
