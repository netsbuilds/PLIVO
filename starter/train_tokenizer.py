"""Trains the BPE tokenizer on train_corpus.txt and verifies lossless round-trip.
Run once: python train_tokenizer.py
"""
import os
from tokenizer import BPETokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(HERE, "..", "data", "train_corpus.txt")
DEV = os.path.join(HERE, "..", "data", "dev_eval.txt")
OUT = os.path.join(HERE, "tokenizer.json")  # matches tokenizer.py's load() default

VOCAB_SIZE = 1000  # matches your existing trained tokenizer.json


def bytes_per_token(tok, text):
    ids = tok.encode(text)
    return len(text.encode("utf-8")) / max(len(ids), 1)


def is_devanagari_line(line):
    return any('\u0900' <= ch <= '\u097F' for ch in line)


def main():
    with open(CORPUS, "r", encoding="utf-8") as f:
        train_text = f.read()

    tok = BPETokenizer(vocab_size=VOCAB_SIZE)   # <- fixed: vocab_size in constructor
    print(f"Training BPE, vocab_size={VOCAB_SIZE} ...")
    tok.train(train_text)                       # <- fixed: no extra kwargs
    tok.save(OUT)
    print(f"Saved to {OUT}, actual vocab_size={tok.vocab_size}")

    lines = train_text.splitlines()
    hi_lines = [l for l in lines if is_devanagari_line(l)]
    en_lines = [l for l in lines if not is_devanagari_line(l)]
    hi_text = "\n".join(hi_lines) or " "
    en_text = "\n".join(en_lines) or " "
    print(f"Bytes/token overall: {bytes_per_token(tok, train_text):.3f}")
    print(f"Bytes/token Hindi lines: {bytes_per_token(tok, hi_text):.3f}")
    print(f"Bytes/token English lines: {bytes_per_token(tok, en_text):.3f}")

    with open(DEV, "r", encoding="utf-8") as f:
        dev_text = f.read()
    ok = tok.decode(tok.encode(dev_text)) == dev_text
    print(f"Round-trip on dev_eval.txt: {'PASS' if ok else 'FAIL'}")

    adversarial = [
        "", "a", "🙂🔥", "मात्र", "hello नमस्ते world",
        "a" * 500, "्", "क्षत्रिय", "\n\n\t  ", "𝕏𝕐𝕑",
    ]
    all_pass = True
    for s in adversarial:
        rt = tok.decode(tok.encode(s))
        passed = rt == s
        all_pass &= passed
        if not passed:
            print(f"FAIL on {s!r}: got {rt!r}")
    print(f"Adversarial round-trip: {'ALL PASS' if all_pass else 'SOME FAILED'}")

    n_embd = 160
    print(f"Embedding+head param cost @ n_embd={n_embd}: "
          f"untied={tok.vocab_size * n_embd * 2:,}, tied={tok.vocab_size * n_embd:,}")


if __name__ == "__main__":
    main()