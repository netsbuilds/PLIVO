"""Final verification before submission: param cap, tokenizer/config match,
lossless round-trip, and evaluate.py sanity call.
python final_check.py
"""
import os
import subprocess
import sys

import torch

from model import GPT, Config
import tokenizer as tokenizer_mod

HERE = os.path.dirname(os.path.abspath(__file__))
DEV_EVAL = os.path.join(HERE, "..", "data", "dev_eval.txt")

cfg = Config()
model = GPT(cfg)
n = model.n_params()
print(f"Params: {n:,} / 2,000,000  {'OK' if n <= 2_000_000 else 'OVER CAP!!'}")

tok = tokenizer_mod.load()
if tok.vocab_size != cfg.vocab_size:
    print(f"WARNING: tokenizer.vocab_size={tok.vocab_size} != "
          f"Config.vocab_size={cfg.vocab_size} — train.py sets this "
          f"dynamically at train time, but double check ckpt.pt's saved "
          f"config matches what evaluate.py will load.")

with open(DEV_EVAL, "r", encoding="utf-8") as f:
    dev_text = f.read()
ok = tok.decode(tok.encode(dev_text)) == dev_text
print(f"Round-trip lossless on dev_eval.txt: {'PASS' if ok else 'FAIL'}")

adversarial = ["", "🙂", "मात्र", "hello नमस्ते", "क्षत्रिय" * 10, "\n\n\t"]
all_pass = True
for s in adversarial:
    if tok.decode(tok.encode(s)) != s:
        print(f"FAIL on {s!r}")
        all_pass = False
print(f"Adversarial round-trip: {'ALL PASS' if all_pass else 'SOME FAILED'}")

if os.path.exists(os.path.join(HERE, "ckpt.pt")):
    print("\nRunning evaluate.py against ckpt.pt ...")
    result = subprocess.run(
        [sys.executable, "evaluate.py", "--checkpoint", "ckpt.pt",
         "--text_file", DEV_EVAL],
        cwd=HERE, capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print("evaluate.py FAILED:", result.stderr)
else:
    print("\nNo ckpt.pt found yet — run train.py first.")