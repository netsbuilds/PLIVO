"""Run after every architecture change to confirm the 2M param cap holds.
python check_params.py
"""
from model import GPT, Config

model = GPT(Config())
n = model.n_params()
print(f"Total params: {n:,} / 2,000,000 cap")
if n > 2_000_000:
    raise SystemExit("OVER PARAM CAP — reduce n_embd or n_layer")
print("OK — under cap")