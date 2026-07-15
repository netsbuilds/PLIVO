import json, os, re
from collections import Counter
from tqdm import tqdm
import ast

class BPETokenizer:
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}

    def train(self, text):
        tokens = list(text.encode("utf-8"))
        pbar = tqdm(total=self.vocab_size - 256, desc="Training BPE")
        while len(self.vocab) < self.vocab_size:
            pairs = Counter(zip(tokens, tokens[1:]))
            if not pairs:
                break
            best_pair = pairs.most_common(1)[0][0]

            new_id = len(self.vocab)
            self.merges[best_pair] = new_id
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
                    new_tokens.append(new_id)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
            pbar.update(1)
        pbar.close()

    def encode(self, text):
        b = list(text.encode("utf-8"))
        while len(b) >= 2:
            pairs = {(b[i], b[i + 1]): i for i in range(len(b) - 1)}
            best_pair = None
            for pair in self.merges:
                if pair in pairs:
                    best_pair = pair
                    break
            if not best_pair:
                break
            idx = pairs[best_pair]
            b = b[:idx] + [self.merges[best_pair]] + b[idx + 2:]
        return b

    def decode(self, ids):
        # errors="strict" on purpose: byte-level BPE merges never destroy
        # bytes, so a decode failure here means a real bug, not something
        # to paper over with errors="replace" (which would silently break
        # the lossless guarantee evaluate.py depends on).
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="strict")

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"merges": {str(k): v for k, v in self.merges.items()},
                       "vocab_size": self.vocab_size}, f)


def load(path=None):
    """Return the tokenizer used by evaluate.py/train.py. No-argument call
    resolves tokenizer.json relative to THIS file, not the caller's cwd —
    otherwise grading (which may run with a different cwd) can silently
    fail to find it."""
    here = os.path.dirname(os.path.abspath(__file__))
    if path is None:
        path = os.path.join(here, "tokenizer.json")

    tok = BPETokenizer()
    if os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)
        tok.merges = {ast.literal_eval(k): v for k, v in data["merges"].items()}
        tok.vocab_size = data["vocab_size"]
        for pair, new_id in tok.merges.items():
            tok.vocab[new_id] = tok.vocab[pair[0]] + tok.vocab[pair[1]]
    return tok