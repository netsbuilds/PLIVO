# measure_baseline.py
import sys
# Assume your data is at ../data/train_corpus.txt
with open(r"C:\Users\Hp\Desktop\plivo\llm_handout\data\train_corpus.txt", "r", encoding="utf-8") as f:
    text = f.read()
    # Simple check for Hindi (Devanagari range: \u0900-\u097f)
    hindi_lines = [line for line in text.split('\n') if any('\u0900' <= char <= '\u097f' for char in line)]
    english_lines = [line for line in text.split('\n') if not any('\u0900' <= char <= '\u097f' for char in line)]

print(f"Total characters in train_corpus: {len(text)}")
print(f"Hindi lines: {len(hindi_lines)}, English lines: {len(english_lines)}")

with open(r"C:\Users\Hp\Desktop\plivo\llm_handout\data\dev_eval.txt", "r", encoding="utf-8") as f:
    text = f.read()
    # Simple check for Hindi (Devanagari range: \u0900-\u097f)
    hindi_lines = [line for line in text.split('\n') if any('\u0900' <= char <= '\u097f' for char in line)]
    english_lines = [line for line in text.split('\n') if not any('\u0900' <= char <= '\u097f' for char in line)]

print(f"Total characters in dev_eval: {len(text)}")
print(f"Hindi lines: {len(hindi_lines)}, English lines: {len(english_lines)}")