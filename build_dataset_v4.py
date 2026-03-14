# build_dataset_v4.py
# Run: python build_dataset_v4.py
# Output: datasets/train_v4.bin + datasets/val_v4.bin

import os, random
import numpy as np
import sentencepiece as spm

TOKENIZER  = "tokenizer/mix.model"
BLOCK_SIZE = 512
SEED       = 42
random.seed(SEED)

MIX = {
    "raw_data/fineweb_edu.txt":    0.30,
    "raw_data/wikipedia.txt":      0.20,
    "raw_data/python_code.txt":    0.15,
    "raw_data/books.txt":          0.10,
    "raw_data/openhermes.txt":     0.10,
    "raw_data/alpaca_gpt4.txt":    0.08,
    "raw_data/codealpaca.txt":     0.04,
    "raw_data/gsm8k.txt":          0.02,
    "raw_data/openwebtext.txt":    0.01,
}

os.makedirs("datasets", exist_ok=True)
sp = spm.SentencePieceProcessor()
sp.Load(TOKENIZER)
print(f"Tokenizer loaded — vocab: {sp.GetPieceSize()}")

all_chunks = []
for path, ratio in MIX.items():
    if not os.path.exists(path):
        print(f"[SKIP] {path} not found")
        continue
    print(f"Tokenizing {path} (ratio={ratio})...")
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    docs = [d.strip() for d in text.split("\n\n") if len(d.strip()) > 20]
    target = int(len(docs) * ratio * 10)
    sampled = random.sample(docs, min(target, len(docs)))
    chunks = []
    for doc in sampled:
        tokens = sp.Encode(doc, out_type=int)
        for i in range(0, len(tokens), BLOCK_SIZE):
            chunk = tokens[i:i+BLOCK_SIZE+1]
            if len(chunk) >= 32:
                chunks.append(chunk)
    all_chunks.extend(chunks)
    print(f"  {len(chunks):,} chunks from {len(sampled):,} docs")

print(f"\nTotal chunks: {len(all_chunks):,}")
random.shuffle(all_chunks)

all_tokens = [t for chunk in all_chunks for t in chunk]
print(f"Total tokens: {len(all_tokens):,}")

split = int(len(all_tokens) * 0.9)
train = np.array(all_tokens[:split], dtype=np.uint16)
val   = np.array(all_tokens[split:], dtype=np.uint16)

train.tofile("datasets/train_v4.bin")
val.tofile("datasets/val_v4.bin")

print(f"\n[OK] train_v4.bin: {len(train):,} tokens ({len(train)*2/1e9:.2f} GB)")
print(f"[OK] val_v4.bin:   {len(val):,} tokens")
print("Next: push to GitHub then run on RunPod")
