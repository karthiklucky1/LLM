# build_dataset_v3.py
# Run: python build_dataset_v3.py
# Creates: datasets/train_v3.bin, datasets/val_v3.bin

import os
import random
import numpy as np
import sentencepiece as spm

# ── CONFIG ───────────────────────────────────────────────────────────────────
TOKENIZER_PATH = "tokenizer/mix.model"   # existing tokenizer — do not change
OUTPUT_TRAIN = "datasets/train_v3.bin"
OUTPUT_VAL = "datasets/val_v3.bin"
BLOCK_SIZE = 256                      # must match 40k checkpoint block_size
RANDOM_SEED = 42

# How much of each file to use (None = all)
# Adjust these if you run out of RAM
FILE_LIMITS = {
    "raw_data/alpaca_gpt4.txt": None,    # ~52k examples, ~50MB
    "raw_data/alpaca_cleaned.txt": None,    # ~52k examples, ~40MB
    "raw_data/wikipedia.txt": None,    # 60k articles, large
    "raw_data/python_code.txt": None,    # 60k files
    "raw_data/openwebtext.txt": None,    # 20k docs
}

os.makedirs("datasets", exist_ok=True)
random.seed(RANDOM_SEED)

# ── LOAD TOKENIZER ───────────────────────────────────────────────────────────
print(f"Loading tokenizer from {TOKENIZER_PATH}...")
sp = spm.SentencePieceProcessor()
sp.Load(TOKENIZER_PATH)
print(f"  Vocab size: {sp.GetPieceSize()}")

# Verify ### markers are tokenized (critical for instruction format)
test_tokens = sp.Encode("### Instruction:", out_type=int)
print(f"  '### Instruction:' encodes to {len(test_tokens)} tokens: {test_tokens}")
if len(test_tokens) > 8:
    print("  WARNING: ### markers are over-split — instruction format may be weak")
    print("  This is OK for now but consider expanding vocab in future")

# ── TOKENIZE EACH FILE ───────────────────────────────────────────────────────
all_token_chunks = []   # list of token lists, one per chunk

def tokenize_file(path, max_chars=None):
    print(f"\nTokenizing {path}...")
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    if max_chars:
        text = text[:max_chars]

    # Split into documents (double newline = document boundary)
    docs = [d.strip() for d in text.split("\n\n") if len(d.strip()) > 20]
    print(f"  {len(docs)} documents found")

    chunks = []
    total_tokens = 0
    for doc in docs:
        tokens = sp.Encode(doc, out_type=int)
        if len(tokens) < 10:
            continue   # skip very short docs
        # Split long docs into block_size chunks with overlap
        for i in range(0, len(tokens), BLOCK_SIZE):
            chunk = tokens[i:i + BLOCK_SIZE + 1]
            if len(chunk) >= 32:  # minimum useful chunk length
                chunks.append(chunk)
        total_tokens += len(tokens)

    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Total chunks: {len(chunks):,}")
    return chunks

for path, limit in FILE_LIMITS.items():
    if os.path.exists(path):
        chunks = tokenize_file(path, max_chars=limit)
        all_token_chunks.extend(chunks)
    else:
        print(f"\n[SKIP] {path} not found — run download_datasets.py first")

print(f"\nTotal chunks before shuffle: {len(all_token_chunks):,}")

# ── SHUFFLE AT CHUNK LEVEL ───────────────────────────────────────────────────
# Shuffle chunks, not individual tokens — preserves document context
random.shuffle(all_token_chunks)

# ── FLATTEN TO TOKEN ARRAY ───────────────────────────────────────────────────
all_tokens = []
for chunk in all_token_chunks:
    all_tokens.extend(chunk)

print(f"Total tokens: {len(all_tokens):,}")

# ── TRAIN/VAL SPLIT (90/10) ──────────────────────────────────────────────────
split = int(len(all_tokens) * 0.9)
train_tokens = np.array(all_tokens[:split], dtype=np.uint16)
val_tokens = np.array(all_tokens[split:], dtype=np.uint16)

# ── SAVE BINARY FILES ────────────────────────────────────────────────────────
train_tokens.tofile(OUTPUT_TRAIN)
val_tokens.tofile(OUTPUT_VAL)

train_mb = os.path.getsize(OUTPUT_TRAIN) / (1024*1024)
val_mb = os.path.getsize(OUTPUT_VAL) / (1024*1024)

print(f"\n[OK] Saved {OUTPUT_TRAIN}: {len(train_tokens):,} tokens ({train_mb:.1f} MB)")
print(f"[OK] Saved {OUTPUT_VAL}:   {len(val_tokens):,} tokens ({val_mb:.1f} MB)")
print(f"\nNext step: update train.py and run continued training")
