import os
import random

import numpy as np
import sentencepiece as spm

BLOCK = 1024
MIN_CHUNK_LEN = 32


def resolve_tokenizer():
    env_path = os.environ.get("TOKENIZER_MODEL")
    candidates = [
        env_path,
        "tokenizer/mix32k.model",
        "tokenizer/mix.model",
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return path
    raise FileNotFoundError(
        "No tokenizer model found. Set TOKENIZER_MODEL or add tokenizer/mix32k.model."
    )


TOKENIZER = resolve_tokenizer()
sp = spm.SentencePieceProcessor(model_file=TOKENIZER)

print("tokenizer:", TOKENIZER)

files = [
    "raw_data/alpaca_gpt4.txt",
    "raw_data/openhermes.txt",
    "raw_data/codealpaca.txt",
    "raw_data/gsm8k.txt",
]

chunks = []

for path in files:
    if not os.path.exists(path):
        print("missing", path)
        continue

    print("loading", path)

    with open(path, encoding="utf8", errors="ignore") as f:
        text = f.read()

    docs = [d.strip() for d in text.split("\n\n") if len(d.strip()) > 30]

    for doc in docs:
        tokens = sp.encode(doc)

        for i in range(0, len(tokens), BLOCK):
            chunk = tokens[i:i + BLOCK + 1]
            if len(chunk) >= MIN_CHUNK_LEN:
                chunks.append(chunk)

print("chunks:", len(chunks))

if not chunks:
    raise RuntimeError("No instruction chunks were created. Add instruction data under raw_data/ first.")

random.shuffle(chunks)

tokens = [token for chunk in chunks for token in chunk]
split = int(len(tokens) * 0.95)

train = np.array(tokens[:split], dtype=np.uint16)
val = np.array(tokens[split:], dtype=np.uint16)

os.makedirs("datasets", exist_ok=True)

train.tofile("datasets/train_instr.bin")
val.tofile("datasets/val_instr.bin")

print("train tokens:", len(train))
print("val tokens:", len(val))
print("DONE")
