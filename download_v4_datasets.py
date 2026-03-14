# download_v4_datasets.py
# Downloads NEW datasets only — skips files that already exist
# Run: python download_v4_datasets.py

import os
from datasets import load_dataset

os.makedirs("raw_data", exist_ok=True)

def already_exists(path, min_mb=5):
    if not os.path.exists(path):
        return False
    return os.path.getsize(path) / (1024*1024) >= min_mb

# ── 1. FineWeb-edu ─────────────────────────────────────────────
PATH = "raw_data/fineweb_edu.txt"
if already_exists(PATH, min_mb=500):
    print(f"[SKIP] {PATH} already exists")
else:
    print("Downloading FineWeb-edu (300k docs)...")
    try:
        ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                          split="train", streaming=True, trust_remote_code=True)
        with open(PATH, "w", encoding="utf-8") as f:
            count = 0
            for row in ds:
                text = row["text"].strip()
                if len(text) > 200:
                    f.write(text + "\n\n")
                    count += 1
                if count >= 300000:
                    break
                if count % 10000 == 0:
                    print(f"  {count}/300000...")
        print(f"  [OK] fineweb_edu.txt — {count} docs")
    except Exception as e:
        print(f"  [FAIL] {e}")

# ── 2. OpenHermes 2.5 ─────────────────────────────────────────
PATH = "raw_data/openhermes.txt"
if already_exists(PATH, min_mb=20):
    print(f"[SKIP] {PATH} already exists")
else:
    print("Downloading OpenHermes 2.5 (30k examples)...")
    try:
        ds = load_dataset("teknium/OpenHermes-2.5", split="train",
                          streaming=True, trust_remote_code=True)
        with open(PATH, "w", encoding="utf-8") as f:
            count = 0
            for row in ds:
                conv = row.get("conversations", [])
                text = ""
                for turn in conv:
                    role = "### Instruction:" if turn["from"] == "human" else "### Response:"
                    text += f"{role}\n{turn['value']}\n\n"
                if text.strip():
                    f.write(text + "\n")
                    count += 1
                if count >= 30000:
                    break
        print(f"  [OK] openhermes.txt — {count} examples")
    except Exception as e:
        print(f"  [FAIL] {e}")

# ── 3. Books (Project Gutenberg) ──────────────────────────────
PATH = "raw_data/books.txt"
if already_exists(PATH, min_mb=100):
    print(f"[SKIP] {PATH} already exists")
else:
    print("Downloading Gutenberg books...")
    try:
        ds = load_dataset("pgcorpus/gutenberg", split="train",
                          streaming=True, trust_remote_code=True)
        with open(PATH, "w", encoding="utf-8") as f:
            count = 0
            for row in ds:
                text = row.get("text", "").strip()
                if len(text) > 500:
                    f.write(text + "\n\n")
                    count += 1
                if count >= 20000:
                    break
                if count % 2000 == 0:
                    print(f"  {count}/20000...")
        print(f"  [OK] books.txt — {count} books")
    except Exception as e:
        print(f"  [FAIL] {e} — trying bookcorpus fallback...")
        try:
            ds2 = load_dataset("bookcorpus/bookcorpus", split="train",
                               streaming=True, trust_remote_code=True)
            with open(PATH, "w", encoding="utf-8") as f:
                count = 0
                for row in ds2:
                    f.write(row["text"] + "\n")
                    count += 1
                    if count >= 500000:
                        break
            print(f"  [OK fallback] books.txt — {count} lines")
        except Exception as e2:
            print(f"  [FAIL fallback] {e2}")

# ── 4. CodeAlpaca ─────────────────────────────────────────────
PATH = "raw_data/codealpaca.txt"
if already_exists(PATH, min_mb=5):
    print(f"[SKIP] {PATH} already exists")
else:
    print("Downloading CodeAlpaca (20k examples)...")
    try:
        ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
        with open(PATH, "w", encoding="utf-8") as f:
            for row in ds:
                inst = row["instruction"].strip()
                inp = row["input"].strip()
                out = row["output"].strip()
                if inp:
                    text = f"### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n{out}"
                else:
                    text = f"### Instruction:\n{inst}\n\n### Response:\n{out}"
                f.write(text + "\n\n")
        print(f"  [OK] codealpaca.txt — {len(ds)} examples")
    except Exception as e:
        print(f"  [FAIL] {e}")

# ── 5. GSM8K Math Reasoning ───────────────────────────────────
PATH = "raw_data/gsm8k.txt"
if already_exists(PATH, min_mb=1):
    print(f"[SKIP] {PATH} already exists")
else:
    print("Downloading GSM8K (8.5k problems)...")
    try:
        ds = load_dataset("openai/gsm8k", "main", split="train")
        with open(PATH, "w", encoding="utf-8") as f:
            for row in ds:
                text = f"### Instruction:\n{row['question']}\n\n### Response:\n{row['answer']}"
                f.write(text + "\n\n")
        print(f"  [OK] gsm8k.txt — {len(ds)} problems")
    except Exception as e:
        print(f"  [FAIL] {e}")

# ── 6. Wikipedia upgrade 60k → 500k ──────────────────────────
wiki_mb = os.path.getsize("raw_data/wikipedia.txt") / (1024*1024) if os.path.exists("raw_data/wikipedia.txt") else 0
if wiki_mb > 400:
    print(f"[SKIP] wikipedia.txt already large ({wiki_mb:.0f}MB)")
else:
    print(f"Upgrading Wikipedia 60k → 500k articles...")
    try:
        ds = load_dataset("wikimedia/wikipedia", "20231101.en",
                          split="train", streaming=True, trust_remote_code=True)
        with open("raw_data/wikipedia.txt", "w", encoding="utf-8") as f:
            count = 0
            for row in ds:
                if len(row["text"].strip()) > 200:
                    f.write(row["text"] + "\n\n")
                    count += 1
                if count >= 500000:
                    break
                if count % 50000 == 0:
                    print(f"  {count}/500000...")
        print(f"  [OK] wikipedia.txt — {count} articles")
    except Exception as e:
        print(f"  [FAIL] {e}")

# ── SUMMARY ───────────────────────────────────────────────────
print("\n" + "="*50)
print("raw_data/ final state:")
for fname in sorted(os.listdir("raw_data")):
    path = f"raw_data/{fname}"
    mb = os.path.getsize(path) / (1024*1024)
    print(f"  {fname:35s} {mb:7.1f} MB")
print("\nNext: python build_dataset_v4.py")
