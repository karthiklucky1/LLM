import sentencepiece as spm
from datasets import load_dataset

# Load TinyStories
ds = load_dataset("roneneldan/TinyStories")
text = "\n".join(ds["train"]["text"][:50000])  # limit for speed

# Save raw text temporarily
with open("tinystories.txt", "w", encoding="utf-8") as f:
    f.write(text)

# Train SentencePiece tokenizer
spm.SentencePieceTrainer.train(
    input="tinystories.txt",
    model_prefix="tinystories_sp",
    vocab_size=8000,
    model_type="bpe",
    character_coverage=1.0
)

print("Tokenizer training complete.")
