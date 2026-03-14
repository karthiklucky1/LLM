# MiniGPT — Continual Knowledge LLM

Decoder-only GPT transformer (91M params) trained from scratch in PyTorch.
Includes a novel Dad Layer runtime verification architecture.

## Architecture
- 91M parameters (n_layer=12, n_head=12, n_embd=768)
- RoPE positional embeddings + RMSNorm + SwiGLU FFN
- SentencePiece BPE tokenizer (8k vocab)
- block_size=512

## Files
- model.py                 — transformer architecture
- train.py                 — Mac/local training
- train_runpod.py          — RunPod/cloud training (optimized)
- generate.py              — inference
- download_datasets.py     — v3 downloader
- download_v4_datasets.py  — v4 full downloader
- build_dataset.py         — tokenization pipeline
- build_dataset_v4.py      — v4 dataset builder
- test_model.py            — evaluation suite
- train_tokenizer.py       — tokenizer training
- runpod_setup.sh          — one-command RunPod setup

## Best checkpoint
Val loss: 2.0849 at step 40k
Download: HuggingFace [link TBD]

## Quick start
pip install torch sentencepiece datasets huggingface_hub
python download_v4_datasets.py
python build_dataset_v4.py
python train_runpod.py
