#!/bin/bash
set -e
echo "=================================================="
echo "MiniGPT v4 RunPod Setup"
echo "=================================================="

cd /workspace

echo "[1/6] Installing packages..."
pip install sentencepiece datasets huggingface_hub -q

echo "[2/6] Cloning repo..."
git clone https://github.com/karthiklucky1/LLM.git
cd LLM

echo "[3/6] Downloading checkpoint from HuggingFace..."
mkdir -p /workspace/checkpoints
python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='karthiklucky/minigpt-checkpoints',
    filename='minigpt_step_40000.pt',
    local_dir='/workspace/checkpoints'
)"

echo "[4/6] Downloading v4 datasets (20-40 min)..."
python download_v4_datasets.py

echo "[5/6] Building v4 binary dataset..."
python build_dataset_v4.py
mkdir -p /workspace/datasets
cp datasets/train_v4.bin /workspace/datasets/train_v4.bin
cp datasets/val_v4.bin /workspace/datasets/val_v4.bin

echo "[6/6] Starting training..."
nohup python train_runpod.py > training.log 2>&1 &
echo "Training PID: $!"
echo ""
echo "Setup complete. Monitor: tail -f training.log"
