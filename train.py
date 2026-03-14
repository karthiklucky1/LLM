import math
import os
import datetime
from datetime import datetime
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import sentencepiece as spm

# from bpe_dataset import BinaryDataset
from model import MiniGPT


class BinaryDataset(Dataset):
    def __init__(self, bin_file, block_size):
        self.block_size = block_size
        self.tokens = np.memmap(bin_file, dtype=np.uint16, mode="r")

    def __len__(self):
        return len(self.tokens) - self.block_size

    def __getitem__(self, idx):
        x = torch.tensor(self.tokens[idx:idx + self.block_size], dtype=torch.long)
        y = torch.tensor(self.tokens[idx + 1:idx + self.block_size + 1], dtype=torch.long)
        return x, y


# ------------------------------------------------
# Evaluation
# ------------------------------------------------

@torch.no_grad()
def evaluate(model, dataloader, device, max_batches=20):
    model.eval()
    losses = []

    for i, (x, y) in enumerate(dataloader):
        if i >= max_batches:
            break

        x = x.to(device)
        y = y.to(device)

        _, loss, _ = model(x, y)
        losses.append(loss.item())

    model.train()

    if not losses:
        return float("nan")

    return sum(losses) / len(losses)


# ------------------------------------------------
# Cosine LR schedule
# ------------------------------------------------

def get_lr(step, max_lr, min_lr, warmup_steps, total_steps):
    if step < warmup_steps:
        return max_lr * step / warmup_steps

    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))

    return min_lr + (max_lr - min_lr) * cosine_decay


# ------------------------------------------------
# Training
# ------------------------------------------------

def train(
    block_size=256,
    batch_size=4,
    # steps=50000,
    # max_lr=2e-4,
    # min_lr=2e-5,
    steps=60000,
    max_lr=6e-5,
    min_lr=2e-6,
    grad_accum=8,
    # warmup_steps=2000,
    warmup_steps=500,
    run_name=None,
    eval_every=2000,
    sample_every=2000,
    save_every=5000,
):

    print("Starting training...")

    torch.set_float32_matmul_precision("high")

    # device selection
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print("Device:", device)

    # if run_name is None:
    #     run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    # checkpoint_dir = Path("checkpoints") / run_name
    # checkpoint_dir.mkdir(parents=True, exist_ok=True)
    # print("Checkpoint dir:", checkpoint_dir)
    if run_name is None:
        # run_name = f"run_v3_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_name = f"run_v3_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    checkpoint_dir = f"checkpoints/{run_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Checkpoints will be saved to: {checkpoint_dir}")

    tokenizer = spm.SentencePieceProcessor(model_file="tokenizer/mix.model")

    # ------------------------------------------------
    # Dataset
    # ------------------------------------------------

    # train_dataset = BinaryDataset("datasets/train_v2.bin", block_size)
    # val_dataset = BinaryDataset("datasets/val_v2.bin", block_size)
    train_data = "datasets/train_v3.bin"
    val_data = "datasets/val_v3.bin"
    train_dataset = BinaryDataset(train_data, block_size)
    val_dataset = BinaryDataset(val_data, block_size)

    print("Loaded dataset")
    print("Train size:", len(train_dataset))
    print("Val size:", len(val_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=0
    )

    # ------------------------------------------------
    # Model
    # ------------------------------------------------

    model = MiniGPT(
        vocab_size=8000,
        block_size=block_size,
        n_layer=12,
        n_head=12,
        n_embd=768,
        dropout=0.1
    ).to(device)

    # Disabled on Mac/MPS because compile can increase startup/step latency.
    # if hasattr(torch, "compile"):
    #     model = torch.compile(model)

    # ── RESUME FROM CHECKPOINT ──────────────────────────────────────
    RESUME_CHECKPOINT = "checkpoints/run_20260310_234758/minigpt_step_40000.pt"
    # Set to None to start fresh: RESUME_CHECKPOINT = None

    if RESUME_CHECKPOINT and os.path.exists(RESUME_CHECKPOINT):
        print(f"Resuming model weights from: {RESUME_CHECKPOINT}")
        ckpt = torch.load(RESUME_CHECKPOINT, map_location=device)

        # Load model weights only — not optimizer state
        # (optimizer state from old run has stale momentum for old data)
        if "model" in ckpt:
            model.load_state_dict(ckpt["model"])
        else:
            model.load_state_dict(ckpt)  # some checkpoints save weights directly

        print("  Model weights loaded successfully")
        del ckpt  # free memory immediately
    else:
        print("Starting fresh training (no checkpoint resume)")

    # ------------------------------------------------
    # Optimizer
    # ------------------------------------------------

    try:
        optim = torch.optim.AdamW(model.parameters(), lr=max_lr, fused=True)
        print("Using fused AdamW")
    except Exception:
        optim = torch.optim.AdamW(model.parameters(), lr=max_lr)
        print("Using standard AdamW")

    # ------------------------------------------------
    # Training loop
    # ------------------------------------------------

    total_steps = steps
    print("=" * 60)
    print("MiniGPT v3 Continued Training")
    print(f"  Resumed from:    {RESUME_CHECKPOINT}")
    print(f"  New dataset:     {train_data}")
    print(f"  Steps:           {total_steps}")
    print(f"  Max LR:          {max_lr}")
    print(f"  Warmup steps:    {warmup_steps}")
    print(f"  Checkpoint dir:  {checkpoint_dir}")
    print("=" * 60)
    print("Expected val loss progression:")
    print("  Steps 0-1k:   may spike slightly (new data format) — normal")
    print("  Steps 1k-5k:  should drop below 2.10")
    print("  Steps 10k+:   QA format responses should appear in samples")
    print("=" * 60)

    model.train()
    print("Entering training loop")

    data_iter = iter(train_loader)

    for step in range(1, steps + 1):

        optim.zero_grad(set_to_none=True)
        loss_accum = 0.0

        for _ in range(grad_accum):

            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                x, y = next(data_iter)

            x = x.to(device)
            y = y.to(device)

            _, loss, _ = model(x, y)
            loss_accum += loss.item()

            (loss / grad_accum).backward()

        loss_value = loss_accum / grad_accum

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        lr = get_lr(step, max_lr, min_lr, warmup_steps, steps)

        for param_group in optim.param_groups:
            param_group["lr"] = lr

        optim.step()

        # ------------------------------------------------
        # Logging
        # ------------------------------------------------

        if step % eval_every == 0:
            val_loss = evaluate(model, val_loader, device)
            print(
                f"step {step} | lr {lr:.6f} | train loss {loss_value:.4f} | val loss {val_loss:.4f}"
            )

        # ------------------------------------------------
        # Sample generation
        # ------------------------------------------------

        if step % sample_every == 0:

            model.eval()

            prompt = "Hello"
            idx = torch.tensor([tokenizer.encode(prompt)], device=device)

            out = model.generate(
                idx,
                max_new_tokens=120,
                temperature=0.9,
                top_k=80
            )

            print("---- sample ----")
            print(tokenizer.decode(out[0].tolist()))
            print("----------------")

            model.train()

        if step % save_every == 0:
            print(f"Saving checkpoint at step {step}...")
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optim.state_dict(),
                    "step": step,
                    "tokenizer_model": "tokenizer/mix.model",
                    "vocab_size": 8000,
                    "block_size": block_size,
                    "n_layer": 12,
                    "n_head": 12,
                    "n_embd": 768,
                },
                Path(checkpoint_dir) / f"minigpt_step_{step}.pt"
            )

    # ------------------------------------------------
    # Save final model
    # ------------------------------------------------

    print("Saving model...")

    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optim.state_dict(),
            "step": steps,
            "tokenizer_model": "tokenizer/mix.model",
            "vocab_size": 8000,
            "block_size": block_size,
            "n_layer": 12,
            "n_head": 12,
            "n_embd": 768,
        },
        Path(checkpoint_dir) / "minigpt.pt"
    )

    print(f"Saved final checkpoint: {Path(checkpoint_dir) / 'minigpt.pt'}")


if __name__ == "__main__":
    train()
