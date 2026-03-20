from model import MiniGPT
import torch
import math
import os
import numpy as np

# ----------------------------
# basic setup
# ----------------------------
device = "cuda"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

train_data = "datasets/train_v5.bin"
val_data   = "datasets/val_v5.bin"

batch_size   = 8
grad_accum   = 8
total_steps  = 150000

max_lr       = 2e-4
min_lr       = 2e-5
warmup_steps = 2000

eval_every   = 2000
save_path    = "best.pt"
seq_len      = 1024

model_config = dict(
    n_layer=12,
    n_head=12,
    n_embd=768,
    block_size=1024,
    vocab_size=32000,
    dropout=0.1,
)

# ----------------------------
# model / data / optimizer
# ----------------------------
model = MiniGPT(**model_config).to(device)
model.train()

train = np.memmap(train_data, dtype=np.uint16, mode="r")
val   = np.memmap(val_data,   dtype=np.uint16, mode="r")

opt = torch.optim.AdamW(
    model.parameters(),
    lr=max_lr,
    betas=(0.9, 0.95),
    weight_decay=0.1,
)

scaler = torch.amp.GradScaler("cuda")

# ----------------------------
# helpers
# ----------------------------
def get_lr(step: int) -> float:
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    progress = (step - warmup_steps) / max(1, (total_steps - warmup_steps))
    progress = min(max(progress, 0.0), 1.0)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))

def get_batch(data):
    ix = torch.randint(len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([
        torch.from_numpy(data[i:i+seq_len].astype(np.int64))
        for i in ix
    ])
    y = torch.stack([
        torch.from_numpy(data[i+1:i+seq_len+1].astype(np.int64))
        for i in ix
    ])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)

def model_forward_and_loss(model, x, y):
    logits, loss, _ = model(x, targets=y)
    return logits, loss
@torch.no_grad()
def evaluate(num_batches: int = 10) -> float:
    model.eval()
    total = 0.0
    for _ in range(num_batches):
        x, y = get_batch(val)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            _, loss = model_forward_and_loss(model, x, y)
        total += loss.item()
    model.train()
    return total / num_batches

# ----------------------------
# startup sanity check
# ----------------------------
print("=" * 60)
print("Training config")
print(f"device        = {device}")
print(f"batch_size    = {batch_size}")
print(f"grad_accum    = {grad_accum}")
print(f"effective_bs  = {batch_size * grad_accum}")
print(f"block_size    = {model_config['block_size']}")
print(f"vocab_size    = {model_config['vocab_size']}")
print(f"max_lr        = {max_lr}")
print(f"min_lr        = {min_lr}")
print("=" * 60)

# optional quick sanity check before long training
x0, y0 = get_batch(train)
with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
    logits0, loss0 = model_forward_and_loss(model, x0, y0)
print(f"startup sanity | logits shape = {tuple(logits0.shape)} | loss = {loss0.item():.4f}", flush=True)

best = 1e9

# ----------------------------
# training loop
# ----------------------------
for step in range(total_steps):
    lr = get_lr(step)
    for g in opt.param_groups:
        g["lr"] = lr

    opt.zero_grad(set_to_none=True)
    total_loss = 0.0

    for _ in range(grad_accum):
        x, y = get_batch(train)

        with torch.amp.autocast("cuda", dtype=torch.float16):
            _, loss = model_forward_and_loss(model, x, y)
            loss = loss / grad_accum

        scaler.scale(loss).backward()
        total_loss += loss.item() * grad_accum

    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt)
    scaler.update()

    if step % 100 == 0:
        print(f"step {step:6d} | lr {lr:.2e} | train {total_loss:.4f}", flush=True)

    if step % eval_every == 0:
        vloss_avg = evaluate(num_batches=10)
        print(f"step {step:6d} | val {vloss_avg:.4f}", flush=True)

        if vloss_avg < best:
            best = vloss_avg
            torch.save(
                {
                    "model": model.state_dict(),
                    "step": step,
                    "val_loss": vloss_avg,
                    "config": model_config,
                    "batch_size": batch_size,
                    "grad_accum": grad_accum,
                    "max_lr": max_lr,
                    "min_lr": min_lr,
                    "warmup_steps": warmup_steps,
                },
                save_path,
            )
            print(f"saved new best checkpoint -> {save_path}", flush=True)
