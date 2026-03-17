from model import MiniGPT
import torch
import math
import os
import numpy as np

device = "cuda"

train_data = "datasets/train_v5.bin"
val_data   = "datasets/val_v5.bin"

# safer for A6000 with block_size=1024
batch_size   = 8
grad_accum   = 8
total_steps  = 150000

max_lr       = 2e-4
min_lr       = 2e-5
warmup_steps = 2000

eval_every   = 2000
save_path    = "best.pt"

model_config = dict(
    n_layer=12,
    n_head=12,
    n_embd=768,
    block_size=1024,
    vocab_size=32000,
    dropout=0.1,
)

# memory fragmentation fix
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

model = MiniGPT(**model_config).to(device)
model.train()

train = np.memmap(train_data, dtype=np.uint16, mode="r")
val   = np.memmap(val_data,   dtype=np.uint16, mode="r")

opt = torch.optim.AdamW(model.parameters(), lr=max_lr, betas=(0.9, 0.95), weight_decay=0.1)

# AMP for speed + memory savings
scaler = torch.cuda.amp.GradScaler()

def get_lr(step: int) -> float:
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))

def get_batch(data):
    ix = torch.randint(len(data) - 1024 - 1, (batch_size,))
    x = torch.stack([
        torch.from_numpy(data[i:i+1024].astype(np.int64))
        for i in ix
    ])
    y = torch.stack([
        torch.from_numpy(data[i+1:i+1025].astype(np.int64))
        for i in ix
    ])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)

def forward_logits(x):
    out = model(x)
    if isinstance(out, tuple):
        return out[0]
    return out

@torch.no_grad()
def evaluate(num_batches: int = 10) -> float:
    model.eval()
    total = 0.0
    for _ in range(num_batches):
        x, y = get_batch(val)
        with torch.cuda.amp.autocast():
            logits = forward_logits(x)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1)
            )
        total += loss.item()
    model.train()
    return total / num_batches

best = 1e9

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

for step in range(total_steps):
    lr = get_lr(step)
    for g in opt.param_groups:
        g["lr"] = lr

    opt.zero_grad(set_to_none=True)
    total_loss = 0.0

    for _ in range(grad_accum):
        x, y = get_batch(train)

        with torch.cuda.amp.autocast():
            logits = forward_logits(x)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1)
            )
            loss = loss / grad_accum

        scaler.scale(loss).backward()
        total_loss += loss.item()

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
