import datetime
import math
import os
from model import MiniGPT
import numpy as np
import torch.nn.functional as F
import torch

# train_runpod.py
# RunPod A6000 optimized — DO NOT run on Mac


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
assert device == "cuda", "This script requires CUDA. Use train.py for Mac."

# float16 instead of bfloat16 — fixes GradScaler error on A6000
dtype = torch.float16

model_config = dict(
    n_layer=12,
    n_head=12,
    n_embd=768,
    block_size=512,
    vocab_size=8000,
    dropout=0.1,
)

train_data = "/workspace/LLM/datasets/train_v4.bin"
val_data = "/workspace/LLM/datasets/val_v4.bin"
batch_size = 64
grad_accum = 4
total_steps = 100000
max_lr = 3e-4
min_lr = 3e-5
warmup_steps = 2000
eval_every = 500
sample_every = 1000
save_every = 5000
RESUME = "/workspace/LLM/checkpoints/minigpt_step_40000.pt"

run_name = f"runpod_v4_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
ckpt_dir = f"/workspace/LLM/checkpoints/{run_name}"
os.makedirs(ckpt_dir, exist_ok=True)


def get_batch(split):
    path = train_data if split == "train" else val_data
    data = np.memmap(path, dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - model_config["block_size"], (batch_size,))
    x = torch.stack([torch.from_numpy(
        data[i:i+model_config["block_size"]].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(
        data[i+1:i+model_config["block_size"]+1].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


def get_lr(step):
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


# Build model
model = MiniGPT(**model_config)

# Load checkpoint — compatible weights only (handles block_size 256→512 mismatch)
if RESUME and os.path.exists(RESUME):
    ckpt = torch.load(RESUME, map_location="cpu", weights_only=False)
    state = ckpt["model"] if "model" in ckpt else ckpt
    model_state = model.state_dict()
    compatible = {k: v for k, v in state.items()
                  if k in model_state and v.shape == model_state[k].shape}
    model_state.update(compatible)
    model.load_state_dict(model_state)
    print(
        f"Loaded {len(compatible)}/{len(model_state)} tensors from checkpoint")
    del ckpt
else:
    print("No checkpoint found — training from scratch")

model = model.to(device)
model = torch.compile(model)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=max_lr, betas=(0.9, 0.95),
    weight_decay=0.1, fused=True,
)

# GradScaler for float16
scaler = torch.amp.GradScaler("cuda")
best_val_loss = float("inf")
val_loss = float("inf")

print("="*60)
print(f"MiniGPT v4 RunPod Training")
print(f"  Steps:       {total_steps:,}")
print(
    f"  Batch:       {batch_size} x {grad_accum} = {batch_size*grad_accum} effective")
print(f"  Max LR:      {max_lr}")
print(f"  dtype:       float16")
print(f"  block_size:  {model_config['block_size']}")
print(f"  Checkpoint:  {ckpt_dir}")
print("="*60)

for step in range(total_steps):
    lr = get_lr(step)
    for g in optimizer.param_groups:
        g["lr"] = lr

    # Forward + backward with gradient accumulation
    optimizer.zero_grad()
    for _ in range(grad_accum):
        x, y = get_batch("train")
        with torch.autocast(device_type="cuda", dtype=dtype):
            # Handle model returning tuple or just logits
            out = model(x)
            logits = out[0] if isinstance(out, tuple) else out
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1)
            ) / grad_accum
        scaler.scale(loss).backward()

    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()

    # Eval
    if step % eval_every == 0:
        model.eval()
        with torch.no_grad():
            val_losses = []
            for _ in range(20):
                xv, yv = get_batch("val")
                with torch.autocast(device_type="cuda", dtype=dtype):
                    out_v = model(xv)
                    logits_v = out_v[0] if isinstance(out_v, tuple) else out_v
                    vl = F.cross_entropy(
                        logits_v.view(-1, logits_v.size(-1)),
                        yv.view(-1)
                    )
                val_losses.append(vl.item())
            val_loss = sum(val_losses) / len(val_losses)
        train_loss = loss.item() * grad_accum
        print(
            f"step {step:6d} | lr {lr:.2e} | train {train_loss:.4f} | val {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {"model": model.state_dict(), "step": step, "val_loss": val_loss},
                f"{ckpt_dir}/best.pt"
            )
        model.train()

    # Save checkpoint
    if step % save_every == 0 and step > 0:
        torch.save(
            {"model": model.state_dict(), "step": step, "val_loss": val_loss},
            f"{ckpt_dir}/step_{step}.pt"
        )

print(f"Done. Best val loss: {best_val_loss:.4f}")
print(f"Best checkpoint: {ckpt_dir}/best.pt")
