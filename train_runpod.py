# train_runpod.py
# RunPod A6000 optimized — DO NOT run on Mac
# Run: python train_runpod.py

import os, math, time, datetime
import torch
import numpy as np
from model import MiniGPT

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
assert device == "cuda", "This script requires CUDA. Use train.py for Mac."

dtype = torch.bfloat16

model_config = dict(
    n_layer    = 12,
    n_head     = 12,
    n_embd     = 768,
    block_size = 512,
    vocab_size = 8000,
    dropout    = 0.1,
)

train_data   = "datasets/train_v4.bin"
val_data     = "datasets/val_v4.bin"
batch_size   = 64
grad_accum   = 4
total_steps  = 100000
max_lr       = 3e-4
min_lr       = 3e-5
warmup_steps = 2000
eval_every   = 500
sample_every = 1000
save_every   = 5000
RESUME       = "checkpoints/minigpt_step_40000.pt"

run_name = f"runpod_v4_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
ckpt_dir = f"checkpoints/{run_name}"
os.makedirs(ckpt_dir, exist_ok=True)

def get_batch(split):
    data = np.memmap(train_data if split == "train" else val_data,
                     dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - model_config["block_size"], (batch_size,))
    x = torch.stack([torch.from_numpy(data[i  :i+model_config["block_size"]].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+model_config["block_size"]+1].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)

def get_lr(step):
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))

model = MiniGPT(**model_config)

if RESUME and os.path.exists(RESUME):
    ckpt = torch.load(RESUME, map_location="cpu")
    state = ckpt["model"] if "model" in ckpt else ckpt
    model_state = model.state_dict()
    compatible  = {k: v for k, v in state.items()
                   if k in model_state and v.shape == model_state[k].shape}
    model_state.update(compatible)
    model.load_state_dict(model_state)
    print(f"Loaded {len(compatible)}/{len(model_state)} tensors from checkpoint")
    del ckpt

model = model.to(device).to(dtype)
model = torch.compile(model)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=max_lr, betas=(0.9, 0.95),
    weight_decay=0.1, fused=True,
)

TEST_PROMPTS = [
    "### Instruction:\nWhat is the capital of France?\n\n### Response:\n",
    "### Instruction:\nExplain what gravity is.\n\n### Response:\n",
    "### Instruction:\nWrite a Python function to reverse a string.\n\n### Response:\n",
    "Once upon a time there was a",
]

scaler = torch.cuda.amp.GradScaler()
best_val_loss = float("inf")

print("="*60)
print(f"MiniGPT v4 RunPod Training")
print(f"  Steps:       {total_steps:,}")
print(f"  Batch:       {batch_size} x {grad_accum} = {batch_size*grad_accum} effective")
print(f"  Max LR:      {max_lr}")
print(f"  block_size:  {model_config['block_size']}")
print(f"  Checkpoint:  {ckpt_dir}")
print("="*60)

for step in range(total_steps):
    lr = get_lr(step)
    for g in optimizer.param_groups:
        g["lr"] = lr

    optimizer.zero_grad()
    for _ in range(grad_accum):
        x, y = get_batch("train")
        with torch.autocast(device_type="cuda", dtype=dtype):
            logits, loss = model(x, y)
            loss = loss / grad_accum
        scaler.scale(loss).backward()

    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()

    if step % eval_every == 0:
        model.eval()
        with torch.no_grad():
            val_losses = []
            for _ in range(20):
                xv, yv = get_batch("val")
                with torch.autocast(device_type="cuda", dtype=dtype):
                    _, vl = model(xv, yv)
                val_losses.append(vl.item())
            val_loss = sum(val_losses) / len(val_losses)
        print(f"step {step:6d} | lr {lr:.2e} | train {loss.item()*grad_accum:.4f} | val {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"model": model.state_dict(), "step": step, "val_loss": val_loss},
                       f"{ckpt_dir}/best.pt")
        model.train()

    if step % save_every == 0 and step > 0:
        torch.save({"model": model.state_dict(), "step": step, "val_loss": val_loss},
                   f"{ckpt_dir}/step_{step}.pt")

print(f"Done. Best val loss: {best_val_loss:.4f}")
print(f"Best checkpoint: {ckpt_dir}/best.pt")
