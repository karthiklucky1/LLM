import math
import os

import numpy as np
import torch

from model import MiniGPT


device = "cuda" if torch.cuda.is_available() else "cpu"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def resolve_checkpoint():
    path = os.environ.get("BASE_CKPT", "best.pt")
    if os.path.exists(path):
        return path
    raise FileNotFoundError(f"Base checkpoint not found: {path}")


def load_state_dict(path, device):
    ckpt = torch.load(path, map_location=device)
    return ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt


def load_compatible_weights(model, state_dict):
    model_state = model.state_dict()
    compatible = {
        key: value
        for key, value in state_dict.items()
        if key in model_state and value.shape == model_state[key].shape
    }
    skipped = sorted(set(state_dict) - set(compatible))
    model_state.update(compatible)
    model.load_state_dict(model_state)
    return compatible, skipped


def get_batch(data, batch_size, seq_len, device):
    max_start = len(data) - seq_len - 1
    if max_start <= 0:
        raise ValueError(
            f"Dataset is too small for seq_len={seq_len}. Need more than {seq_len + 1} tokens."
        )
    ix = torch.randint(max_start, (batch_size,))
    x = torch.stack([
        torch.from_numpy(data[i:i + seq_len].astype(np.int64))
        for i in ix
    ])
    y = torch.stack([
        torch.from_numpy(data[i + 1:i + seq_len + 1].astype(np.int64))
        for i in ix
    ])
    return x.to(device), y.to(device)


def get_lr(step, total_steps, max_lr, min_lr, warmup_steps):
    if step < warmup_steps:
        return max_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, (total_steps - warmup_steps))
    progress = min(max(progress, 0.0), 1.0)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def main():
    train_data = "datasets/train_instr.bin"
    val_data = "datasets/val_instr.bin"

    batch_size = 8
    grad_accum = 8
    steps = 8000
    eval_every = 500
    seq_len = 1024

    max_lr = 2e-5
    min_lr = 2e-6
    warmup_steps = 200

    model_config = dict(
        n_layer=12,
        n_head=12,
        n_embd=768,
        block_size=1024,
        vocab_size=32000,
        dropout=0.1,
    )

    checkpoint_path = resolve_checkpoint()

    if not os.path.exists(train_data) or not os.path.exists(val_data):
        raise FileNotFoundError(
            "Instruction dataset not found. Run build_instruction_dataset.py first."
        )

    print(f"device: {device}")
    print(f"base checkpoint: {checkpoint_path}")

    model = MiniGPT(**model_config).to(device)
    state_dict = load_state_dict(checkpoint_path, device)
    compatible, skipped = load_compatible_weights(model, state_dict)
    print(f"loaded tensors: {len(compatible)}/{len(model.state_dict())}")
    if skipped:
        print(f"skipped tensors: {len(skipped)}")
    model.train()

    train = np.memmap(train_data, dtype=np.uint16, mode="r")
    val = np.memmap(val_data, dtype=np.uint16, mode="r")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=max_lr,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    best_val = float("inf")

    for step in range(steps):
        lr = get_lr(step, steps, max_lr, min_lr, warmup_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        train_loss_total = 0.0

        for _ in range(grad_accum):
            x, y = get_batch(train, batch_size, seq_len, device)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
                _, loss, _ = model(x, y)
                train_loss_total += loss.item()
                loss = loss / grad_accum
            scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        if step % 100 == 0:
            print(
                f"step {step:6d} | lr {lr:.2e} | train {(train_loss_total / grad_accum):.4f}",
                flush=True,
            )

        if step % eval_every == 0:
            model.eval()
            with torch.no_grad():
                x_val, y_val = get_batch(val, batch_size, seq_len, device)
                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
                    _, val_loss, _ = model(x_val, y_val)
                    val_loss = val_loss.item()
            model.train()

            print(f"step {step:6d} | val {val_loss:.4f}", flush=True)

            if val_loss < best_val:
                best_val = val_loss
                torch.save(
                    {
                        "model": model.state_dict(),
                        "step": step,
                        "val_loss": val_loss,
                        "base_checkpoint": checkpoint_path,
                        "config": model_config,
                        "batch_size": batch_size,
                        "grad_accum": grad_accum,
                        "steps": steps,
                        "max_lr": max_lr,
                        "min_lr": min_lr,
                        "warmup_steps": warmup_steps,
                    },
                    "best_finetuned.pt",
                )
                print("saved new best checkpoint -> best_finetuned.pt", flush=True)

    print(f"done | best val: {best_val:.4f}")


if __name__ == "__main__":
    main()
