from model import MiniGPT
import torch,math,os
import numpy as np

device="cuda"

train_data="datasets/train_v5.bin"
val_data="datasets/val_v5.bin"

batch_size=32
grad_accum=8
total_steps=150000

max_lr=2e-4
min_lr=2e-5
warmup_steps=2000

model_config=dict(
    n_layer=12,
    n_head=12,
    n_embd=768,
    block_size=1024,
    vocab_size=32000,
    dropout=0.1
)

model=MiniGPT(**model_config).to(device)

train=np.memmap(train_data,dtype=np.uint16,mode='r')
val=np.memmap(val_data,dtype=np.uint16,mode='r')

opt=torch.optim.AdamW(model.parameters(),lr=max_lr)

def lr(step):
    if step<warmup_steps:
        return max_lr*step/warmup_steps
    progress=(step-warmup_steps)/(total_steps-warmup_steps)
    return min_lr+0.5*(max_lr-min_lr)*(1+math.cos(math.pi*progress))

def batch(data):
    ix=torch.randint(len(data)-1024,(batch_size,))
    x=torch.stack([torch.from_numpy(data[i:i+1024].astype(np.int64)) for i in ix])
    y=torch.stack([torch.from_numpy(data[i+1:i+1025].astype(np.int64)) for i in ix])
    return x.to(device),y.to(device)

best=1e9

for step in range(total_steps):

    opt.param_groups[0]['lr'] = lr(step)
    opt.zero_grad(set_to_none=True)

    total_loss = 0.0

    for _ in range(grad_accum):

        x, y = batch(train)

        logits = model(x)

        loss=torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1)
        )

        loss = loss / grad_accum   # ✅ FIXED

        loss.backward()

        total_loss += loss.item()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # ✅ IMPORTANT

    opt.step()

    if step % 100 == 0:
        print("step", step, "train", total_loss)

    # better validation
    if step % 2000 == 0:

        model.eval()
        vloss_total = 0.0

        with torch.no_grad():
            for _ in range(10):   # ✅ average over 10 batches
                x, y = batch(val)
                v = model(x)

                vloss = torch.nn.functional.cross_entropy(
                    v.view(-1, v.size(-1)),
                    y.view(-1)
                )

                vloss_total += vloss.item()

        vloss_avg = vloss_total / 10

        print("val", vloss_avg)

        model.train()

        if vloss_avg < best:
            best = vloss_avg

            torch.save({
                "model": model.state_dict(),
                "step": step,
                "val_loss": vloss_avg,
                "config": model_config
            }, "best.pt")
