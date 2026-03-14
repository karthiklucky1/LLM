import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------
# RoPE (Rotary Positional Embeddings)
# ------------------------------------------------

def build_rope_cache(seq_len, head_dim, device, base=10000, start_pos=0):
    half = head_dim // 2
    inv_freq = 1.0 / \
        (base ** (torch.arange(0, half, device=device).float() / half))
    positions = torch.arange(start_pos, start_pos + seq_len, device=device).float()

    freqs = torch.einsum("t,f->tf", positions, inv_freq)

    cos = torch.cos(freqs)[None, None, :, :]
    sin = torch.sin(freqs)[None, None, :, :]

    return cos, sin


def apply_rope(x, cos, sin):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]

    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos

    return torch.cat([out1, out2], dim=-1)


# ------------------------------------------------
# RMSNorm
# ------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(rms + self.eps)
        return self.weight * x


# ------------------------------------------------
# Self Attention
# ------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout=0.1):
        super().__init__()

        assert n_embd % n_head == 0

        self.n_head = n_head
        self.head_dim = n_embd // n_head

        assert self.head_dim % 2 == 0, "RoPE requires even head_dim"

        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)

        self.dropout = nn.Dropout(dropout)

        mask = torch.tril(torch.ones(block_size, block_size, dtype=torch.bool))
        self.register_buffer("mask", mask)

    def forward(self, x, kv_cache=None):
        B, T, C = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        past_len = 0 if kv_cache is None else kv_cache[0].shape[2]
        cos, sin = build_rope_cache(T, self.head_dim, x.device, start_pos=past_len)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        new_cache = (k, v)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        T_k = k.shape[2]
        row_start = T_k - T
        if T_k <= self.mask.shape[1]:
            causal = self.mask[row_start:row_start + T, :T_k]
        else:
            q_pos = torch.arange(row_start, row_start + T, device=x.device).unsqueeze(1)
            k_pos = torch.arange(T_k, device=x.device).unsqueeze(0)
            causal = k_pos <= q_pos
        att = att.masked_fill(~causal, float("-inf"))

        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        out = self.proj(out)

        return self.dropout(out), new_cache


# ------------------------------------------------
# SwiGLU Feedforward
# ------------------------------------------------

class SwiGLU(nn.Module):
    def __init__(self, n_embd, dropout=0.1):
        super().__init__()

        hidden_dim = int((8 / 3) * n_embd)
        hidden_dim = ((hidden_dim + 255) // 256) * 256  # round for efficiency

        self.w1 = nn.Linear(n_embd, hidden_dim, bias=False)
        self.w2 = nn.Linear(n_embd, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, n_embd, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.silu(self.w1(x)) * self.w2(x)
        x = self.w3(x)
        return self.dropout(x)


# ------------------------------------------------
# Transformer Block
# ------------------------------------------------

class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout=0.1):
        super().__init__()

        self.norm1 = RMSNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)

        self.norm2 = RMSNorm(n_embd)
        self.mlp = SwiGLU(n_embd, dropout)

    def forward(self, x, kv_cache=None):
        attn_out, new_cache = self.attn(self.norm1(x), kv_cache)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, new_cache


# ------------------------------------------------
# MiniGPT
# ------------------------------------------------

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_layer=4, n_head=4, n_embd=256, dropout=0.1):
        super().__init__()

        self.block_size = block_size

        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            Block(n_embd, n_head, block_size, dropout)
            for _ in range(n_layer)
        ])

        self.norm_f = RMSNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        # weight tying (very important for language models)
        self.lm_head.weight = self.tok_emb.weight

    def forward(self, idx, targets=None, kv_cache=None):
        B, T = idx.shape

        if T > self.block_size:
            raise ValueError("Sequence too long")

        x = self.tok_emb(idx)
        x = self.drop(x)

        new_caches = []
        for i, blk in enumerate(self.blocks):
            cache = None if kv_cache is None else kv_cache[i]
            x, new_cache = blk(x, cache)
            new_caches.append(new_cache)

        x = self.norm_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )

        return logits, loss, new_caches

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None):
        kv_cache = [None] * len(self.blocks)
        logits, _, kv_cache = self(idx, kv_cache=kv_cache)

        for _ in range(max_new_tokens):
            logits = logits[:, -1, :]
            if temperature is not None and temperature <= 0:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
                idx = torch.cat([idx, next_id], dim=1)
                logits, _, kv_cache = self(next_id, kv_cache=kv_cache)
                continue

            logits = logits / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")

            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                probs = F.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(probs, dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits = logits.masked_fill(indices_to_remove, -float("inf"))

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
            logits, _, kv_cache = self(next_id, kv_cache=kv_cache)

        return idx
