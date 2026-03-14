import argparse
from pathlib import Path
import torch
import sentencepiece as spm
import re

from model import MiniGPT


def normalize_state_dict(state_dict):
    if any(k.startswith("_orig_mod.") for k in state_dict):
        return {k.replace("_orig_mod.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def infer_config_from_state_dict(state_dict, ckpt):
    config = {}

    tok_emb = state_dict.get("tok_emb.weight")
    if tok_emb is not None:
        config["vocab_size"] = tok_emb.shape[0]
        config["n_embd"] = tok_emb.shape[1]

    mask = state_dict.get("blocks.0.attn.mask")
    if mask is not None:
        config["block_size"] = mask.shape[-1]

    layer_indices = set()
    for key in state_dict:
        m = re.match(r"blocks\.(\d+)\.", key)
        if m:
            layer_indices.add(int(m.group(1)))
    if layer_indices:
        config["n_layer"] = max(layer_indices) + 1

    config["n_head"] = ckpt.get("n_head", 8)
    config["vocab_size"] = ckpt.get(
        "vocab_size", config.get("vocab_size", 8000))
    config["block_size"] = ckpt.get(
        "block_size", config.get("block_size", 128))
    config["n_layer"] = ckpt.get("n_layer", config.get("n_layer", 8))
    config["n_embd"] = ckpt.get("n_embd", config.get("n_embd", 512))

    return config


def load_model(
    checkpoint_path="minigpt.pt",
    n_layer_override=None,
    n_head_override=None,
    n_embd_override=None,
):
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = normalize_state_dict(ckpt.get("model", ckpt))
    cfg = infer_config_from_state_dict(state_dict, ckpt)
    if n_layer_override is not None:
        cfg["n_layer"] = n_layer_override
    if n_head_override is not None:
        cfg["n_head"] = n_head_override
    if n_embd_override is not None:
        cfg["n_embd"] = n_embd_override

    model = MiniGPT(
        vocab_size=cfg["vocab_size"],
        block_size=cfg["block_size"],
        n_layer=cfg["n_layer"],
        n_head=cfg["n_head"],
        n_embd=cfg["n_embd"],
        dropout=0.0,
    ).to(device)

    model.load_state_dict(state_dict)
    model.eval()

    tokenizer_path = ckpt.get("tokenizer_model", "tokenizer/mix.model")
    tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_path)

    return model, tokenizer, device


def find_latest_checkpoint():
    ckpts = sorted(Path("checkpoints").glob("**/*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if ckpts:
        return str(ckpts[0])
    return "minigpt.pt"


@torch.no_grad()
def generate_text(
    prompt,
    checkpoint_path="minigpt.pt",
    max_new_tokens=150,
    temperature=0.8,
    top_k=50,
    top_p=0.9,
    n_layer_override=None,
    n_head_override=None,
    n_embd_override=None,
):
    model, tokenizer, device = load_model(
        checkpoint_path,
        n_layer_override=n_layer_override,
        n_head_override=n_head_override,
        n_embd_override=n_embd_override,
    )

    idx = torch.tensor([tokenizer.encode(prompt)],
                       dtype=torch.long, device=device)

    out = model.generate(
        idx,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )

    return tokenizer.decode(out[0].tolist())


@torch.no_grad()
def generate_with_model(
    prompt,
    model,
    tokenizer,
    device,
    max_new_tokens=150,
    temperature=0.8,
    top_k=50,
    top_p=0.9,
):
    idx = torch.tensor([tokenizer.encode(prompt)],
                       dtype=torch.long, device=device)

    out = model.generate(
        idx,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )

    return tokenizer.decode(out[0].tolist())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=150)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    args = parser.parse_args()

    checkpoint_path = args.checkpoint or find_latest_checkpoint()
    print(f"Loading checkpoint: {checkpoint_path}")

    model, tokenizer, device = load_model(checkpoint_path=checkpoint_path)
    while True:
        prompt = input("Prompt: ").strip()
        if not prompt:
            break
        result = generate_with_model(
            prompt,
            model,
            tokenizer,
            device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        print("\n--- Output ---\n")
        print(result)
