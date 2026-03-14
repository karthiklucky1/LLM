# test_model.py
# Run: python test_model.py --checkpoint checkpoints/run_v3_<timestamp>/minigpt_step_60000.pt
# Or:  python test_model.py  (uses latest checkpoint automatically)

import torch
import argparse
import os
import glob
import sentencepiece as spm

# ── ARGS ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, default=None,
                    help="Path to checkpoint. If not set, finds latest automatically.")
parser.add_argument("--temperature", type=float, default=0.7)
parser.add_argument("--top_k", type=int, default=50)
parser.add_argument("--top_p", type=float, default=0.9)
parser.add_argument("--max_new_tokens", type=int, default=200)
args = parser.parse_args()

# ── FIND CHECKPOINT ──────────────────────────────────────────────────────────
def find_latest_checkpoint():
    all_ckpts = glob.glob("checkpoints/run_v3_*/minigpt_step_*.pt")
    if not all_ckpts:
        all_ckpts = glob.glob("checkpoints/*/minigpt_step_*.pt")
    if not all_ckpts:
        return None
    # Sort by step number
    def get_step(path):
        try:
            return int(path.split("_step_")[1].replace(".pt", ""))
        except Exception:
            return 0
    return max(all_ckpts, key=get_step)

checkpoint_path = args.checkpoint or find_latest_checkpoint()
if not checkpoint_path:
    print("ERROR: No checkpoint found. Run training first.")
    raise SystemExit(1)
print(f"Using checkpoint: {checkpoint_path}")

# ── LOAD MODEL ───────────────────────────────────────────────────────────────
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device}")

ckpt = torch.load(checkpoint_path, map_location=device)

# Import your model class (adjust import if needed)
try:
    from model import MiniGPT          # try model.py first
except ImportError:
    try:
        from train import MiniGPT      # some projects keep model in train.py
    except ImportError:
        print("ERROR: Cannot import MiniGPT. Make sure model.py exists.")
        raise SystemExit(1)

# Build model config from checkpoint or use defaults
# model_config = ckpt.get("config", {
model_config = ckpt.get("config", {
    "n_layer": ckpt.get("n_layer", 12),
    "n_head": ckpt.get("n_head", 12),
    "n_embd": ckpt.get("n_embd", 768),
    "block_size": ckpt.get("block_size", 256),
    "vocab_size": ckpt.get("vocab_size", 8000),
})
model = MiniGPT(**model_config)
state = ckpt["model"] if "model" in ckpt else ckpt
model.load_state_dict(state)
model.eval()
model.to(device)
print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

# ── LOAD TOKENIZER ───────────────────────────────────────────────────────────
sp = spm.SentencePieceProcessor()
sp.Load("tokenizer/mix.model")

# ── GENERATE FUNCTION ────────────────────────────────────────────────────────
@torch.no_grad()
def generate(prompt, max_new_tokens=200, temperature=0.7, top_k=50, top_p=0.9):
    tokens = sp.Encode(prompt, out_type=int)
    x = torch.tensor([tokens], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        x_cond = x[:, -model_config.get("block_size", 256):]
        # logits = model(x_cond)
        logits, _, _ = model(x_cond)
        logits = logits[:, -1, :] / temperature

        # Top-k filtering
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')

        # Top-p filtering
        if top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            cumprobs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_idx_to_remove = cumprobs - torch.softmax(sorted_logits, dim=-1) > top_p
            sorted_logits[sorted_idx_to_remove] = float('-inf')
            logits = torch.zeros_like(logits).scatter_(1, sorted_idx, sorted_logits)

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        x = torch.cat([x, next_token], dim=1)

        # Stop at EOS if model has it
        if next_token.item() == sp.eos_id():
            break

    output_tokens = x[0, len(tokens):].tolist()
    return sp.Decode(output_tokens)

# ── TEST SUITE ───────────────────────────────────────────────────────────────
tests = [
    {
        "category": "INSTRUCTION — Factual QA",
        "prompt": "### Instruction:\nWhat is the capital of France?\n\n### Response:\n",
        "expect": "Paris",
        "note": "Was broken before (said 'United States'). Should now say Paris."
    },
    {
        "category": "INSTRUCTION — Explanation",
        "prompt": "### Instruction:\nExplain what gravity is in simple terms.\n\n### Response:\n",
        "expect": "force / mass / pull / attract",
        "note": "Was broken before (generated random story). Should now explain gravity."
    },
    {
        "category": "INSTRUCTION — List generation",
        "prompt": "### Instruction:\nList 3 benefits of drinking water.\n\n### Response:\n",
        "expect": "1. / 2. / 3. or numbered list",
        "note": "Tests structured output format."
    },
    {
        "category": "CODE — Python function",
        "prompt": "### Instruction:\nWrite a Python function to reverse a string.\n\n### Response:\n",
        "expect": "def reverse",
        "note": "Should produce valid Python function, not prose."
    },
    {
        "category": "CODE — continuation",
        "prompt": "def fibonacci(n):\n    # Return the nth fibonacci number\n    ",
        "expect": "if n <= 1 / return",
        "note": "Tests raw code completion without instruction format."
    },
    {
        "category": "STORY — regression check",
        "prompt": "Once upon a time there was a little girl named Lily who loved",
        "expect": "narrative continuation",
        "note": "Model should still do stories (regression check — should NOT be broken)."
    },
    {
        "category": "FACTUAL — Wikipedia style",
        "prompt": "The Eiffel Tower is located in",
        "expect": "Paris / France",
        "note": "Basic factual recall."
    },
]

print("\n" + "=" * 60)
print("TEST RESULTS")
print("=" * 60)

passed = 0
for i, test in enumerate(tests):
    print(f"\n[{i+1}/{len(tests)}] {test['category']}")
    print(f"  Prompt: {test['prompt'][:80].strip()}...")
    print(f"  Expected keyword: {test['expect']}")
    print(f"  Note: {test['note']}")

    output = generate(
        test["prompt"],
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p
    )

    print(f"  Output: {output[:300].strip()}")

    # Simple keyword check
    expected_keywords = [k.strip().lower() for k in test["expect"].split("/")]
    found = any(kw in output.lower() for kw in expected_keywords)
    status = "PASS" if found else "CHECK"  # CHECK = manual review needed
    if found:
        passed += 1
    print(f"  Status: [{status}]")
    print("-" * 40)

print(f"\nAuto-passed: {passed}/{len(tests)}")
print("Note: 'CHECK' items need manual review — output may be correct but use different words")
print("\nComparison guide:")
print("  Before v3: Paris question → 'capital of the United States' (wrong)")
print("  After v3:  Paris question → 'Paris' or 'Paris, France' (correct)")
print("  Before v3: Gravity → random story")
print("  After v3:  Gravity → actual explanation of gravitational force")
