"""
chat.py — terminal chat interface for your Part 8 PPO policy.

Loads the RLHF-trained PolicyWithValue model + BPE tokenizer and gives you
an interactive, streaming (token-by-token) chat loop in the terminal.

Run from inside `part_8/`:
    python chat.py
    python chat.py --ckpt runs/ppo-demo/model_last.pt --temperature 0.7
    python chat.py --ckpt ../part_6/runs/sft-demo/model_last.pt   # chat with the pre-PPO SFT model instead

Type 'exit' or 'quit' (or Ctrl+C) to leave. Type 'reset' to clear conversation memory.
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import torch

from policy import PolicyWithValue

sys.path.append(str(Path(__file__).resolve().parents[1] / "part_4"))
from tokenizer_bpe import BPETokenizer  # type: ignore

sys.path.append(str(Path(__file__).resolve().parents[1] / "part_6"))
from formatters import format_prompt_only  # type: ignore


def load_model_and_tokenizer(ckpt_path: str, bpe_dir: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    # Different scripts in this repo saved the config under different keys
    # ('config' in train_ppo/train_sft, 'cfg' in some eval scripts) — support both.
    cfg = ckpt.get("config", ckpt.get("cfg", {}))

    tok = BPETokenizer(vocab_size=cfg.get("vocab_size", 8000))
    tok.load(bpe_dir)

    model = PolicyWithValue(
        vocab_size=cfg.get("vocab_size", tok.vocab_size),
        block_size=cfg.get("block_size", 256),
        n_layer=cfg.get("n_layer", 2),
        n_head=cfg.get("n_head", 2),
        n_embd=cfg.get("n_embd", 128),
    ).to(device)

    state_dict = ckpt["model"]
    # Reward-model / SFT-only checkpoints only have LM weights, not the value head.
    if any(k.startswith("lm.") for k in state_dict):
        model.load_state_dict(state_dict, strict=True)
    else:
        model.lm.load_state_dict(state_dict, strict=True)

    model.eval()
    return model, tok, cfg


@torch.no_grad()
def stream_generate(model, tok, prompt_ids, device, max_new_tokens=150,
                     temperature=0.7, top_k=50, top_p=None, eos_id=1, block_size=256):
    """Token-by-token generation that prints text as it's produced.
    Mirrors GPTModern.generate() but yields decoded text deltas instead of
    returning everything at once.
    """
    try:
        from utils import top_k_top_p_filtering as _filter
    except Exception:
        def _filter(x, **_):
            return x

    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    kvs = [None] * len(model.lm.blocks)
    produced_ids = []
    printed_text = ""

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:] if kvs[0] is None else idx[:, -1:]
        start_pos = 0 if kvs[0] is None else kvs[0].k.size(2)

        logits, _, kvs = model.lm(idx_cond, kv_cache_list=kvs, start_pos=start_pos)
        next_logits = logits[:, -1, :] / max(temperature, 1e-6)
        next_logits = _filter(next_logits, top_k=top_k, top_p=top_p)
        probs = torch.softmax(next_logits, dim=-1)
        next_id = (torch.argmax(probs, dim=-1, keepdim=True) if temperature == 0.0
                   else torch.multinomial(probs, 1))

        token_id = next_id.item()
        if eos_id is not None and token_id == eos_id:
            break

        produced_ids.append(token_id)
        idx = torch.cat([idx, next_id], dim=1)

        # Re-decode the whole generated-so-far span each step (safe for BPE,
        # where a single new token can change how prior bytes are merged/rendered)
        # and print only the new suffix.
        full_text = tok.decode(produced_ids)
        new_piece = full_text[len(printed_text):]
        if new_piece:
            print(new_piece, end="", flush=True)
            printed_text = full_text

    print()  # trailing newline
    return printed_text


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="runs/ppo-demo/model_last.pt",
                    help="Path to policy checkpoint (PPO by default; point at part_6 SFT ckpt to compare)")
    p.add_argument("--bpe_dir", type=str, default="../part_4/runs/part4-demo/tokenizer")
    p.add_argument("--max_tokens", type=int, default=150)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--top_p", type=float, default=None)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--context", action="store_true",
                    help="Include prior turns in the prompt (model was trained single-turn, so quality may vary)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Loading model from {args.ckpt} on {device} ...")
    model, tok, cfg = load_model_and_tokenizer(args.ckpt, args.bpe_dir, device)
    print(f"Ready. Config: {cfg}\n")
    print("Type 'exit' to quit, 'reset' to clear conversation memory.\n")

    history = ""  # only used if --context is passed

    while True:
        try:
            user_in = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_in:
            continue
        if user_in.lower() in ("exit", "quit"):
            print("Bye!")
            break
        if user_in.lower() == "reset":
            history = ""
            print("(conversation memory cleared)\n")
            continue

        if args.context:
            history += format_prompt_only(user_in).replace("</s>", "")
            prompt_text = history
        else:
            prompt_text = format_prompt_only(user_in).replace("</s>", "")

        prompt_ids = tok.encode(prompt_text)
        block_size = cfg.get("block_size", 256)
        prompt_ids = prompt_ids[-block_size:]

        print("Assistant: ", end="", flush=True)
        reply = stream_generate(
            model, tok, prompt_ids, device,
            max_new_tokens=args.max_tokens, temperature=args.temperature,
            top_k=args.top_k, top_p=args.top_p, block_size=block_size,
        )
        print()

        if args.context:
            history += reply + "</s>"


if __name__ == "__main__":
    main()