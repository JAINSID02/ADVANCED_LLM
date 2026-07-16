from __future__ import annotations
import argparse
import torch
import torch.nn as nn
from pathlib import Path
torch.manual_seed(0)

# Reuse GPTModern from Part 3
import sys
from pathlib import Path as _P
sys.path.append(str(_P(__file__).resolve().parents[1]/'part_3'))
from model_modern import GPTModern # type: ignore
from dataset_sft import load_tiny_hf, load_training_data
from collator_sft import SFTCollator
from curriculum import LengthCurriculum

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=str, default='huggingface', help='huggingface (Alpaca + chitchat) or "chitchat_only"')
    p.add_argument('--alpaca_split', type=str, default='train[:3000]', help='HF slice string, e.g. train[:3000] or train (full)')
    p.add_argument('--no_chitchat', action='store_true', help='disable the hand-written greeting/identity dataset')
    p.add_argument('--chitchat_repeat', type=int, default=3, help='oversample factor for the small chitchat set')
    p.add_argument('--ckpt', type=str, required=False, help='base checkpoint to fine-tune from (e.g. part_4 pretrained model)')
    p.add_argument('--out', type=str, default='runs/sft')
    p.add_argument('--steps', type=int, default=200)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--block_size', type=int, default=256)
    p.add_argument('--n_layer', type=int, default=4)
    p.add_argument('--n_head', type=int, default=4)
    p.add_argument('--n_embd', type=int, default=256)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--bpe_dir', type=str, default='../part_4/runs/part4-demo/tokenizer') # assumes tokenizer exists from Part 4
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    # Load training data: Alpaca slice (+ hand-written chitchat set, unless disabled)
    if args.data == 'chitchat_only':
        from chitchat_data import load_chitchat
        items = load_chitchat(repeat=1)
    else:
        items = load_training_data(alpaca_split=args.alpaca_split,
                                    include_chitchat=not args.no_chitchat,
                                    chitchat_repeat=args.chitchat_repeat)

    # Print few samples
    print(f"Loaded {len(items)} SFT items. Few samples:")
    for it in items[:3]:
        print(f"PROMPT: {it.prompt}\nRESPONSE: {it.response}\n{'-'*40}")

    # Curriculum over (prompt,response)
    tuples = [(it.prompt , it.response) for it in items]
    cur = list(LengthCurriculum(tuples))
    print(len(cur))

    # Collator + model
    # peek at tokenizer vocab size before building collator/model
    _peek_col = SFTCollator(block_size = args.block_size , bpe_dir = args.bpe_dir)
    vocab_size = _peek_col.vocab_size

    # If resuming/fine-tuning from a base checkpoint, adopt ITS architecture
    # dims so the state_dict actually matches (CLI dims are ignored in that case).
    n_layer, n_head, n_embd, block_size = args.n_layer, args.n_head, args.n_embd, args.block_size
    ckpt = None
    if args.ckpt:
        ckpt = torch.load(args.ckpt , map_location = device)
        base_cfg = ckpt.get('config' , {})
        n_layer = base_cfg.get('n_layer', n_layer)
        n_head = base_cfg.get('n_head', n_head)
        n_embd = base_cfg.get('n_embd', n_embd)
        block_size = base_cfg.get('block_size', block_size)
        if base_cfg.get('vocab_size') and base_cfg['vocab_size'] != vocab_size:
            raise RuntimeError(
                f"Base checkpoint vocab_size ({base_cfg['vocab_size']}) != tokenizer vocab_size "
                f"({vocab_size}). Make sure --bpe_dir points at the SAME tokenizer used to "
                f"pretrain the base checkpoint.")
        print(f"Adopting architecture from base checkpoint: n_layer={n_layer}, n_head={n_head}, "
              f"n_embd={n_embd}, block_size={block_size}")

    # rebuild collator with the FINAL block_size (may differ from args.block_size
    # if we just adopted it from the base checkpoint above)
    col = SFTCollator(block_size = block_size , bpe_dir = args.bpe_dir)

    model = GPTModern(vocab_size = col.vocab_size , block_size = block_size ,
                      n_layer = n_layer , n_head = n_head , n_embd=n_embd,
                      use_rmsnorm = True , use_swiglu = True , rope =True).to(device)

    if ckpt is not None:
        model.load_state_dict(ckpt['model'])

    opt = torch.optim.AdamW(model.parameters() , lr = args.lr , betas =(0.9 , 0.95) , weight_decay = 0.1)
    model.train()

    # Simple loop (single machine). We just cycle curriculum to fill batches, for a few steps.

    step =0
    i = 0
    while step < args.steps:
        batch = cur[i:i+args.batch_size]
        if not batch:
            # restart curriculum
            # cur = list(LengthCurriculum(tuples));
            i=0
            continue

        xb , yb = col.collate(batch)
        xb ,yb = xb.to(device) , yb.to(device)

        logits , loss , _ = model(xb, yb)
        opt.zero_grad(set_to_none= True)
        loss.backward()
        opt.step()
        step +=1 ; i+= args.batch_size
        if step % 20 ==0:
            print(f"step {step}: loss={loss.item():.4f}")

    Path(args.out).mkdir(parents=True, exist_ok=True)
    cfg = {
        "vocab_size":col.vocab_size,
        "block_size":block_size,
        "n_layer": n_layer,
        "n_head":n_head,
        "n_embd":n_embd,
        "dropout":0.0,
        "use_rmsnorm":True,
        "use_swiglu":True,
        "rope":True,
        # tokenizer info (best-effort)
        "tokenizer_type":"byte" if col.vocab_size == 256 else "bpe",
        "tokenizer_dir":args.bpe_dir,
        }
    
    torch.save({'model':model.state_dict() , 'config':cfg} , str(Path(args.out)/'model_last.pt'))
    print(f"Saved SFT checkpoint to {args.out}/model_last.pt")

if __name__ == '__main__':
    main()