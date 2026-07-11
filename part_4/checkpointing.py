from __future__ import annotations
import os
from pathlib import Path
from typing import Any  , Dict , Optional  , Tuple
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]/'part 3'))
import time
import torch
import shutil
import torch.nn as nn

DEF_NAME = "model_last.pt"

# ----------------------------- TB-only helpers (safe no-ops otherwise) ----------------------------- #

def _is_tb(logger)->bool:
    return getattr(logger , "w" , None) is not None

# checkpointing._log_hparams_tb
def _log_hparams_tb(logger,args,total_steps):
    if not  _is_tb(logger): return
    try:
        h = dict(vocab_size=args.vocab_size,block_size=args.block_size,n_layer=args.n_layer ,
                 n_head=args.n_head,n_embd=args.n_embd,dropout=args.dropout,lr=args.lr,
                 warmup_steps=args.warmup_steps,batch_size=args.batch_size,grad_accum=args.grad_accum_steps,
                 mixed_precision=args.mixed_precision,steps=args.steps,epochs=args.epochs)
        
        logger.hparams(h, {"meta/total_steps": float(total_steps)})

    except Exception:
        pass

def _maybe_log_graph_tb(logger,model,xb,yb):
    if not hasattr(logger,"graph"):
        return
    
    try:
        class _TensorOnly(nn.Module):
            def __init__(self,m):
                super().__init__(); self.m=m.eval()

            def forward(self,x,y=None):
                out =self.m(x,y) if y is not None else self.m(x)
                if isinstance(out,(list,tuple)):
                    for o in out :
                        if torch.is_tensor(o):
                            return o
                        
                    return out[0]
                return out
            
        wrapped = _TensorOnly(model).to(xb.device)
        logger.graph(wrapped , (xb,yb))

    except Exception:
        pass

def _log_model_stats(logger , model  , step:int , do_hists : bool = False ):
    if not _is_tb(logger): return

    try:
        params = [p for p in model.parameters() if p.requires_grad]
        total_param_norm = torch.norm(torch.stack([p.detach().norm(2) for p in params]),2).item()
        grads = [p.grad for p in params if p.grad is not None]
        total_grad_norm = float('nan')
        if grads:
            total_grad_norm = torch.norm(torch.stack([g.detach().norm(2) for g in grads]) ,2).item()
        logger.log(step=step, **{
            "train/param_global_l2": total_param_norm,
            "train/grad_global_l2": total_grad_norm,
        })

        if do_hists:
            for  name,p in model.named_parameters():
                logger.hist(f"params/{name}",p,step)
                if p.grad is not None:
                    logger.hist(f"grads/{name}", p.grad, step)
    
    except Exception:
        pass
                    
