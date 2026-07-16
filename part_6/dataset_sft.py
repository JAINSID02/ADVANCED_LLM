from __future__ import annotations
from typing import List  ,Dict , Tuple
from dataclasses import dataclass
import os
import traceback

try :
    from datasets import load_dataset

except Exception:
    print("Couldn't import `datasets`. Will use fallback data only.")
    load_dataset = None

from formatters import Example

@dataclass
class SFTItem:
    prompt : str
    response : str

def load_tiny_hf(split : str = "train[:200]" , sample_dataset : bool = False) ->List[SFTItem]:
    """Try to load a tiny instruction dataset from HF; fall back to a baked-in list.
    We use `tatsu-lab/alpaca` as a familiar schema (instruction, input, output) and keep only a slice.
    """

    items : List[SFTItem] = []
    if load_dataset is not None and not sample_dataset:
        try:
            ds = load_dataset("tatsu-lab/alpaca", split=split)
            for row in ds:
                instr = row.get("instruction" , "").strip()
                inp = row.get("input","").strip()
                out = row.get("output","").strip()

                if inp:
                    instr = instr + "\n" + inp

                if instr and out:
                    items.append(SFTItem(prompt = instr , response = out))

        except Exception as e:
            print("Failed to load Hugging Face dataset:")
            traceback.print_exc()

    if not items:
        # fallback tiny dataset

        seeds = [("First prime number", "2"),
                 ("What are the three primary colors?", "red"),
                 ("Device name which points to direction?", "compass"),]
        
        items = [SFTItem(prompt = p , response = r) for p  , r in seeds]


    return items


def load_training_data(alpaca_split: str = "train[:3000]", include_chitchat: bool = True,
                        chitchat_repeat: int = 3) -> List[SFTItem]:
    """Combined SFT data source: a (configurable-size) slice of Alpaca for general
    instruction-following + a small hand-written chit-chat/identity set (6.x),
    oversampled by `chitchat_repeat` so it isn't drowned out by Alpaca's volume.

    This is what train_sft.py should call instead of the old hardcoded
    `load_tiny_hf(split='train[:24]')`.
    """
    items = load_tiny_hf(split=alpaca_split, sample_dataset=False)

    if include_chitchat:
        from chitchat_data import load_chitchat
        items = items + load_chitchat(repeat=chitchat_repeat)

    return items