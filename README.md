# LLM From Scratch

A complete large language model — architecture, tokenizer, pretraining, supervised fine-tuning, reward modeling, and RLHF alignment — built entirely from first principles in PyTorch, with no pre-built model libraries (no Hugging Face `Trainer`, no `trl`, no off-the-shelf transformer blocks).

**[Live demo →](your-streamlit-link-here)** — chat with a version fine-tuned to answer questions about me.

---

## Table of Contents

- [Why I built this](#why-i-built-this)
- [What this project does](#what-this-project-does)
- [Repository structure](#repository-structure)
- [Pipeline, stage by stage](#pipeline-stage-by-stage)
  - [Part 1 — Core Transformer Architecture](#part-1--core-transformer-architecture)
  - [Part 2 — Training a Tiny LLM](#part-2--training-a-tiny-llm)
  - [Part 3 — Modernizing the Architecture](#part-3--modernizing-the-architecture)
  - [Part 4 — Scaling Up](#part-4--scaling-up)
  - [Part 5 — Mixture-of-Experts](#part-5--mixture-of-experts)
  - [Part 6 — Supervised Fine-Tuning](#part-6--supervised-fine-tuning)
  - [Part 7 — Reward Modeling](#part-7--reward-modeling)
  - [Part 8 — RLHF with PPO](#part-8--rlhf-with-ppo)
- [Model configuration](#model-configuration)
- [Datasets used](#datasets-used)
- [Engineering deep dives](#engineering-deep-dives)
  - [Diagnosing a memorized base model](#diagnosing-a-memorized-base-model)
  - [Diagnosing and fixing PPO mode collapse](#diagnosing-and-fixing-ppo-mode-collapse)
- [The "Ask About Me" demo](#the-ask-about-me-demo)
- [Running it yourself](#running-it-yourself)
- [Tech stack](#tech-stack)
- [Known limitations](#known-limitations)
- [Author](#author)

---

## Why I built this

Most "build an LLM" walkthroughs stop at pretraining a small model that can babble plausible-looking text. That leaves out the part of the stack that actually makes modern LLMs useful and safe to interact with: instruction tuning, preference modeling, and reinforcement learning from human feedback. I wanted to understand — and be able to explain in depth, not just recite — every stage between "raw transformer" and "a model you can actually have a conversation with," including the parts that don't work on the first try.

This repo is that full pipeline, implemented and debugged end to end by hand.

## What this project does

The project implements the same conceptual pipeline used to train models like ChatGPT and Claude, compressed to a scale that trains on a single machine:

```
raw architecture → tokenizer → base pretraining → supervised fine-tuning → reward model → RLHF (PPO) → chat interface
```

Each stage is a separate, runnable module with its own training script, so the effect of each stage can be inspected independently (e.g. comparing the pre-PPO and post-PPO checkpoints side by side).

## Repository structure

```
llm-from-scratch/
├── part_1/     # attention mechanics, transformer block (from-scratch math + PyTorch)
├── part_2/     # byte-tokenizer toy GPT, training loop, sampling
├── part_3/     # RoPE, RMSNorm, SwiGLU, KV-cache, sliding-window attention
├── part_4/     # BPE tokenizer, base pretraining, train/val pipeline
├── part_5/     # Mixture-of-Experts layer demos
├── part_6/     # supervised fine-tuning (instruction data + custom about-me data)
├── part_7/     # reward model training
├── part_8/     # PPO/RLHF training, chat.py, Streamlit demo (app.py)
├── requirements.txt
└── README.md
```

## Pipeline, stage by stage

### Part 1 — Core Transformer Architecture
- Implemented positional embeddings from scratch, comparing absolute learned embeddings against sinusoidal embeddings
- Derived and coded self-attention from first principles, working through the scaled dot-product computation by hand on a toy example before implementing it
- Built a single attention head, then extended it to multi-head attention (splitting into heads, concatenation, output projection)
- Implemented the feed-forward MLP sublayer (GELU activation, dimensionality expansion/contraction)
- Added residual connections and LayerNorm
- Assembled all of the above into a complete, stackable Transformer block

### Part 2 — Training a Tiny LLM
- Implemented byte-level tokenization as the first, simplest tokenizer
- Built the dataset pipeline: batching and shifting sequences for next-token prediction
- Implemented cross-entropy loss with correct label shifting
- Wrote the full training loop from scratch — no Hugging Face `Trainer`, no external training framework
- Implemented sampling strategies: temperature scaling, top-k, and top-p (nucleus) sampling
- Added validation-loss evaluation to track generalization during training

### Part 3 — Modernizing the Architecture
- Replaced LayerNorm with **RMSNorm**, comparing gradient behavior and convergence speed between the two
- Implemented **RoPE** (Rotary Positional Embeddings) from theory through code
- Replaced the standard MLP activation with **SwiGLU**
- Implemented **KV-caching** to avoid recomputing attention over the full sequence at every generation step
- Implemented **sliding-window attention** with an **attention sink**, and a **rolling-buffer KV cache** to support streaming generation over long sequences

### Part 4 — Scaling Up
- Replaced byte-level tokenization with a trained **BPE (Byte Pair Encoding)** tokenizer (8,000-token vocabulary)
- Added **gradient accumulation** and **mixed-precision** training to fit larger effective batch sizes
- Implemented learning-rate scheduling with warmup
- Added checkpointing and resume support, including safe interrupt handling (SIGINT-safe saves)
- Wired up TensorBoard logging for train/val loss curves, with decoupled cheap-scalar vs. expensive-sample logging intervals

### Part 5 — Mixture-of-Experts
- Implemented MoE theory: expert routing, gating networks, and load balancing
- Built MoE layers in PyTorch and combined them with dense layers in a hybrid architecture, as an architectural exploration alongside the dense model used for the main pipeline

### Part 6 — Supervised Fine-Tuning
- Built an instruction-formatted dataset pipeline (prompt + response pairs)
- Implemented causal LM loss with masked labels, so loss is only computed over the response tokens, not the prompt
- Fine-tuned the base pretrained model on a general instruction dataset, a hand-written chit-chat set, and a custom personal Q&A dataset (see [The "Ask About Me" demo](#the-ask-about-me-demo))
- Evaluated outputs against expected responses to catch undertraining and dataset imbalance issues

### Part 7 — Reward Modeling
- Built a preference dataset pipeline (pairwise chosen/rejected response rankings)
- Implemented a transformer-encoder reward model architecture with a scalar reward head
- Implemented Bradley–Terry pairwise loss for preference learning
- Added sanity checks (pairwise accuracy tracking) to confirm the reward model was actually learning a meaningful preference signal rather than collapsing to a trivial solution

### Part 8 — RLHF with PPO
- Implemented the policy network: the SFT-tuned language model with an added value head for reward prediction
- Wired the trained Part 7 reward model in as the reward signal for rollouts
- Implemented the PPO objective, balancing reward maximization against a KL penalty that constrains the policy from drifting too far from the SFT reference model
- Built the full training loop: sample prompts → generate completions → score with the reward model → compute advantages → optimize the policy via clipped PPO loss
- Added reward normalization, gradient clipping, and KL-controlled rollout length for training stability
- Built `chat.py`, a terminal chat interface with live token-by-token streaming generation, that can load and compare checkpoints from any stage (SFT, PPO, etc.)

## Model configuration

The dense transformer used throughout the pipeline:

| Hyperparameter | Value |
|---|---|
| Layers | 6 |
| Attention heads | 8 |
| Embedding dimension | 384 |
| Context length (block size) | 256 |
| Vocabulary size (BPE) | 8,000 |
| Positional encoding | RoPE |
| Normalization | RMSNorm |
| Activation | SwiGLU |

## Datasets used

| Stage | Dataset |
|---|---|
| Base pretraining | Custom text corpus (~5M tokens) |
| Supervised fine-tuning | General instruction data + hand-written chit-chat set + custom personal Q&A dataset |
| Reward modeling | Preference pairs (chosen/rejected response rankings) |
| RLHF (PPO) | Reused SFT instruction prompts as the rollout prompt pool |

## Engineering deep dives

These are the two most instructive debugging problems I ran into — and think are actually more valuable to talk about than the parts that worked on the first try.

### Diagnosing a memorized base model

An early full pretraining run reached a suspiciously low training loss. Rather than assume it had converged well, I tested it directly: prompting the model with a mid-corpus excerpt produced a **word-for-word verbatim continuation** matching the actual training text — clear evidence of memorization, not generalization.

I traced the cause mathematically: the number of training steps, multiplied by batch size and gradient accumulation, meant the model had been shown over 3x repeated exposure to the same ~5M-token corpus with zero regularization (dropout disabled). The fix was to recompute the correct number of steps for roughly one full epoch of the corpus, and enable dropout — eliminating the repeated-exposure memorization while keeping training time practical.

### Diagnosing and fixing PPO mode collapse

After running PPO, the model's chat behavior degraded to repeating a single generic phrase ("Hey!") regardless of the input prompt — a total loss of the prompt-aware behavior the pre-PPO SFT checkpoint had. Rather than treat this as a black box, I isolated it stage by stage:

1. Reloaded the pre-PPO SFT checkpoint directly and confirmed it still produced varied, prompt-aware responses — proving the collapse was introduced by PPO, not inherited from SFT.
2. Inspected the PPO rollout generation code and found sampling was configured at a very low temperature and top-k, meaning the policy had almost no room to explore alternative responses during training.
3. Reasoned through the failure mode: a rollout that's this deterministic, combined with a reward model trained on a small preference dataset, lets the policy find one response the reward model happens to score acceptably across many prompts and collapse onto it — classic RLHF reward hacking.

The fix: raised rollout sampling temperature and top-k to restore exploration, added an entropy bonus term to the PPO loss to directly penalize collapsing to a low-diversity output distribution, and tightened the KL-penalty coefficient to keep the policy closer to the known-good SFT reference. Retraining with these changes restored varied, prompt-aware responses.

## The "Ask About Me" demo

To make the pipeline demonstrable interactively rather than just readable as code, I fine-tuned a checkpoint on a custom-written Q&A dataset about my own background, skills, and projects, and built a Streamlit chat interface (`part_8/app.py`) around it. It's the version linked at the top of this README — every response you see there is generated live by the model described above, not templated or hardcoded.

## Running it yourself

```bash
# clone and set up environment
git clone <your-repo-url>
cd llm-from-scratch
pip install -r requirements.txt

# example: run base pretraining (Part 4)
cd part_4
python train.py --data corpus.txt --out runs/base-v1 --bpe --vocab_size 8000 \
  --n_layer 6 --n_head 8 --n_embd 384 --block_size 256 --steps 320 --dropout 0.1

# example: supervised fine-tuning (Part 6)
cd ../part_6
python train_sft.py --ckpt ../part_4/runs/base-v1/model_last.pt \
  --bpe_dir ../part_4/runs/base-v1/tokenizer --out runs/sft-v1 --steps 1200

# chat with a trained checkpoint in the terminal
cd ../part_8
python chat.py --ckpt ../part_6/runs/sft-v1/model_last.pt \
  --bpe_dir ../part_4/runs/base-v1/tokenizer

# or launch the Streamlit demo
streamlit run app.py
```

See each `part_N/` folder for the full set of training flags for that stage.

## Tech stack

Python, PyTorch, custom BPE tokenizer (implemented from scratch), TensorBoard, Streamlit


## Author

Built by Sidharth — [GitHub](https://github.com/JAINSID02) · [LinkedIn](https://linkedin.com/in/jisidharthjain)