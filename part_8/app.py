"""
app.py — Streamlit frontend for the Sidharth-Q&A chatbot.

Run from inside `part_8/`:
    streamlit run app.py
"""

from __future__ import annotations
import sys
from pathlib import Path

import streamlit as st
import torch

from chat import load_model_and_tokenizer

sys.path.append(str(Path(__file__).resolve().parents[1] / "part_6"))
from formatters import format_prompt_only  # type: ignore

# ------------------------------------------------------------------
DEFAULT_CKPT = "../part_6/runs/sft-aboutme-v2/model_last.pt"
DEFAULT_BPE_DIR = "../part_4/runs/base-v3/tokenizer"

DEMO_PROMPTS = [
    "Who is Sidharth?",
    "What is Sidharth's CGPA?",
    "Tell me about MAARS",
    "What is Sidharth's Self-RAG AI Agent?",
    "What fine-tuning techniques does Sidharth know?",
    "What is Sidharth's LLM-from-scratch project?",
    "What roles is Sidharth looking for?",
]

ABOUT_HTML = """
This chatbot demonstrates an LLM built and trained entirely from scratch in PyTorch — every stage engineered end-to-end, with no pre-built model libraries or shortcuts.
The build showcases the full skill set behind modern LLM engineering: implementing self-attention and multi-head attention from first principles, designing a complete transformer block, and building a custom BPE tokenizer and training loop. The architecture was engineered with the same techniques used in today's production models — RoPE, RMSNorm, SwiGLU, and KV-caching for efficient inference — then scaled with gradient accumulation, mixed precision, and learning-rate scheduling.
Beyond pretraining, the project covers the full alignment stack: supervised fine-tuning on custom instruction data, reward model training on preference data, and reinforcement learning from human feedback (RLHF) via PPO — the same alignment approach behind models like ChatGPT and Claude.
This deployment is fine-tuned to answer questions about Sidharth's background, skills, and projects — a working demonstration of the entire pipeline, from raw architecture to a deployed conversational model.
"""

# ------------------------------------------------------------------
st.set_page_config(page_title="Ask About Sidharth", page_icon="◆", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;0,6..72,600;1,6..72,500&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }

.stApp { background-color: #0a0a0a; color: #eaeaea; }
.block-container { padding-top: 2.6rem; max-width: 760px; }

.mark {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 34px; height: 34px;
    border: 1px solid #3a3a3a;
    border-radius: 50%;
    font-family: 'Newsreader', serif;
    font-style: italic;
    font-size: 1.05rem;
    color: #eaeaea;
    margin-bottom: 0.9rem;
}

.hero-title {
    font-family: 'Newsreader', serif;
    font-style: italic;
    font-weight: 500;
    font-size: 3rem;
    letter-spacing: -0.015em;
    color: #f7f7f7;
    margin-bottom: 0.15rem;
    line-height: 1.05;
}

.hero-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.76rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #9a9a9a;
    margin-bottom: 1.8rem;
    border-top: 1px solid #262626;
    padding-top: 0.7rem;
    display: flex;
    justify-content: space-between;
}

.about-toggle summary {
    cursor: pointer;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #9a9a9a;
    padding: 0.7rem 0;
    list-style: none;
}
.about-toggle summary::-webkit-details-marker { display: none; }
.about-toggle summary:before { content: "＋  "; color: #eaeaea; }
.about-toggle[open] summary:before { content: "－  "; }
.about-toggle {
    border-top: 1px solid #262626;
    border-bottom: 1px solid #262626;
    margin-bottom: 1.8rem;
}
.about-body {
    font-size: 0.9rem;
    line-height: 1.7;
    color: #c2c2c2;
    padding-bottom: 1.2rem;
    white-space: pre-line;
}
.about-body b { color: #f0f0f0; }

.section-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #777;
    margin: 0 0 0.7rem 0;
}

.stButton button {
    background-color: #121212;
    color: #d8d8d8;
    border: 1px solid #2e2e2e;
    border-radius: 6px;
    font-family: 'Inter', sans-serif;
    font-size: 0.85rem;
    padding: 0.6rem 0.9rem;
    width: 100%;
    text-align: left;
    transition: all 0.15s ease;
}
.stButton button:hover {
    border-color: #eaeaea;
    color: #ffffff;
    background-color: #1a1a1a;
}

/* custom chat bubbles */
.chat-row { display: flex; margin: 1.1rem 0; gap: 0.7rem; align-items: flex-start; }
.chat-row.user { flex-direction: row-reverse; }

.avatar {
    flex-shrink: 0;
    width: 30px; height: 30px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-family: 'Newsreader', serif;
    font-style: italic;
    font-size: 0.95rem;
}
.avatar.assistant { border: 1px solid #3a3a3a; color: #eaeaea; background: #141414; }
.avatar.user { border: 1px solid #444; color: #0a0a0a; background: #eaeaea; font-style: normal; font-family: 'Inter', sans-serif; font-weight: 600; }

.bubble {
    max-width: 78%;
    padding: 0.75rem 1rem;
    border-radius: 10px;
    font-size: 0.92rem;
    line-height: 1.55;
}
.bubble.assistant { background: #141414; border: 1px solid #262626; color: #eaeaea; border-top-left-radius: 3px; }
.bubble.user { background: #eaeaea; color: #0a0a0a; border-top-right-radius: 3px; }

.stChatInput textarea, [data-testid="stChatInput"] textarea {
    font-family: 'Inter', sans-serif !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="mark">S</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">Ask About Sidharth</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub"><span>An LLM built from first principles — architecture, tokenizer, pretraining, fine tuning and alignment</span></div>',
    unsafe_allow_html=True,
)

st.markdown(f"""
<details class="about-toggle">
<summary>How this works</summary>
<div class="about-body">{ABOUT_HTML}</div>
</details>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model...")
def get_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok, cfg = load_model_and_tokenizer(DEFAULT_CKPT, DEFAULT_BPE_DIR, device)
    return model, tok, cfg, device


try:
    model, tok, cfg, device = get_model()
except Exception as e:
    st.error(f"Couldn't load the model checkpoint. Check DEFAULT_CKPT / DEFAULT_BPE_DIR at the top of app.py.\n\n{e}")
    st.stop()


@torch.no_grad()
def generate_reply_stream(prompt_text: str, max_new_tokens=150, temperature=0.7, top_k=50, eos_id=1):
    try:
        from part_3.utils import top_k_top_p_filtering as _filter
    except Exception:
        def _filter(x, **_):
            return x

    block_size = cfg.get("block_size", 256)
    prompt_ids = tok.encode(prompt_text)[-block_size:]
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    kvs = [None] * len(model.lm.blocks)
    produced_ids = []
    printed_text = ""

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:] if kvs[0] is None else idx[:, -1:]
        start_pos = 0 if kvs[0] is None else kvs[0].k.size(2)

        logits, _, kvs = model.lm(idx_cond, kv_cache_list=kvs, start_pos=start_pos)
        next_logits = logits[:, -1, :] / max(temperature, 1e-6)
        next_logits = _filter(next_logits, top_k=top_k, top_p=None)
        probs = torch.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, 1)

        token_id = next_id.item()
        if eos_id is not None and token_id == eos_id:
            break

        produced_ids.append(token_id)
        idx = torch.cat([idx, next_id], dim=1)

        full_text = tok.decode(produced_ids)
        new_piece = full_text[len(printed_text):]
        if new_piece:
            printed_text = full_text
            yield new_piece


def render_bubble(role: str, text: str) -> str:
    avatar_letter = "S" if role == "assistant" else "Y"
    return f"""
    <div class="chat-row {role}">
        <div class="avatar {role}">{avatar_letter}</div>
        <div class="bubble {role}">{text}</div>
    </div>
    """


# ------------------------------------------------------------------
st.markdown('<div class="section-label">Try asking</div>', unsafe_allow_html=True)

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None
if "messages" not in st.session_state:
    st.session_state.messages = []

cols = st.columns(2)
for i, p in enumerate(DEMO_PROMPTS):
    if cols[i % 2].button(p, key=f"demo_{i}"):
        st.session_state.pending_prompt = p

st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)

# ------------------------------------------------------------------
chat_container = st.container()

with chat_container:
    for msg in st.session_state.messages:
        st.markdown(render_bubble(msg["role"], msg["content"]), unsafe_allow_html=True)

user_input = st.chat_input("Ask something about Sidharth...")
if st.session_state.pending_prompt:
    user_input = st.session_state.pending_prompt
    st.session_state.pending_prompt = None

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with chat_container:
        st.markdown(render_bubble("user", user_input), unsafe_allow_html=True)

        prompt_text = format_prompt_only(user_input).replace("</s>", "")
        placeholder = st.empty()
        accumulated = ""
        for delta in generate_reply_stream(prompt_text):
            accumulated += delta
            placeholder.markdown(render_bubble("assistant", accumulated), unsafe_allow_html=True)

    st.session_state.messages.append({"role": "assistant", "content": accumulated})