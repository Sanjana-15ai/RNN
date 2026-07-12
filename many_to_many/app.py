"""
LINGUA — Neural Signal Decoder
==============================
Many-to-many RNN translation (GRU encoder-decoder), wrapped in a custom
dark "glass-panel / signal-path" Streamlit UI.

Design concept
--------------
The model is, literally, a chain of signals: an encoder compresses an
English sentence into a hidden state, and a decoder fires one token at
a time, each with its own confidence. So instead of printing plain
text, the translation renders as a "decoding trail": one glowing chip
per output word, appearing in sequence, with brightness/glow driven by
the softmax probability the model actually had at that step. Low
confidence words look dim and flicker; high confidence words glow
solid. That's the model's uncertainty made visible instead of hidden.

Dataset: loaded from "translation.txt" (tab-separated:
english sentence<TAB>french sentence), same folder as this file.

Run:
    pip install torch streamlit
    streamlit run app.py
"""

import random
import re
import unicodedata

import matplotlib.pyplot as plt
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ----------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------
st.set_page_config(page_title="LINGUA // Neural Signal Decoder", page_icon="◈", layout="centered")

SOS, EOS, PAD = "<sos>", "<eos>", "<pad>"
DATASET_PATH = "translation.txt"

# ----------------------------------------------------------------------
# Design tokens + global CSS
# ----------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --bg-void: #070a12;
    --bg-panel: rgba(20, 25, 41, 0.55);
    --bg-panel-solid: #141929;
    --border-soft: rgba(140, 155, 210, 0.14);
    --accent-source: #56E8C9;   /* incoming signal — English */
    --accent-target: #B98CFF;   /* decoded signal — French */
    --text-primary: #EAEDF6;
    --text-muted: #7C88A6;
}

#MainMenu, footer, header {visibility: hidden;}

.stApp {
    background:
        radial-gradient(ellipse 80% 50% at 20% -10%, rgba(86, 232, 201, 0.10), transparent),
        radial-gradient(ellipse 80% 50% at 80% 0%, rgba(185, 140, 255, 0.10), transparent),
        var(--bg-void);
    font-family: 'Inter', sans-serif;
    color: var(--text-primary);
}

.block-container { padding-top: 2.5rem; max-width: 760px; }

/* ---------- Hero ---------- */
.hero-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.22em;
    color: var(--accent-source);
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 2.6rem;
    line-height: 1.05;
    margin: 0 0 0.5rem 0;
    background: linear-gradient(100deg, var(--accent-source) 0%, var(--accent-target) 100%);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
}
.hero-sub {
    color: var(--text-muted);
    font-size: 0.98rem;
    max-width: 46ch;
    margin-bottom: 1.6rem;
}

/* ---------- Telemetry row ---------- */
.telemetry {
    display: flex;
    gap: 0;
    border: 1px solid var(--border-soft);
    background: var(--bg-panel);
    border-radius: 14px;
    overflow: hidden;
    margin-bottom: 1.8rem;
    backdrop-filter: blur(10px);
}
.telemetry-item {
    flex: 1;
    padding: 0.9rem 1rem;
    border-right: 1px solid var(--border-soft);
}
.telemetry-item:last-child { border-right: none; }
.telemetry-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.25rem;
}
.telemetry-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.25rem;
    font-weight: 600;
    color: var(--text-primary);
}

/* ---------- Panels ---------- */
.panel {
    border: 1px solid var(--border-soft);
    background: var(--bg-panel);
    border-radius: 16px;
    padding: 1.4rem 1.5rem;
    margin-bottom: 1.4rem;
    backdrop-filter: blur(10px);
}
.panel-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--accent-source);
    margin-bottom: 0.7rem;
}

/* ---------- Streamlit input / button overrides ---------- */
div[data-testid="stTextInput"] input {
    background: rgba(8, 11, 20, 0.7) !important;
    border: 1px solid var(--border-soft) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
    padding: 0.65rem 0.9rem !important;
}
div[data-testid="stTextInput"] input:focus {
    border-color: var(--accent-target) !important;
    box-shadow: 0 0 0 1px var(--accent-target) !important;
}
.stButton button {
    background: linear-gradient(100deg, var(--accent-source), var(--accent-target)) !important;
    color: #08090f !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    padding: 0.55rem 1.3rem !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
}
.stButton button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(185, 140, 255, 0.35) !important;
}
.stButton button p { color: #08090f !important; font-weight: 600 !important; }

/* Example chip buttons (secondary look) */
.example-row .stButton button {
    background: rgba(140, 155, 210, 0.08) !important;
    border: 1px solid var(--border-soft) !important;
    color: var(--text-primary) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8rem !important;
    padding: 0.4rem 0.9rem !important;
}
.example-row .stButton button p { color: var(--text-primary) !important; }
.example-row .stButton button:hover { box-shadow: 0 0 12px rgba(86, 232, 201, 0.25) !important; }

/* ---------- Decoding trail (signature element) ---------- */
.trail {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    align-items: center;
}
.chip {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    font-size: 1.05rem;
    padding: 0.5rem 0.95rem;
    border-radius: 999px;
    color: var(--text-primary);
    opacity: 0;
    animation: chip-in 0.45s ease forwards;
    position: relative;
}
.chip-arrow {
    color: var(--text-muted);
    font-size: 0.85rem;
    opacity: 0;
    animation: chip-in 0.3s ease forwards;
}
@keyframes chip-in {
    from { opacity: 0; transform: translateY(6px) scale(0.94); }
    to   { opacity: 1; transform: translateY(0) scale(1); }
}
.conf-meter {
    height: 5px;
    border-radius: 4px;
    background: rgba(140, 155, 210, 0.15);
    overflow: hidden;
    margin-top: 0.9rem;
}
.conf-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--accent-source), var(--accent-target));
    border-radius: 4px;
}
.conf-caption {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: var(--text-muted);
    margin-top: 0.4rem;
}
.source-echo {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: var(--accent-source);
    margin-bottom: 0.6rem;
}
</style>
""",
    unsafe_allow_html=True,
)


def normalize(s: str) -> str:
    s = unicodedata.normalize("NFD", s.lower().strip())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-z.!? ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


class Vocab:
    def __init__(self, name):
        self.name = name
        self.word2idx = {PAD: 0, SOS: 1, EOS: 2}
        self.idx2word = {0: PAD, 1: SOS, 2: EOS}

    def add_sentence(self, sentence):
        for w in sentence.split(" "):
            if w not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[w] = idx
                self.idx2word[idx] = w

    def __len__(self):
        return len(self.word2idx)

    def encode(self, sentence, max_len):
        ids = [self.word2idx.get(w, 0) for w in sentence.split(" ")]
        ids = ids[: max_len - 1] + [self.word2idx[EOS]]
        ids += [self.word2idx[PAD]] * (max_len - len(ids))
        return ids


class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.gru = nn.GRU(emb_dim, hidden_dim, batch_first=True)

    def forward(self, src):
        embedded = self.embedding(src)
        outputs, hidden = self.gru(embedded)
        return outputs, hidden


class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.gru = nn.GRU(emb_dim, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_step, hidden):
        embedded = self.embedding(input_step)
        output, hidden = self.gru(embedded, hidden)
        logits = self.out(output.squeeze(1))
        return logits, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, tgt_vocab, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.tgt_vocab = tgt_vocab
        self.device = device

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        batch_size, tgt_len = tgt.shape
        outputs = torch.zeros(batch_size, tgt_len, len(self.tgt_vocab), device=self.device)

        _, hidden = self.encoder(src)
        input_step = torch.full((batch_size, 1), self.tgt_vocab.word2idx[SOS],
                                 dtype=torch.long, device=self.device)

        for t in range(tgt_len):
            logits, hidden = self.decoder(input_step, hidden)
            outputs[:, t, :] = logits
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = logits.argmax(1, keepdim=True)
            input_step = tgt[:, t].unsqueeze(1) if teacher_force else top1

        return outputs

    @torch.no_grad()
    def translate_with_confidence(self, sentence, src_vocab, max_len):
        """Returns list of (word, confidence) — confidence is the softmax
        probability the decoder assigned to the word it actually picked."""
        self.eval()
        sentence = normalize(sentence)
        src = torch.tensor([src_vocab.encode(sentence, max_len)]).to(self.device)
        _, hidden = self.encoder(src)

        input_step = torch.tensor([[self.tgt_vocab.word2idx[SOS]]], device=self.device)
        results = []
        for _ in range(max_len):
            logits, hidden = self.decoder(input_step, hidden)
            probs = F.softmax(logits, dim=1)
            top1 = logits.argmax(1).item()
            confidence = probs[0, top1].item()
            if top1 == self.tgt_vocab.word2idx[EOS]:
                break
            results.append((self.tgt_vocab.idx2word.get(top1, "?"), confidence))
            input_step = torch.tensor([[top1]], device=self.device)
        return results


# ----------------------------------------------------------------------
# Cache data loading + training so it only runs ONCE, not on every
# Streamlit rerun (Streamlit reruns the whole script on each interaction).
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="Training the model (one-time)...")
def load_and_train(epochs=300, emb_dim=64, hidden_dim=128, lr=0.01):
    with open(DATASET_PATH, encoding="utf-8") as f:
        raw_pairs = [
            tuple(line.strip().split("\t"))
            for line in f
            if line.strip() and "\t" in line
        ]

    pairs = [(normalize(en), normalize(fr)) for en, fr in raw_pairs]

    src_vocab, tgt_vocab = Vocab("eng"), Vocab("fra")
    for en, fr in pairs:
        src_vocab.add_sentence(en)
        tgt_vocab.add_sentence(fr)

    max_len = max(max(len(en.split()), len(fr.split())) for en, fr in pairs) + 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    src_tensor = torch.tensor([src_vocab.encode(en, max_len) for en, _ in pairs]).to(device)
    tgt_tensor = torch.tensor([tgt_vocab.encode(fr, max_len) for _, fr in pairs]).to(device)

    encoder = Encoder(len(src_vocab), emb_dim, hidden_dim).to(device)
    decoder = Decoder(len(tgt_vocab), emb_dim, hidden_dim).to(device)
    model = Seq2Seq(encoder, decoder, tgt_vocab, device).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    losses = []
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        outputs = model(src_tensor, tgt_tensor, teacher_forcing_ratio=0.5)
        loss = criterion(outputs.reshape(-1, outputs.shape[-1]), tgt_tensor.reshape(-1))
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return model, src_vocab, tgt_vocab, max_len, len(pairs), losses


def render_trail(results):
    """Render the signature 'decoding trail': one glowing chip per output
    word, brightness driven by the model's own confidence at that step."""
    parts = []
    for i, (word, conf) in enumerate(results):
        # colour interpolation from muted grey -> accent-target as confidence rises
        glow = 0.15 + 0.85 * conf
        border_alpha = 0.25 + 0.6 * conf
        bg_alpha = 0.08 + 0.30 * conf
        parts.append(
            f'<span class="chip" style="'
            f'animation-delay:{i * 0.12:.2f}s;'
            f'background: rgba(185, 140, 255, {bg_alpha:.2f});'
            f'border: 1px solid rgba(185, 140, 255, {border_alpha:.2f});'
            f'box-shadow: 0 0 {14 * glow:.0f}px rgba(185, 140, 255, {glow * 0.55:.2f});'
            f'" title="confidence {conf * 100:.1f}%">{word}</span>'
        )
        if i < len(results) - 1:
            parts.append(f'<span class="chip-arrow" style="animation-delay:{i * 0.12 + 0.06:.2f}s;">›</span>')
    return f'<div class="trail">{"".join(parts)}</div>'


# ----------------------------------------------------------------------
# Hero
# ----------------------------------------------------------------------
st.markdown('<div class="hero-eyebrow">encoder → hidden state → decoder</div>', unsafe_allow_html=True)
st.markdown('<h1 class="hero-title">LINGUA<br>Neural Signal Decoder</h1>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">A GRU encoder compresses your sentence into a single hidden '
    'state. A GRU decoder then fires one French word at a time — each one shown below '
    'exactly as confident (or unsure) as the model actually was.</div>',
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Load / train
# ----------------------------------------------------------------------
try:
    model, src_vocab, tgt_vocab, max_len, num_pairs, losses = load_and_train()
except FileNotFoundError:
    st.error(f"Could not find '{DATASET_PATH}'. Put it in the same folder as app.py.")
    st.stop()

st.markdown(
    f"""
<div class="telemetry">
    <div class="telemetry-item">
        <div class="telemetry-label">Training Pairs</div>
        <div class="telemetry-value">{num_pairs}</div>
    </div>
    <div class="telemetry-item">
        <div class="telemetry-label">Source Vocab</div>
        <div class="telemetry-value">{len(src_vocab)}</div>
    </div>
    <div class="telemetry-item">
        <div class="telemetry-label">Target Vocab</div>
        <div class="telemetry-value">{len(tgt_vocab)}</div>
    </div>
    <div class="telemetry-item">
        <div class="telemetry-label">Final Loss</div>
        <div class="telemetry-value">{losses[-1]:.3f}</div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Input panel
# ----------------------------------------------------------------------
st.markdown('<div class="panel">', unsafe_allow_html=True)
st.markdown('<div class="panel-label">Input signal</div>', unsafe_allow_html=True)
sentence = st.text_input(" ", value="i am happy", label_visibility="collapsed",
                          placeholder="Type an English sentence...")
go = st.button("Decode →", type="primary")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="example-row">', unsafe_allow_html=True)
examples = ["i am happy", "i love you", "good morning", "i like coffee", "how are you", "where is the bathroom"]
cols = st.columns(3)
picked = None
for i, ex in enumerate(examples):
    if cols[i % 3].button(ex, key=f"ex_{i}"):
        picked = ex
st.markdown('</div>', unsafe_allow_html=True)

active_sentence = picked or (sentence if go else None)

if active_sentence and active_sentence.strip():
    results = model.translate_with_confidence(active_sentence, src_vocab, max_len)
    avg_conf = sum(c for _, c in results) / len(results) if results else 0.0

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-label">Decoded output</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="source-echo">EN · {active_sentence.strip()}</div>', unsafe_allow_html=True)
    if results:
        st.markdown(render_trail(results), unsafe_allow_html=True)
        st.markdown(
            f'<div class="conf-meter"><div class="conf-fill" style="width:{avg_conf * 100:.0f}%;"></div></div>'
            f'<div class="conf-caption">avg. decoder confidence · {avg_conf * 100:.1f}%</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<span class="chip-arrow" style="opacity:1;">(empty output)</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Training curve (dark themed)
# ----------------------------------------------------------------------
with st.expander("Show training loss curve"):
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    fig.patch.set_facecolor("#0a0e17")
    ax.set_facecolor("#0a0e17")
    ax.plot(losses, color="#B98CFF", linewidth=1.6)
    ax.fill_between(range(len(losses)), losses, color="#B98CFF", alpha=0.08)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_color("#2a3148")
    ax.tick_params(colors="#7C88A6", labelsize=8)
    ax.set_xlabel("epoch", color="#7C88A6", fontsize=9)
    ax.set_ylabel("loss", color="#7C88A6", fontsize=9)
    ax.grid(alpha=0.08, color="#7C88A6")
    st.pyplot(fig)

st.markdown(
    '<div class="conf-caption" style="margin-top:1.5rem;">'
    'Trained only on translation.txt — works best on sentences similar to those in the dataset.'
    '</div>',
    unsafe_allow_html=True,
)