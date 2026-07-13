# RNN PRACTICAL [MANY TO ONE]
# SMS Spam Detection using Simple RNN (Many-to-one)
# Dataset: spam.csv

import os
import re
import pickle
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, SimpleRNN, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---------------- Configuration ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(BASE_DIR, "spam_model.keras")
TOKENIZER = os.path.join(BASE_DIR, "tokenizer.pkl")
DATA_PATH = os.path.join(BASE_DIR, "spam.csv")

MAX_WORDS = 5000
MAX_LEN = 50


# ---------------- Clean Text ----------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)   # remove punctuation/symbols
    text = re.sub(r"\s+", " ", text)           # collapse multiple spaces into one
    return text.strip()


# ---------------- Train Model ----------------
def train_model():

    print("training dataset...")

    df = pd.read_csv(DATA_PATH, encoding="latin-1")
    df = df[["v1", "v2"]]
    df.columns = ["label", "message"]

    print(df.head())
    print(df["label"].value_counts())

    # convert labels into numbers
    df["label"] = df["label"].map({"ham": 0, "spam": 1})

    # clean sms text
    df["message"] = df["message"].apply(clean_text)

    # Tokenizer
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(df["message"])

    sequences = tokenizer.texts_to_sequences(df["message"])
    X = pad_sequences(sequences, maxlen=MAX_LEN, padding="post")
    Y = df["label"].values

    print("X shape:", X.shape)
    print("Y shape:", Y.shape)

    # save tokenizer
    with open(TOKENIZER, "wb") as f:
        pickle.dump(tokenizer, f)

    # train test split
    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    # build RNN model
    model = Sequential()
    model.add(Embedding(input_dim=MAX_WORDS, output_dim=128, input_length=MAX_LEN))
    model.add(SimpleRNN(128))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()

    # train
    model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

    # save the model
    model.save(MODEL)

    # evaluate
    loss, accuracy = model.evaluate(x_test, y_test)
    print("\naccuracy:", accuracy)

    predictions = (model.predict(x_test) > 0.5).astype("int")
    print(classification_report(y_test, predictions))


# ---------------- Prediction (top-level, cached) ----------------
@st.cache_resource
def load_model_and_tokenizer():
    model = load_model(MODEL)
    with open(TOKENIZER, "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer


def predict_sms(message):
    model, tokenizer = load_model_and_tokenizer()

    cleaned = clean_text(message)
    sequence = tokenizer.texts_to_sequences([cleaned])
    sequence = pad_sequences(sequence, maxlen=MAX_LEN, padding="post")

    probability = model.predict(sequence, verbose=0)[0][0]

    if probability > 0.5:
        return "Spam", probability
    else:
        return "Ham", 1 - probability


# ---------------- Train if model doesn't exist ----------------
if not os.path.exists(MODEL):
    train_model()

# ---------------- Streamlit UI ----------------
st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="📡",
    layout="centered"
)

# ---- Custom CSS: terminal / signal-scanner theme ----
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;700&display=swap');

html, body, [class*="css"] { font-family: 'JetBrains Mono', monospace; }

.stApp {
    background:
        repeating-linear-gradient(0deg, rgba(57,255,136,0.025) 0px, rgba(57,255,136,0.025) 1px, transparent 1px, transparent 3px),
        radial-gradient(circle at 15% 0%, rgba(57,255,136,0.08) 0%, transparent 45%),
        radial-gradient(circle at 85% 100%, rgba(255,184,107,0.06) 0%, transparent 50%),
        linear-gradient(180deg, #04060a 0%, #070a10 55%, #04060a 100%);
    background-attachment: fixed;
}

#MainMenu, footer, header {visibility: hidden;}
.block-container { padding-top: 2.4rem; max-width: 700px; }

/* ---------- scanning sweep line ---------- */
.scanline {
    position: fixed;
    left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, #39ff88, transparent);
    opacity: 0.5;
    animation: sweep 5s linear infinite;
    z-index: 0;
}
@keyframes sweep {
    0%   { top: 0%; }
    100% { top: 100%; }
}

/* ---------- Header ---------- */
.term-bar {
    display: flex; align-items: center; gap: 8px;
    background: rgba(57,255,136,0.06);
    border: 1px solid rgba(57,255,136,0.25);
    border-radius: 10px 10px 0 0;
    padding: 0.5rem 0.9rem;
    font-size: 0.72rem;
    color: #6ee7a0;
    letter-spacing: 1.5px;
}
.term-dot { width: 9px; height: 9px; border-radius: 50%; display: inline-block; }
.hero-wrap {
    text-align: left;
    margin-bottom: 1.8rem;
    border: 1px solid rgba(57,255,136,0.25);
    border-top: none;
    border-radius: 0 0 12px 12px;
    padding: 1.4rem 1.6rem 1.6rem 1.6rem;
    background: rgba(6,10,14,0.6);
}
.hero-badge {
    display: inline-block;
    color: #ffb86b;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 0.7rem;
}
.hero-badge::before { content: "// "; color: #4a5568; }
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.3rem;
    font-weight: 700;
    color: #eafff1;
    margin-bottom: 0.5rem;
    letter-spacing: -0.5px;
    line-height: 1.15;
}
.hero-title .cursor {
    display: inline-block;
    width: 10px; height: 1.9rem;
    background: #39ff88;
    margin-left: 4px;
    vertical-align: text-bottom;
    animation: blink 1s step-start infinite;
}
@keyframes blink { 50% { opacity: 0; } }
.hero-sub {
    color: #7a8a94;
    font-size: 0.92rem;
    font-weight: 400;
    max-width: 540px;
    line-height: 1.55;
}
.hero-sub .accent { color: #39ff88; }

/* ---------- Input card ---------- */
.input-card {
    background: rgba(8,12,16,0.75);
    border: 1px solid rgba(57,255,136,0.22);
    border-radius: 14px;
    padding: 1.5rem 1.6rem;
    box-shadow: 0 0 0 1px rgba(0,0,0,0.3), 0 14px 40px rgba(0,0,0,0.5);
    margin-bottom: 1.5rem;
    position: relative;
    z-index: 1;
}
.input-label {
    font-size: 0.72rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #39ff88;
    font-weight: 700;
    margin-bottom: 0.7rem;
}
.input-label::before { content: "> "; color: #4a5568; }

/* Force an OPAQUE dark textarea so light text is always readable,
   regardless of Streamlit's own theme underneath it. */
.stTextArea textarea {
    background-color: #0b0f13 !important;
    background-image: none !important;
    border: 1px solid rgba(57,255,136,0.3) !important;
    border-radius: 10px !important;
    color: #d6ffe4 !important;
    caret-color: #39ff88 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.96rem !important;
    padding: 14px !important;
    -webkit-text-fill-color: #d6ffe4 !important;
}
.stTextArea textarea:focus {
    border-color: #39ff88 !important;
    box-shadow: 0 0 0 3px rgba(57,255,136,0.15) !important;
}
.stTextArea textarea::placeholder { color: #4a5568 !important; opacity: 1 !important; }
/* Some Streamlit versions wrap the textarea in a themed div; neutralize it too */
div[data-baseweb="textarea"] { background-color: #0b0f13 !important; }

div.stButton > button {
    background: #0b0f13;
    color: #39ff88 !important;
    font-weight: 700;
    border: 1.5px solid #39ff88 !important;
    border-radius: 10px;
    padding: 0.7rem 1.6rem;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    font-size: 0.85rem;
    width: 100%;
    transition: all 0.2s ease;
}
div.stButton > button:hover {
    background: #39ff88;
    color: #04060a !important;
    box-shadow: 0 0 22px rgba(57,255,136,0.5);
}
div.stButton > button p { color: inherit !important; font-weight: 700 !important; }

/* ---------- Result card ---------- */
.field-card {
    background: rgba(8,12,16,0.8);
    border: 1px solid rgba(57,255,136,0.2);
    border-radius: 14px;
    padding: 1.8rem 1.6rem;
    text-align: left;
    box-shadow: 0 14px 40px rgba(0,0,0,0.5);
    animation: cardIn 0.4s ease both;
    margin-top: 1.4rem;
}
.field-card.spam { border-color: rgba(255,107,107,0.55); box-shadow: 0 0 30px rgba(255,107,107,0.12); }
.field-card.ham  { border-color: rgba(57,255,136,0.55); box-shadow: 0 0 30px rgba(57,255,136,0.12); }

@keyframes cardIn {
    from { opacity: 0; transform: translateY(14px); }
    to   { opacity: 1; transform: translateY(0); }
}

.result-tag {
    display: inline-block;
    font-size: 0.7rem;
    letter-spacing: 2px;
    font-weight: 700;
    padding: 0.25rem 0.7rem;
    border-radius: 999px;
    margin-bottom: 0.8rem;
    text-transform: uppercase;
}
.result-tag.spam { color: #ff6b6b; border: 1px solid rgba(255,107,107,0.5); background: rgba(255,107,107,0.08); }
.result-tag.ham  { color: #39ff88; border: 1px solid rgba(57,255,136,0.5); background: rgba(57,255,136,0.08); }

.result-label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.9rem;
    font-weight: 700;
    color: #eafff1;
    margin: 0.1rem 0 0.3rem 0;
}
.result-line { color: #8a9aa4; font-size: 0.92rem; margin-bottom: 0.9rem; }

.stat-pill {
    display: inline-block;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.12);
    color: #b9c4c9;
    padding: 0.35rem 0.9rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 500;
    letter-spacing: 0.5px;
}

/* Confidence bar */
.stProgress > div > div {
    background: linear-gradient(90deg, #39ff88, #ffb86b) !important;
}
.stProgress > div {
    background: rgba(255,255,255,0.06) !important;
    border-radius: 999px !important;
}

/* Expander styling */
div[data-testid="stExpander"] {
    background: rgba(6,10,14,0.7);
    border: 1px solid rgba(57,255,136,0.18);
    border-radius: 12px;
    margin-top: 1.1rem;
}
div[data-testid="stExpander"] summary {
    color: #6ee7a0 !important;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    letter-spacing: 0.5px;
}
</style>

<div class="scanline"></div>
""", unsafe_allow_html=True)

# ---- Header ----
st.markdown("""
    <div class="term-bar">
        <span class="term-dot" style="background:#ff6b6b;"></span>
        <span class="term-dot" style="background:#ffb86b;"></span>
        <span class="term-dot" style="background:#39ff88;"></span>
        &nbsp;spam_detector.rnn
    </div>
    <div class="hero-wrap">
        <div class="hero-badge">simple rnn · many-to-one</div>
        <div class="hero-title">SMS Spam Detector<span class="cursor"></span></div>
        <div class="hero-sub">Drop in a message below. A recurrent neural network reads it
        token by token and decides — <span class="accent">safe</span> or
        <span class="accent">spam</span> — in real time.</div>
    </div>
""", unsafe_allow_html=True)

# ---- Input card ----
st.markdown('<div class="input-card">', unsafe_allow_html=True)
st.markdown('<div class="input-label">message input</div>', unsafe_allow_html=True)
message = st.text_area(
    " ",
    placeholder="e.g. Congratulations! You've won a free prize, claim now...",
    label_visibility="collapsed",
)
predict_clicked = st.button("[ ANALYZE MESSAGE ]")
st.markdown('</div>', unsafe_allow_html=True)

if predict_clicked:
    if message.strip() == "":
        st.warning("⚠️ Please enter a message first.")
    else:
        with st.spinner("Scanning token sequence..."):
            prediction, probability = predict_sms(message)

        confidence_pct = round(probability * 100, 2)
        card_class = "spam" if prediction == "Spam" else "ham"
        tag_text = "threat detected" if prediction == "Spam" else "signal clean"
        message_line = (
            "This message shows spam-like patterns — proceed with caution."
            if prediction == "Spam"
            else "No spam indicators found in this message."
        )

        # ---- Result card ----
        st.markdown(f"""
            <div class="field-card {card_class}">
                <span class="result-tag {card_class}">{tag_text}</span>
                <div class="result-label">{prediction.upper()}</div>
                <div class="result-line">{message_line}</div>
                <span class="stat-pill">confidence · {confidence_pct}%</span>
            </div>
        """, unsafe_allow_html=True)

        st.write("")
        st.progress(int(confidence_pct))

        # ---- Extra detail expander ----
        with st.expander("view technical details"):
            st.write(f"**Cleaned input:** `{clean_text(message)}`")
            st.write(f"**Raw model probability (spam):** `{round(probability, 4)}`")
            st.write(f"**Max sequence length:** {MAX_LEN} tokens")
            st.write(f"**Vocabulary size:** {MAX_WORDS} words")