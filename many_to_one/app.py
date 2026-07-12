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
    page_icon="📱",
    layout="centered"
)

# ---- Custom CSS: dark glass "signal panel" theme (matches Country Explorer) ----
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background:
        radial-gradient(circle at 10% -10%, rgba(123,241,255,0.16) 0%, transparent 40%),
        radial-gradient(circle at 90% 0%, rgba(244,114,182,0.14) 0%, transparent 45%),
        radial-gradient(circle at 50% 100%, rgba(167,139,250,0.18) 0%, transparent 50%),
        linear-gradient(180deg, #0b0a1f 0%, #100c26 60%, #0b0a1f 100%);
    background-attachment: fixed;
}

#MainMenu, footer, header {visibility: hidden;}
.block-container { padding-top: 2.6rem; max-width: 760px; }

/* Floating decorative blobs */
.blob {
    position: fixed;
    border-radius: 50%;
    filter: blur(90px);
    opacity: 0.35;
    z-index: 0;
    animation: floaty 10s ease-in-out infinite;
}
@keyframes floaty {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-25px); }
}

/* ---------- Hero ---------- */
.hero-wrap { text-align: center; margin-bottom: 2rem; }
.hero-badge {
    display: inline-block;
    padding: 0.35rem 1rem;
    border-radius: 999px;
    background: rgba(123,241,255,0.1);
    border: 1px solid rgba(123,241,255,0.3);
    color: #7bf1ff;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 1rem;
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 3rem;
    font-weight: 700;
    background: linear-gradient(90deg, #7bf1ff 0%, #a78bfa 45%, #f472b6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.3rem;
    letter-spacing: -1px;
    line-height: 1.1;
}
.hero-sub {
    color: #a9a6c9;
    font-size: 1.05rem;
    font-weight: 400;
    max-width: 520px;
    margin: 0 auto;
}

/* ---------- Input card ---------- */
.input-card {
    background: rgba(255,255,255,0.045);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 22px;
    padding: 1.8rem 2rem;
    backdrop-filter: blur(18px);
    box-shadow: 0 12px 40px rgba(0,0,0,0.4);
    margin-bottom: 1.6rem;
    position: relative;
    z-index: 1;
}
.input-label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.75rem;
    letter-spacing: 1.6px;
    text-transform: uppercase;
    color: #8be9fd;
    font-weight: 600;
    margin-bottom: 0.8rem;
}

.stTextArea textarea {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.14) !important;
    border-radius: 14px !important;
    color: #f5f3ff !important;
    font-size: 1rem !important;
    padding: 14px !important;
}
.stTextArea textarea:focus {
    border-color: rgba(167,139,250,0.6) !important;
    box-shadow: 0 0 0 3px rgba(167,139,250,0.15) !important;
}
.stTextArea textarea::placeholder { color: #6f6c93 !important; }

div.stButton > button {
    background: linear-gradient(90deg, #7bf1ff, #a78bfa, #f472b6);
    background-size: 200% auto;
    color: #0b0a1f;
    font-weight: 700;
    border: none;
    border-radius: 14px;
    padding: 0.75rem 1.6rem;
    font-family: 'Space Grotesk', sans-serif;
    letter-spacing: 0.3px;
    font-size: 1rem;
    width: 100%;
    transition: all 0.3s ease;
    box-shadow: 0 8px 24px rgba(167,139,250,0.35);
}
div.stButton > button:hover {
    background-position: right center;
    transform: translateY(-2px);
    box-shadow: 0 12px 30px rgba(167,139,250,0.5);
}
div.stButton > button p { color: #0b0a1f !important; font-weight: 700 !important; }

/* ---------- Result card ---------- */
.field-card {
    background: linear-gradient(150deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    animation: cardIn 0.5s cubic-bezier(0.2, 0.8, 0.2, 1) both;
    margin-top: 1.5rem;
}
.field-card.spam { border-color: rgba(244,114,182,0.5); box-shadow: 0 16px 40px rgba(244,114,182,0.15); }
.field-card.ham { border-color: rgba(123,241,255,0.5); box-shadow: 0 16px 40px rgba(123,241,255,0.15); }

@keyframes cardIn {
    from { opacity: 0; transform: translateY(22px) scale(0.98); }
    to   { opacity: 1; transform: translateY(0) scale(1); }
}

.result-emoji { font-size: 2.6rem; margin-bottom: 0.4rem; }
.result-label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0.1rem 0;
}
.result-line { color: #a9a6c9; font-size: 0.98rem; margin-bottom: 0.4rem; }

.stat-pill {
    display: inline-block;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.14);
    color: #d5d2f5;
    padding: 0.4rem 1rem;
    border-radius: 999px;
    font-size: 0.85rem;
    margin-top: 1rem;
    font-weight: 500;
}

/* Confidence bar */
.stProgress > div > div {
    background: linear-gradient(90deg, #7bf1ff, #a78bfa, #f472b6) !important;
}
.stProgress > div {
    background: rgba(255,255,255,0.08) !important;
}

/* Expander styling */
div[data-testid="stExpander"] {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    margin-top: 1.2rem;
}
div[data-testid="stExpander"] summary {
    color: #d5d2f5 !important;
    font-family: 'Space Grotesk', sans-serif;
}
</style>

<div class="blob" style="width:340px; height:340px; top:-100px; left:-120px; background:#7bf1ff;"></div>
<div class="blob" style="width:380px; height:380px; top:-60px; right:-140px; background:#f472b6; animation-delay: 2s;"></div>
""", unsafe_allow_html=True)

# ---- Header ----
st.markdown("""
    <div class="hero-wrap">
        <div class="hero-badge">🧠 Simple RNN · Many-to-One</div>
        <div class="hero-title">SMS Spam Detector</div>
        <div class="hero-sub">Paste any SMS message and let a recurrent neural network
        read between the lines — spam or safe, decided in real time.</div>
    </div>
""", unsafe_allow_html=True)

# ---- Input card ----
st.markdown('<div class="input-card">', unsafe_allow_html=True)
st.markdown('<div class="input-label">✍️ Message</div>', unsafe_allow_html=True)
message = st.text_area(
    " ",
    placeholder="e.g. Congratulations! You've won a free prize, claim now...",
    label_visibility="collapsed",
)
predict_clicked = st.button("🔍 Analyze Message")
st.markdown('</div>', unsafe_allow_html=True)

if predict_clicked:
    if message.strip() == "":
        st.warning("⚠️ Please enter a message first.")
    else:
        with st.spinner("🧠 Reading between the lines..."):
            prediction, probability = predict_sms(message)

        confidence_pct = round(probability * 100, 2)
        card_class = "spam" if prediction == "Spam" else "ham"
        emoji = "🚨" if prediction == "Spam" else "✅"
        message_line = (
            "This looks like SPAM — be careful!" if prediction == "Spam"
            else "This looks like a normal, safe message."
        )

        # ---- Result card ----
        st.markdown(f"""
            <div class="field-card {card_class}">
                <div class="result-emoji">{emoji}</div>
                <div class="result-label">{prediction.upper()}</div>
                <div class="result-line">{message_line}</div>
                <span class="stat-pill">Confidence · {confidence_pct}%</span>
            </div>
        """, unsafe_allow_html=True)

        st.write("")
        st.progress(int(confidence_pct))

        # ---- Extra detail expander ----
        with st.expander("🔬 See technical details"):
            st.write(f"**Cleaned input:** `{clean_text(message)}`")
            st.write(f"**Raw model probability (spam):** `{round(probability, 4)}`")
            st.write(f"**Max sequence length:** {MAX_LEN} tokens")
            st.write(f"**Vocabulary size:** {MAX_WORDS} words")