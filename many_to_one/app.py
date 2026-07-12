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
MODEL = "spam_model.keras"
TOKENIZER = "tokenizer.pkl"

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

    df = pd.read_csv("spam.csv", encoding="latin-1")
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

# ---- Custom CSS ----
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e9f2 100%);
    }
    .title-container {
        text-align: center;
        padding: 20px 0 10px 0;
    }
    .title-container h1 {
        background: linear-gradient(90deg, #6a11cb 0%, #2575fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.6rem;
        font-weight: 800;
        margin-bottom: 0;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1rem;
        margin-bottom: 30px;
    }
    .stTextArea textarea {
        border-radius: 14px;
        border: 2px solid #d6d9e0;
        font-size: 1rem;
        padding: 14px;
    }
    .stTextArea textarea:focus {
        border-color: #6a11cb;
        box-shadow: 0 0 0 3px rgba(106, 17, 203, 0.15);
    }
    div.stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #6a11cb 0%, #2575fc 100%);
        color: white;
        border: none;
        border-radius: 14px;
        padding: 12px 0;
        font-size: 1.1rem;
        font-weight: 600;
        transition: transform 0.15s ease;
    }
    div.stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 6px 18px rgba(106, 17, 203, 0.35);
    }
    .result-card {
        border-radius: 18px;
        padding: 28px;
        text-align: center;
        margin-top: 25px;
        animation: popIn 0.4s ease;
    }
    .spam-card {
        background: linear-gradient(135deg, #ff5f6d 0%, #ffc371 100%);
        color: white;
    }
    .ham-card {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e063 100%);
        color: white;
    }
    .result-emoji {
        font-size: 3rem;
    }
    .result-label {
        font-size: 2rem;
        font-weight: 800;
        margin: 5px 0;
    }
    @keyframes popIn {
        0% { transform: scale(0.85); opacity: 0; }
        100% { transform: scale(1); opacity: 1; }
    }
    </style>
""", unsafe_allow_html=True)

# ---- Header ----
st.markdown("""
    <div class="title-container">
        <h1>📱 SMS Spam Detector</h1>
    </div>
    <p class="subtitle">Powered by a Simple RNN &nbsp;·&nbsp; Many-to-One Sequence Model 🧠</p>
""", unsafe_allow_html=True)

# ---- Input ----
message = st.text_area(
    "✍️ Type or paste an SMS message below:",
    placeholder="e.g. Congratulations! You've won a free prize, claim now..."
)

predict_clicked = st.button("🔍 Analyze Message")

if predict_clicked:
    if message.strip() == "":
        st.warning("⚠️ Please enter a message first.")
    else:
        with st.spinner("🧠 Reading between the lines..."):
            prediction, probability = predict_sms(message)

        confidence_pct = round(probability * 100, 2)
        card_class = "spam-card" if prediction == "Spam" else "ham-card"
        emoji = "🚨" if prediction == "Spam" else "✅"
        message_line = (
            "This looks like SPAM — be careful!" if prediction == "Spam"
            else "This looks like a normal, safe message."
        )

        # ---- Animated result card ----
        st.markdown(f"""
            <div class="result-card {card_class}">
                <div class="result-emoji">{emoji}</div>
                <div class="result-label">{prediction.upper()}</div>
                <div>{message_line}</div>
            </div>
        """, unsafe_allow_html=True)

        # ---- Confidence gauge ----
        st.write("")
        st.markdown("**Model Confidence**")
        st.progress(int(confidence_pct))
        st.markdown(
            f"<p style='text-align:center; font-size:1.3rem; font-weight:700;'>{confidence_pct}%</p>",
            unsafe_allow_html=True
        )

        # ---- Extra detail expander ----
        with st.expander("🔬 See technical details"):
            st.write(f"**Cleaned input:** `{clean_text(message)}`")
            st.write(f"**Raw model probability (spam):** `{round(probability, 4)}`")
            st.write(f"**Max sequence length:** {MAX_LEN} tokens")
            st.write(f"**Vocabulary size:** {MAX_WORDS} words")