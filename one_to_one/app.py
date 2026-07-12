"""
🌾 Rain → Grain — One-to-One Simple RNN
A single rainfall reading flows through one recurrent cell and becomes
a single crop-yield estimate. One input. One output. One idea, visualized.

Run with:  streamlit run app.py
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "yield_df.csv")
MODEL_PATH = os.path.join(BASE_DIR, "crop_yield_rnn.keras")

st.set_page_config(page_title="Rain -> Grain | One-to-One RNN", page_icon="🌾", layout="centered")

# ──────────────────────────────────────────────────────────────────────────
# DATA + MODEL (silent, cached — no controls exposed)
# ──────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)[["average_rain_fall_mm_per_year", "hg/ha_yield"]].dropna()
    df.columns = ["rainfall_mm", "yield_hg_ha"]
    df["yield_tons_ha"] = df["yield_hg_ha"] * 0.0001
    df = df[(df.rainfall_mm > 0) & (df.yield_tons_ha > 0)]
    # Collapse to one yield value per unique rainfall reading, then smooth.
    # Raw rainfall <-> yield is extremely noisy (yield mixes many crops,
    # countries, and farming conditions), so training directly on it makes
    # the network collapse to predicting the mean for every input. A rolling
    # smooth over the sorted, de-duplicated rainfall axis keeps the real,
    # data-driven trend while removing the noise that was drowning it out.
    agg = df.groupby("rainfall_mm", as_index=False)["yield_tons_ha"].mean()
    agg = agg.sort_values("rainfall_mm").reset_index(drop=True)
    agg["yield_smooth"] = agg["yield_tons_ha"].rolling(window=17, center=True, min_periods=1).mean()
    return df, agg

df_raw, df = load_data()

@st.cache_resource(show_spinner=False)
def get_model_and_scalers():
    x_scaler, y_scaler = MinMaxScaler(), MinMaxScaler()
    X = x_scaler.fit_transform(df[["rainfall_mm"]])
    y = y_scaler.fit_transform(df[["yield_smooth"]])
    X = X.reshape(-1, 1, 1)

    if os.path.exists(MODEL_PATH):
        model = keras.models.load_model(MODEL_PATH)
    else:
        model = Sequential([
            SimpleRNN(24, activation="tanh", input_shape=(1, 1)),
            Dense(16, activation="relu"),
            Dense(1, activation="linear"),
        ])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss="mse")
        early_stop = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=50, restore_best_weights=True
        )
        model.fit(X, y, validation_split=0.15, epochs=500, batch_size=8,
                  verbose=0, shuffle=True, callbacks=[early_stop])
        model.save(MODEL_PATH)
    return model, x_scaler, y_scaler

with st.spinner("Calibrating the rain to grain neural pathway..."):
    model, x_scaler, y_scaler = get_model_and_scalers()

# ──────────────────────────────────────────────────────────────────────────
# PREMIUM CSS — soft, muted palette. Injected with st.html so the raw
# <style> block is never run through markdown text parsing (which is what
# caused the CSS to leak onto the page as plain text before).
# ──────────────────────────────────────────────────────────────────────────
st.html("""
<link href="https://fonts.googleapis.com/css2?family=Sora:wght@500;700;800&family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
:root{
  --sage:#8fbfa0; --sky:#9db8d9; --clay:#e0b48e; --ivory:#eee7db;
}
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }

.stApp {
  background:
    radial-gradient(circle at 18% -10%, rgba(143,191,160,0.16), transparent 45%),
    radial-gradient(circle at 88% 8%, rgba(224,180,142,0.10), transparent 42%),
    linear-gradient(180deg, #12181a 0%, #161f1c 55%, #12181a 100%);
  color: var(--ivory);
}
.block-container { padding-top: 2.2rem; max-width: 760px; }

.eyebrow {
  text-align: center; color: var(--clay); letter-spacing: .22em; font-size: .7rem;
  font-weight: 600; text-transform: uppercase; margin-bottom: .4rem;
}
.title {
  text-align: center; font-family: 'Sora', sans-serif; font-weight: 800; font-size: 2.4rem;
  color: #f4efe6; margin: 0 0 .5rem 0;
}
.title .accent { color: var(--sage); }
.subtitle {
  text-align: center; color: #a9b3ac; font-size: .98rem; max-width: 520px;
  margin: 0 auto 2.2rem auto; line-height: 1.5;
}

.flow-wrap { margin: 1.6rem 0 1.8rem 0; }
.node { text-align: center; }
.node-circle {
  width: 104px; height: 104px; border-radius: 50%; margin: 0 auto .6rem auto;
  display: flex; align-items: center; justify-content: center; font-size: 2rem;
  border: 1.5px solid rgba(238,231,219,0.14);
  background: radial-gradient(circle at 35% 30%, rgba(238,231,219,0.05), rgba(238,231,219,0.01));
  position: relative;
}
.node-circle.input { box-shadow: 0 0 26px -8px rgba(157,184,217,0.55); border-color: rgba(157,184,217,0.4); }
.node-circle.cell {
  box-shadow: 0 0 30px -6px rgba(143,191,160,0.6); border-color: rgba(143,191,160,0.45);
  animation: pulse 3.2s ease-in-out infinite;
}
.node-circle.output { box-shadow: 0 0 26px -8px rgba(224,180,142,0.55); border-color: rgba(224,180,142,0.42); }
@keyframes pulse {
  0%, 100% { box-shadow: 0 0 24px -6px rgba(143,191,160,0.45); }
  50% { box-shadow: 0 0 38px -2px rgba(143,191,160,0.7); }
}
.node-label { font-size: .7rem; letter-spacing: .1em; text-transform: uppercase; color: #93a29a; }
.node-value { font-family: 'Sora', sans-serif; font-weight: 700; font-size: 1.02rem; color: #f4efe6; margin-top: .15rem; }
.loop-arrow { position: absolute; top: -14px; right: -12px; font-size: .9rem; color: var(--sage); opacity: .8; }

.flow-line {
  height: 2px; margin-top: 51px; position: relative; overflow: hidden; border-radius: 2px;
  background: rgba(238,231,219,0.07);
}
.flow-line-glow {
  position: absolute; top: 0; left: -40%; height: 100%; width: 40%;
  background: linear-gradient(90deg, transparent, rgba(143,191,160,0.75), transparent);
  animation: travel 2.4s linear infinite;
}
.flow-line-glow.right { animation-delay: 1.2s; }
@keyframes travel { 0% { left: -40%; } 100% { left: 100%; } }

.formula {
  text-align: center; font-family: 'Sora', sans-serif; color: #a7c9b3; font-size: .82rem;
  letter-spacing: .02em; margin: .4rem 0 2.2rem 0; opacity: .8;
}

.result-card {
  background: linear-gradient(135deg, rgba(143,191,160,0.14), rgba(224,180,142,0.07));
  border: 1px solid rgba(143,191,160,0.3); border-radius: 22px; padding: 2rem; text-align: center;
  box-shadow: 0 25px 55px -30px rgba(143,191,160,0.4);
}
.result-label { color: #a3b3aa; font-size: .76rem; text-transform: uppercase; letter-spacing: .12em; }
.result-value {
  font-family: 'Sora', sans-serif; font-weight: 800; font-size: 2.9rem; margin-top: .3rem;
  color: #cfe6d7;
}
.result-unit { color: #9fada5; font-size: .93rem; margin-top: .1rem; }

.stSlider [data-baseweb="slider"] { margin-top: .5rem; }
label[data-testid="stWidgetLabel"] p { color: #cdd6cf !important; font-weight: 500; }
</style>
""")

# ──────────────────────────────────────────────────────────────────────────
# HERO
# ──────────────────────────────────────────────────────────────────────────
st.html("""
<div class="eyebrow">One-to-One Simple RNN</div>
<div class="title">Rain 🌧️ becomes <span class="accent">Grain</span> 🌾</div>
<div class="subtitle">One rainfall reading enters a single recurrent cell, which folds it
through its learned memory and releases one number: the expected crop yield.</div>
""")

# ──────────────────────────────────────────────────────────────────────────
# INPUT (single control)
# ──────────────────────────────────────────────────────────────────────────
rainfall = st.slider(
    "Rainfall (mm / year)",
    float(df_raw.rainfall_mm.min()), float(df_raw.rainfall_mm.max()),
    float(df_raw.rainfall_mm.mean()), step=1.0,
)

x_scaled = x_scaler.transform([[rainfall]]).reshape(1, 1, 1)
pred_scaled = model.predict(x_scaled, verbose=0)
pred = float(y_scaler.inverse_transform(pred_scaled)[0][0])

# ──────────────────────────────────────────────────────────────────────────
# CONCEPT FLOW DIAGRAM (Input -> RNN cell -> Output)
# ──────────────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns([2, 1, 2, 1, 2])
with c1:
    st.html(f"""
    <div class="node">
      <div class="node-circle input">🌧️</div>
      <div class="node-label">Input · x</div>
      <div class="node-value">{rainfall:.0f} mm</div>
    </div>""")
with c2:
    st.html('<div class="flow-line"><div class="flow-line-glow"></div></div>')
with c3:
    st.html("""
    <div class="node">
      <div class="node-circle cell">🔁<span class="loop-arrow">↻</span></div>
      <div class="node-label">Simple RNN Cell</div>
      <div class="node-value">h = tanh(Wx+Uh)</div>
    </div>""")
with c4:
    st.html('<div class="flow-line"><div class="flow-line-glow right"></div></div>')
with c5:
    st.html(f"""
    <div class="node">
      <div class="node-circle output">🌾</div>
      <div class="node-label">Output · y</div>
      <div class="node-value">{pred:.2f} t/ha</div>
    </div>""")

st.html('<div class="formula">y = Wy &middot; tanh(Wx&middot;x + Wh&middot;h + b) + c</div>')

# ──────────────────────────────────────────────────────────────────────────
# RESULT
# ──────────────────────────────────────────────────────────────────────────
st.html(f"""
<div class="result-card">
  <div class="result-label">Predicted Crop Yield</div>
  <div class="result-value">{pred:.2f}</div>
  <div class="result-unit">tons / hectare, for {rainfall:.0f} mm of rainfall</div>
</div>
""")