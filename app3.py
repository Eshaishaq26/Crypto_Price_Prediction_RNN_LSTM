import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import kagglehub
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, Dense

st.set_page_config(page_title="Crypto Price Prediction", page_icon="🪙", layout="wide")

st.title("🪙 Crypto Price Prediction using RNN + LSTM")
st.write("Select any cryptocurrency to see its predicted prices using AI models (RNN + LSTM).")

# ============================================
# 1️⃣ Load Dataset (Download once)
# ============================================
@st.cache_data
def load_dataset():
    path = kagglehub.dataset_download("tr1gg3rtrash/time-series-top-100-crypto-currency-dataset")
    files = [f for f in os.listdir(path) if f.endswith(".csv")]
    dfs = []
    for f in files:
        df = pd.read_csv(os.path.join(path, f))
        df["Coin"] = f.split("-")[0]
        dfs.append(df)
    data = pd.concat(dfs, ignore_index=True)
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data.sort_values(["Coin", "timestamp"], inplace=True)
    return data

data = load_dataset()

# ============================================
# 2️⃣ Coin Selection
# ============================================
coins = sorted(data["Coin"].unique())
coin_name = st.selectbox("Select a Coin", coins, index=0)

coin_data = data[data["Coin"] == coin_name][["timestamp", "close"]].dropna().copy()
coin_data.set_index("timestamp", inplace=True)

# ============================================
# 3️⃣ Preprocessing
# ============================================
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(coin_data)

def create_sequences(dataset, time_step=60):
    X, y = [], []
    for i in range(len(dataset) - time_step - 1):
        X.append(dataset[i:(i + time_step), 0])
        y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 60
X, y = create_sequences(scaled_data, time_step)
X = X.reshape(X.shape[0], X.shape[1], 1)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# ============================================
# 4️⃣ Define Models
# ============================================
def build_rnn():
    model = Sequential([
        SimpleRNN(50, return_sequences=True, input_shape=(time_step, 1)),
        SimpleRNN(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def build_lstm():
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(time_step, 1)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# ============================================
# 5️⃣ Train Both Models
# ============================================
with st.spinner(f"Training RNN and LSTM models for {coin_name}..."):
    rnn_model = bu_
