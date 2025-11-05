import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import kagglehub
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, Dense

# Streamlit Page Setup
st.set_page_config(page_title="Crypto Price Prediction", page_icon="🪙", layout="wide")

st.title("🪙 Crypto Price Prediction using RNN + LSTM")
st.write("Select any cryptocurrency to see its predicted prices using AI models (RNN + LSTM).")

# ============================================
# 1️⃣ Load and Merge Dataset
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
# 2️⃣ Dropdown for Coin Selection
# ============================================
coins = sorted(data["Coin"].unique())
coin_name = st.selectbox("Select a Coin", coins, index=0)

coin_data = data[data["Coin"] == coin_name][["timestamp", "close"]].dropna().copy()
coin_data.set_index("timestamp", inplace=True)

# ============================================
# 3️⃣ Data Preprocessing
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
# 4️⃣ Build RNN and LSTM Models
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
    rnn_model = build_rnn()
    lstm_model = build_lstm()

    rnn_history = rnn_model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)
    lstm_history = lstm_model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)

# Compare model performance
rnn_loss = rnn_history.history["loss"][-1]
lstm_loss = lstm_history.history["loss"][-1]

best_model = rnn_model if rnn_loss < lstm_loss else lstm_model
best_model_name = "RNN" if rnn_loss < lstm_loss else "LSTM"

st.success(f"✅ {best_model_name} model selected based on better performance (lower loss).")

# ============================================
# 6️⃣ Predictions
# ============================================
predicted = best_model.predict(X_test)
predicted = scaler.inverse_transform(predicted.reshape(-1, 1))
actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# ============================================
# 7️⃣ Future 30-Day Forecast
# ============================================
future_input = scaled_data[-time_step:].reshape(1, time_step, 1)
future_predictions = []

for _ in range(30):
    next_price = best_model.predict(future_input)[0, 0]
    future_predictions.append(next_price)
    future_input = np.append(future_input[:, 1:, :], [[[next_price]]], axis=1)

future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# ============================================
# 8️⃣ Visualization
# ============================================
col1, col2 = st.columns(2)

with col1:
    st.subheader(f"📊 {coin_name} Actual vs Predicted Prices ({best_model_name} Model)")
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(actual, label="Actual Price", color='green')
    ax1.plot(predicted, label="Predicted Price", color='orange')
    ax1.set_xlabel("Days")
    ax1.set_ylabel("Price")
    ax1.legend()
    st.pyplot(fig1)

with col2:
    st.subheader(f"📈 Next 30 Days Forecast for {coin_name}")
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(range(1, 31), future_predictions, label="Predicted Future Price", color='red')
    ax2.set_xlabel("Days Ahead")
    ax2.set_ylabel("Predicted Price")
    ax2.legend()
    st.pyplot(fig2)


