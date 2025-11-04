# Crypto Price Prediction App (RNN + LSTM)
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import kagglehub
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, Dense

# Streamlit UI


st.set_page_config(page_title="Crypto Price Predictor", layout="wide")
st.title("🪙 Crypto Price Prediction using RNN + LSTM")
st.markdown("### Predict next 30 days' prices for any top cryptocurrency")
# Load and Merge Dataset


@st.cache_data
def load_data():
    st.info("Downloading dataset from Kaggle...")
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
    
    return data, sorted(data["Coin"].unique())

data, coins = load_data()
st.success(f"Dataset loaded successfully with {len(coins)} coins!")

#Select Coin


coin_name = st.selectbox("Select a Coin", coins, index=0)
coin_data = data[data["Coin"] == coin_name][["timestamp", "close"]].dropna().copy()
coin_data.set_index("timestamp", inplace=True)

st.line_chart(coin_data["close"], height=250, use_container_width=True)
st.caption(f"Showing closing price trend for **{coin_name}**")

# Data Preprocessing


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

#Build + Train Model

st.info("🧠 Training RNN + LSTM model... please wait (~1–2 mins)")
model = Sequential([
    SimpleRNN(50, return_sequences=True, input_shape=(time_step, 1)),
    LSTM(50),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)
st.success("Model training completed!")
#Predictions


predicted = model.predict(X_test)
predicted = scaler.inverse_transform(predicted.reshape(-1, 1))
actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plot Actual vs Predicted
fig1, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(actual, label="Actual", color='green')
ax1.plot(predicted, label="Predicted", color='orange')
ax1.set_title(f"{coin_name} — Actual vs Predicted Prices")
ax1.set_xlabel("Days")
ax1.set_ylabel("Price")
ax1.legend()
st.pyplot(fig1)

#Predict Next 30 Days

future_input = scaled_data[-time_step:].reshape(1, time_step, 1)
future_predictions = []

for _ in range(30):
    next_price = model.predict(future_input)[0, 0]
    future_predictions.append(next_price)
    future_input = np.append(future_input[:, 1:, :], [[[next_price]]], axis=1)

future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
# Plot Future Forecast
fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.plot(range(1, 31), future_predictions, color='red')
ax2.set_title(f"{coin_name} — 30-Day Future Forecast")
ax2.set_xlabel("Days Ahead")
ax2.set_ylabel("Predicted Price")
st.pyplot(fig2)

st.success("Prediction Completed Successfully!")
st.balloons()
