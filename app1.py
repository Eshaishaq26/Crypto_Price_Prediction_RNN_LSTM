# Crypto Price Prediction using RNN + LSTM


import os
import numpy as np
import pandas as pd
import kagglehub
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, Dense
import matplotlib.pyplot as plt
import streamlit as st

# Load and Merge Dataset
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
# Streamlit UI
st.title(" Crypto Price Prediction App (RNN + LSTM)")
st.write("Select any cryptocurrency to predict its future price trend.")

coins = sorted(data["Coin"].unique())
coin_name = st.selectbox("Select a Coin:", coins)

coin_data = data[data["Coin"] == coin_name][["timestamp", "close"]].dropna().copy()
coin_data.set_index("timestamp", inplace=True)
#Data Preprocessing
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
#Build RNN + LSTM Model

model = Sequential([
    SimpleRNN(50, return_sequences=True, input_shape=(time_step, 1)),
    LSTM(50),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)
# Predictions

predicted = model.predict(X_test)
predicted = scaler.inverse_transform(predicted.reshape(-1, 1))
actual = scaler.inverse_transform(y_test.reshape(-1, 1))
#Future Prediction (Next 30 Days)
future_input = scaled_data[-time_step:].reshape(1, time_step, 1)
future_predictions = []

for _ in range(30):
    next_price = model.predict(future_input)[0, 0]
    future_predictions.append(next_price)
    future_input = np.append(future_input[:, 1:, :], [[[next_price]]], axis=1)

future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
#Visualization
st.subheader(f"{coin_name} Actual vs Predicted Price")
plt.figure(figsize=(10, 5))
plt.plot(actual, label="Actual Price", color='green')
plt.plot(predicted, label="Predicted Price", color='orange')
plt.xlabel("Days")
plt.ylabel("Price")
plt.legend()
st.pyplot(plt)

st.subheader(f"{coin_name} Future 30-Day Prediction")
plt.figure(figsize=(10, 5))
plt.plot(range(1, 31), future_predictions, label="Predicted Future Price", color='red')
plt.xlabel("Days Ahead")
plt.ylabel("Predicted Price")
plt.legend()
st.pyplot(plt)



