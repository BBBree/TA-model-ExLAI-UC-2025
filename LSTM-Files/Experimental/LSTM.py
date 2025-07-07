import yfinance as yf
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
import pandas as pd

import torch
from torch import nn

import numpy as np

symbol = "AAPL"

ticker = yf.Ticker(symbol)

historical_data = ticker.history(period="max", interval="1d")
historical_data = historical_data[["Open", "High", "Low", "Close"]]

scaler = MinMaxScaler()
normal = scaler.fit_transform(historical_data)

# print(type(normal))

normal_tensor = torch.tensor(normal, dtype=torch.float32)

# print(normal_tensor)

window = 100 #The window of time the LSTM model will look at
x, y = [], []

for i in range(len(normal_tensor) - window):
    x.append(normal_tensor[i:i+window]) #X - input, every aspect of the tensor in i
    y.append(normal_tensor[i + window][3]) #Y - output the close price that is to be predicted from the tensor of the previous element

x = torch.stack(x)             # Shape: (N, 1, 4)
y = torch.tensor(y).unsqueeze(-1)  # Shape: (N, 1)

class LSTM_Model(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()

        self.re = nn.LSTM(4, hidden_size=hidden_size, batch_first=True)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, X):
        X, _ = self.re(X)
        X = X[:, -1, :] 
        X = self.linear(X)
        # X = self.relu(X)
        return X

#Split the data into a train and a test set
split = round(0.8 * len(x))

#Splitting the data from x and y, with each getting a train and test
x_train = x[:split]
y_train = y[:split]
x_test = x[split:]
y_test = y[split:]

# print(len(normal_tensor))
# print(split)

# print(split / len(normal_tensor))

lstm = LSTM_Model(hidden_size=64)
optimizer = torch.optim.AdamW(lstm.parameters(), lr=0.001)
loss_fn = nn.MSELoss()
epochs = 150

train_losses = []
test_losses = []
for i in range(epochs):
    y_hat = lstm(x_train)

    train_loss = loss_fn(y_hat, y_train)

    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()


    with torch.no_grad():
        # Evaluate on test set
        y_hat = lstm(x_test)

        test_loss = loss_fn(y_hat, y_test)

    train_losses.append(train_loss.item())
    test_losses.append(test_loss.item())

    print(f"Epoch {i + 1}/{epochs} - Train Loss: {train_loss.item():.4f}, Test Loss: {test_loss.item():.4f}")

#Plotting predicted graphs
with torch.no_grad():
    predictions = lstm(x_test)

#Indicies from the dates of the original test set.
test_indices_original = range(window + len(x_train), window + len(x))
test_dates = historical_data.index[test_indices_original]

predictions = predictions.squeeze().numpy()
y_test_actual = y_test.squeeze().numpy()

# Prepare dummy array to inverse scale
pred_full = np.zeros((len(predictions), 4))
actual_full = np.zeros((len(y_test_actual), 4))

# Fill in the predicted close price at index 3
pred_full[:, 3] = predictions
actual_full[:, 3] = y_test_actual

# Invert scaling
predicted_prices = scaler.inverse_transform(pred_full)[:, 3]
actual_prices = scaler.inverse_transform(actual_full)[:, 3]

#Ploting
plt.figure(figsize=(12, 6))
plt.plot(test_dates, actual_prices, label="Actual Close Price")
plt.plot(test_dates, predicted_prices, label="Predicted Close Price")
plt.title("Actual vs Predicted Close Prices (Test Set)")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)  # Rotate date labels for readability
plt.tight_layout()
plt.show()


# plt.figure(figsize=(12, 6))
# plt.plot(actual_prices, label="Actual Close Price")
# plt.plot(predicted_prices, label="Predicted Close Price")
# plt.title("Actual vs Predicted Close Prices (Test Set)")
# plt.xlabel("Time (Days)")
# plt.ylabel("Price (USD)")
# plt.legend()
# plt.grid(True)
# plt.show()

