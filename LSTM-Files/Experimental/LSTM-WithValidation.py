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


normal_tensor = torch.tensor(normal, dtype=torch.float32)


window = 50 #The window of time the LSTM model will look at
x, y = [], []

for i in range(len(normal_tensor) - window):
    x.append(normal_tensor[i:i+window]) #X - input, every aspect of the tensor in i
    y.append(normal_tensor[i + window][3]) #Y - output the close price that is to be predicted from the tensor of the previous element

x = torch.stack(x)
y = torch.tensor(y).unsqueeze(-1)

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

# Total size of the input
total_size = len(x)

# Spliting the ratio in three: train, validation and test
train_ratio = 0.7
val_ratio = 0.15 
test_ratio = 0.15 

# Sizes to split the array
train_size = int(train_ratio * total_size)
val_size = int(val_ratio * total_size)
test_size = total_size - train_size - val_size

#Indexes from the beginning to train_size
x_train = x[:train_size]
y_train = y[:train_size]

#Indexes between train_size (inclusive) and val_size (exclusive)
x_val = x[train_size:train_size + val_size]
y_val = y[train_size:train_size + val_size]

#Indexes from train_size + val_size to the end
x_test = x[train_size + val_size:]
y_test = y[train_size + val_size:]


lstm = LSTM_Model(hidden_size=64)
optimizer = torch.optim.AdamW(lstm.parameters(), lr=0.01)
loss_fn = nn.MSELoss()
epochs = 100

train_losses = []
val_losses = []
test_losses = []

for epoch in range(epochs):
    # Training the data
    lstm.train()
    y_hat = lstm(x_train)
    train_loss = loss_fn(y_hat, y_train)

    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    # Validation
    lstm.eval()
    with torch.no_grad():
        y_val_pred = lstm(x_val)
        val_loss = loss_fn(y_val_pred, y_val)

    train_losses.append(train_loss.item())
    val_losses.append(val_loss.item())

    print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

#Plotting predicted graphs
with torch.no_grad():
    predictions = lstm(x_test)

#Adding dates to the graph

#Indicies from the dates of the original test set.
test_indices_original = range(window + train_size + val_size, window + len(x))
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

# Final evaluation on test set
with torch.no_grad():
    y_test_pred = lstm(x_test)
    test_loss = loss_fn(y_test_pred, y_test)
    print(f"\nFinal Test Loss: {test_loss.item():.4f}")


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