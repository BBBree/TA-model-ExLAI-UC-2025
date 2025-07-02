import yfinance as yf
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
import pandas as pd

import torch
from torch import nn

import numpy as np

# TODO: Maybe, make the output be the rate of increase (as a percent) instead of the actual close price

#Parameters
data_period = "10y"
data_interval = "1d" #How precise is the data being measured is

window = 90 #The window of time the LSTM model will look at

split_ratio = 0.9

batch_size = 32 #DataLoader

#LSTM NN
epochs = 300
learning_rate = 0.0001

hidden_size = 64
dropout = 0.2
num_layers = 1

symbol = "AAPL"

ticker = yf.Ticker(symbol)

historical_data = ticker.history(period=data_period, interval=data_interval)
historical_data = historical_data[["Open", "High", "Low", "Close"]]

split = int(split_ratio * len(historical_data))

train_data = historical_data[:split]
test_data = historical_data[split:]

scaler = StandardScaler()
train_normal = scaler.fit_transform(train_data)
test_normal = scaler.transform(test_data)

normal = np.concatenate([train_normal, test_normal], axis=0)

normal_tensor = torch.tensor(normal, dtype=torch.float32)


x, y = [], []

for i in range(len(normal_tensor) - window):
    x.append(normal_tensor[i:i+window]) #X - input, every aspect of the tensor in i
    y.append(normal_tensor[i + window][3]) #Y - output the close price that is to be predicted from the tensor of the previous element

x = torch.stack(x)             # Shape: (N, 1, 4)
y = torch.tensor(y).unsqueeze(-1)  # Shape: (N, 1)

class LSTM_Model(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()

        self.re = nn.LSTM(4, hidden_size=hidden_size, batch_first=True, dropout=dropout, num_layers=num_layers)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, X):
        X, _ = self.re(X)
        X = X[:, -1, :] 
        X = self.linear(X)
        # X = self.relu(X)
        return X

adjusted_split = split - window

x_train = x[:adjusted_split]
y_train = y[:adjusted_split]
x_test = x[adjusted_split:]
y_test = y[adjusted_split:]

# print(len(normal_tensor))
# print(split)

# print(split / len(normal_tensor))

lstm = LSTM_Model(hidden_size=64)
optimizer = torch.optim.AdamW(lstm.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()

train_losses = []
test_losses = []

from torch.utils.data import TensorDataset, DataLoader


train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TensorDataset(x_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


for epoch in range(epochs):
    lstm.train() # Training mode
    total_train_loss = 0
    for x_batch, y_batch in train_loader:
        y_hat = lstm(x_batch)
        loss = loss_fn(y_hat, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    # Evaluation
    lstm.eval() # Evaluation mode
    total_test_loss = 0
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            y_hat = lstm(x_batch)
            loss = loss_fn(y_hat, y_batch)
            total_test_loss += loss.item()
    
    avg_train_loss = total_train_loss / len(train_loader)
    avg_test_loss = total_test_loss / len(test_loader)
    
    train_losses.append(avg_train_loss)
    test_losses.append(avg_test_loss)

    print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}")

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

