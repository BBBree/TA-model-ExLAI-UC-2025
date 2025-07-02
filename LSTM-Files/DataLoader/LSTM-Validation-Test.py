import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import torch
from torch import nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

# 1. --- Data Loading and Splitting ---
symbol = "AAPL"
ticker = yf.Ticker(symbol)

historical_data = ticker.history(period="max", interval="1d")
historical_data = historical_data[["Open", "High", "Low", "Close"]]

# Split data into DataFrames first
train_ratio = 0.7
val_ratio = 0.15
n = len(historical_data)
train_end_idx = int(n * train_ratio)
val_end_idx = int(n * (train_ratio + val_ratio))

train_df = historical_data[:train_end_idx]
val_df = historical_data[train_end_idx:val_end_idx]
test_df = historical_data[val_end_idx:]

# 2. --- Scaling ---
# Fit scaler ONLY on training data, then transform all sets
scaler = MinMaxScaler()
# Note: scaler returns NumPy arrays
train_scaled = scaler.fit_transform(train_df)
val_scaled = scaler.transform(val_df)
test_scaled = scaler.transform(test_df)

# 3. --- Convert to Tensors ---
# THIS IS THE KEY FIX: Convert numpy arrays to tensors here
train_tensor = torch.tensor(train_scaled, dtype=torch.float32)
val_tensor = torch.tensor(val_scaled, dtype=torch.float32)
test_tensor = torch.tensor(test_scaled, dtype=torch.float32)

# 4. --- Create Windowed Sequences ---
def create_sequences(data, window_size):
    x, y = [], []
    for i in range(len(data) - window_size):
        x.append(data[i:i + window_size])
        y.append(data[i + window_size, 3]) # Target is the 'Close' price
    return torch.stack(x), torch.stack(y).unsqueeze(-1)

window = 50
x_train, y_train = create_sequences(train_tensor, window)
x_val, y_val = create_sequences(val_tensor, window)
x_test, y_test = create_sequences(test_tensor, window)

print(f"Train shapes: x={x_train.shape}, y={y_train.shape}")
print(f"Val shapes:   x={x_val.shape}, y={y_val.shape}")
print(f"Test shapes:  x={x_test.shape}, y={y_test.shape}")

# 5. --- DataLoader and Model ---
batch_size = 64
train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False)
test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

class LSTM_Model(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(4, hidden_size, batch_first=True, num_layers=2, dropout=0.2)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, X):
        X, _ = self.lstm(X)
        X = X[:, -1, :]
        X = self.linear(X)
        return X

# 6. --- Training Loop ---
lstm = LSTM_Model(hidden_size=64)
optimizer = torch.optim.AdamW(lstm.parameters(), lr=0.001)
loss_fn = nn.MSELoss()
epochs = 35

for epoch in range(epochs):
    lstm.train()
    for x_batch, y_batch in train_loader:
        y_hat = lstm(x_batch)
        loss = loss_fn(y_hat, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    lstm.eval()
    val_loss_total = 0
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            y_val_pred = lstm(x_batch)
            val_loss = loss_fn(y_val_pred, y_batch)
            val_loss_total += val_loss.item()

    avg_val_loss = val_loss_total / len(val_loader)
    print(f"Epoch {epoch + 1}/{epochs} - Val Loss: {avg_val_loss:.6f}")

# 7. --- Final Evaluation and Plotting ---
lstm.eval()
predictions_scaled = []
with torch.no_grad():
    for x_batch, y_batch in test_loader:
        y_test_pred = lstm(x_batch)
        predictions_scaled.append(y_test_pred.squeeze().numpy())

predictions_scaled = np.concatenate(predictions_scaled)
y_test_actual_scaled = y_test.squeeze().numpy()

# Prepare dummy array to inverse scale
pred_full = np.zeros((len(predictions_scaled), 4))
actual_full = np.zeros((len(y_test_actual_scaled), 4))
pred_full[:, 3] = predictions_scaled
actual_full[:, 3] = y_test_actual_scaled

# Invert scaling
predicted_prices = scaler.inverse_transform(pred_full)[:, 3]
actual_prices = scaler.inverse_transform(actual_full)[:, 3]

# Get the correct dates for the test set plot
test_dates = test_df.index[window:]

# Plotting
plt.figure(figsize=(14, 7))
plt.plot(test_dates, actual_prices, label="Actual Close Price", color='blue')
plt.plot(test_dates, predicted_prices, label="Predicted Close Price", color='red', linestyle='--')
plt.title("Actual vs Predicted Close Prices (Test Set)")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()