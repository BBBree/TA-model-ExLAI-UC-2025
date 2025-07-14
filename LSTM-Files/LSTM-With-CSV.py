import yfinance as yf
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
import pandas as pd

import torch
from torch import nn

import numpy as np

torch.manual_seed(42)
np.random.seed(42)

#Parameters
data_period = 15 # In years
data_interval = "1d" # How precise is the data being measured is

window = 90 # The window of time the LSTM model will look at

split_ratio = 0.8

batch_size = 32 # DataLoader

# LSTM NN
epochs = 50
learning_rate = 0.001

hidden_size = 128
dropout = 0.2
num_layers = 1

symbol = "AAPL"

# #Parameters
# data_period = 10 # In years
# data_interval = "1d" # How precise is the data being measured is

# window = 90 # The window of time the LSTM model will look at

# split_ratio = 0.8

# batch_size = 32 # DataLoader

# # LSTM NN
# epochs = 25
# learning_rate = 0.001

# hidden_size = 128
# dropout = 0.2
# num_layers = 1

# symbol = "AAPL"



ticker = yf.Ticker(symbol)

historical_data = pd.read_csv("Dataset/APPL.csv", parse_dates=["Date"])
#Converts items in the date category to datetime objects, instead of strings
historical_data.set_index("Date", inplace=True)

features_num = len(historical_data.keys()) # For LSTM Model

# Makes historical_data only use a portion of the entire stock
if (data_period != "max"):
    import re

    end_date = historical_data.index.max() # Last date of historical data
    start_date = end_date - pd.DateOffset(years=data_period) # data_period years before end_date
    historical_data = historical_data.loc[historical_data.index >= start_date] # Elements after the start_date

# Split the indicies according to the split ratio
split = int(split_ratio * len(historical_data))

# Assign data points according to the split ratio
train_data = historical_data[:split]
test_data = historical_data[split:]

# Scaling both the test and train, separately, while using the fit from the train data
scaler = StandardScaler()
train_normal = scaler.fit_transform(train_data)
test_normal = scaler.transform(test_data)

# Convert the normalized sets into tensors
train_tensor = torch.tensor(train_normal, dtype=torch.float32)
test_tensor = torch.tensor(test_normal, dtype=torch.float32)


def create_seq(input_tensor, window_size):
    x, y = [], []

    for i in range(len(input_tensor) - window_size):
        x.append(input_tensor[i:i+window_size]) #X - input, every aspect of each window in i
        y.append(input_tensor[i+window_size][3]) #Y - output the close price that is to be predicted from the tensor of the previous element

    return torch.stack(x), torch.tensor(y).unsqueeze(1)

# Create train and test sequences for x and y
x_train, y_train = create_seq(train_tensor, window)
x_test, y_test = create_seq(test_tensor, window)


class LSTM_Model(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()

        self.re = nn.LSTM(features_num, hidden_size=hidden_size, batch_first=True, dropout=dropout, num_layers=num_layers)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, X):
        X, _ = self.re(X)
        X = X[:, -1, :] 
        X = self.linear(X)
        # X = self.relu(X)
        return X


lstm = LSTM_Model(hidden_size=hidden_size)
optimizer = torch.optim.AdamW(lstm.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()

train_losses = []
test_losses = []


from torch.utils.data import TensorDataset, DataLoader

train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

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

# A new scaler for inverse normalization with only the close prices
close_scaler = StandardScaler()
close_scaler.fit(train_data[['Close']])

with torch.no_grad():
    predictions = lstm(x_test)

predicted_prices = close_scaler.inverse_transform(predictions.numpy())

actual_prices = close_scaler.inverse_transform(y_test.numpy())

# Plotting
test_dates = test_data.index[window:]
plt.figure(figsize=(12, 6))
plt.plot(test_dates, actual_prices, label="Actual Close Price")
plt.plot(test_dates, predicted_prices, label="Predicted Close Price", alpha=0.7)
plt.title("Actual vs Predicted Close Prices (Test Set)")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()