import yfinance as yf

import matplotlib.pyplot as plt
 
import sklearn as sklearn
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection

import pandas as pd
import numpy as np

import torch
from torch import nn
import torch.nn as nn
import torch.optim as optim

#LSTM -------------------------------------------------------------

torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#Parameters
data_period = 10 # In years
data_interval = "1d" # How precise is the data being measured is

window = 60 # The window of time the LSTM model will look at

split_ratio = 0.8

batch_size = 32 # DataLoader

# LSTM NN
epochs = 32
learning_rate = 0.001

hidden_size = 128
dropout = 0.2
num_layers = 1

symbol = "AAPL"
from_file = True #Whether to use yfinance or a file for the historical data

# Reading from file ---------------------------------------------------

if (from_file != False):
    ticker = yf.Ticker(symbol)
    historical_data = ticker.history(period=f"{data_period}y", interval=data_interval)
    historical_data = historical_data[["Open", "High", "Low", "Close", "Volume"]]

else:
    historical_data = pd.read_csv(f"Dataset/{symbol}.csv", parse_dates=["Date"])
    #Converts items in the date category to datetime objects, instead of strings
    historical_data.set_index("Date", inplace=True)

features_num = len(historical_data.keys()) # For LSTM Model

# Makes historical_data only use a portion of the entire stock
if (data_period != "max"):
    import re

    end_date = historical_data.index.max() # Last date of historical data
    start_date = end_date - pd.DateOffset(years=data_period) # data_period years before end_date
    historical_data = historical_data.loc[historical_data.index >= start_date] # Elements after the start_date

# Splitting data ----------------------------------------------------------

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


# Model Trainging -------------------------------------------------------------

class LSTM_Model(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()

        if (num_layers > 1):
            self.re = nn.LSTM(features_num, hidden_size=hidden_size, batch_first=True, num_layers=num_layers, dropout=dropout)
        else:
            self.re = nn.LSTM(features_num, hidden_size=hidden_size, batch_first=True, num_layers=num_layers)
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

test_dates_LSTM = test_data.index[window:]


#CNN -------------------------------------------------------------

torch.manual_seed(42)
np.random.seed(42)

# parameters
input_channels = 5 # open, high, low, close, volume
activation_function = nn.LeakyReLU()
learning_rate = 0.001
epochs = 1000
loss_function = nn.MSELoss()
window_size = 15 # 15 day window
scaler = StandardScaler()

data_period = 10 # In years

# Changes in which file is being used
df = pd.read_csv(f"Dataset/{symbol}.csv", parse_dates=["Date"])
#Converts items in the date category to datetime objects, instead of strings
df.set_index("Date", inplace=True)

df = df.sort_values("Date")

if (data_period != "max"):
    import re

    end_date = df.index.max() # Last date of historical data
    start_date = end_date - pd.DateOffset(years=data_period) # data_period years before end_date
    df = df.loc[df.index >= start_date] # Elements after the start_date

# Extract data directly
open_data = df["Open"].tolist()
high_data = df["High"].tolist()
low_data = df["Low"].tolist()
close_data = df["Close"].tolist()
volume_data = df["Volume"].tolist()


data = []

# first we fit our scaler to our mean and standard deviation
scaled_data = scaler.fit_transform(
  torch.tensor([open_data, high_data, low_data, close_data, volume_data]).T
)

# this gets the closing price to predict but we don't want the first windows input data 
target = torch.tensor(scaled_data[window_size:, 3:4], dtype=torch.float)

# this selects our windows and adds them to one list
for i in range(len(open_data) - window_size):
  data.append([
    open_data[i:i + window_size],
    high_data[i:i + window_size], 
    low_data[i:i + window_size], 
    close_data[i:i + window_size], 
    volume_data[i:i + window_size]])

# in order to scale our data we have to swap the rows and columns around
data = torch.tensor(data).permute([0, 2, 1])

# this scales each window to our mean and standard deviation
for i in range(len(data)):
  data[i] = torch.tensor(scaler.transform(data[i]))

# change it back
data = data.permute([0, 2, 1])

windowed_dates = df.index[window_size:]

# Now split with same size as train/test targets
  
train_x, test_x, train_y, test_y = sklearn.model_selection.train_test_split(data, target, random_state=7, shuffle=False)

split_index = len(train_y)
train_dates = windowed_dates[:split_index]
test_dates = windowed_dates[split_index:]

class Net(nn.Module): # the nn.Module is set up as the parent class for the Net Class
    def __init__(self):
        super(Net, self).__init__() # This calls the init method of the nn.Module parent class to ensure that its been initialized

        self.model = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=64, kernel_size=2, stride=1),
            activation_function,
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=2, stride=1),
            activation_function,
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(in_features=384, out_features=128),
            activation_function,
            nn.Linear(128, 64),
            activation_function,
            nn.Linear(64, 1),
        )

    def forward(self, x):

        return self.model(x)


net = Net()

def train(model: nn.Module, train_x, train_y, epochs, learning_rate):
  optimizer = optim.Adam(net.parameters(), lr=learning_rate)

  for epoch in range(epochs):
    y_hat = net(train_x)
    loss = loss_function(y_hat, train_y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0: # print out the loss as the model is training
      print(f"Epoch: {epoch}, Loss: {loss.item():.10f}")

train(net, train_x, train_y, epochs, learning_rate)

y_hat = net(test_x)
print(loss_function(y_hat, test_y))

price_prediction = y_hat.tolist()
# Convert price_prediction to numpy array and inverse transform only the closing price column

# Convert to numpy array and ensure shape is (n_samples, 1)
price_prediction_np = np.array(price_prediction).reshape(-1, 1)

# Create a placeholder array with the same number of features as original data
# Fill with zeros, then set the close price column (index 3) to the predicted values
full_pred = np.zeros((price_prediction_np.shape[0], scaled_data.shape[1]))
full_pred[:, 3] = price_prediction_np[:, 0]

# Inverse transform
inv_price_prediction = scaler.inverse_transform(full_pred)[:, 3]
# Inverse transform the actual test_y values
test_y_np = test_y.numpy().reshape(-1, 1)
full_actual = np.zeros((test_y_np.shape[0], scaled_data.shape[1]))
full_actual[:, 3] = test_y_np[:, 0]
inv_actual_price = scaler.inverse_transform(full_actual)[:, 3]

# Create a DataFrame to compare predictions and actual prices
comparison_df = pd.DataFrame({
    'Predicted_Close': inv_price_prediction,
    'Actual_Close': inv_actual_price
})
pd.set_option('display.max_rows', None)
# print(comparison_df)

font_size = 15

plt.figure(figsize=(4, 3))

plt.plot(test_dates, inv_price_prediction, label="CNN Predicted Close Price", color = "Orange")
plt.plot(test_dates, inv_actual_price, label="Actual Close", color = "Blue")

plt.plot(test_dates_LSTM, actual_prices, color = "Blue")
plt.plot(test_dates_LSTM, predicted_prices, label="LSTM Predicted Close Price", alpha=0.7, color = "Red")

plt.xlabel("Date", fontsize = font_size)
plt.ylabel("Price (USD)", fontsize = font_size)
plt.title(f"{symbol} Closing Price Prediction", fontsize = font_size)
plt.legend()
plt.xticks(rotation=45)
plt.xticks(fontsize = font_size)
plt.yticks(fontsize = font_size)
plt.tight_layout()
plt.grid(True)
plt.show()
