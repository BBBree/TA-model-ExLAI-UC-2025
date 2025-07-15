import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import sklearn as sklearn
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn import model_selection

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

# Change

    # df = pd.read_csv("CNN/data.csv")
    # df = df[df["Company"] == "SBUX"] # select stock option

    # open_data = list()
    # close_data = list()
    # high_data = list()
    # low_data = list()
    # volume_data = list()

    # # fixing dataset format

    # start_index = df.index[0]

    # for i in range(len(df)):
    #     i += start_index
    #     open_data.append(float(df["Open"][i][1:]))
    #     close_data.append(float(df["Close/Last"][i][1:]))
    #     high_data.append(float(df["High"][i][1:]))
    #     low_data.append(float(df.Low[i][1:]))
    #     volume_data.append(float(df.Volume[i]))


    # open_data.reverse()
    # high_data.reverse()
    # low_data.reverse()
    # close_data.reverse()
    # volume_data.reverse()

# Change

# Changes in which file is being used
symbol = "AAPL"
df = pd.read_csv(f"Dataset/{symbol}.csv", parse_dates=["Date"])
#Converts items in the date category to datetime objects, instead of strings
df.set_index("Date", inplace=True)

# df["Date"] = pd.to_datetime(df["Date"], utc = True)
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

# Change


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
print(comparison_df)

# plt.plot(inv_price_prediction)
# plt.plot(inv_actual_price)
# plt.show()

plt.figure(figsize=(12, 6))
plt.plot(test_dates, inv_price_prediction, label="Predicted Close")
plt.plot(test_dates, inv_actual_price, label="Actual Close")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.title(f"{symbol} Closing Price Prediction")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
