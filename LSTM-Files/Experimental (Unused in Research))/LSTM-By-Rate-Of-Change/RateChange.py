import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pandas as pd
import torch
from torch import nn
import numpy as np

# Parameters
torch.manual_seed(42)
np.random.seed(42)
data_period = "10y"
data_interval = "1d"
window = 90
split_ratio = 0.9
batch_size = 32
epochs = 50
learning_rate = 0.0001
hidden_size = 64
dropout = 0.2
num_layers = 2
symbol = "AAPL"

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data loading
ticker = yf.Ticker(symbol)
historical_data = ticker.history(period=data_period, interval=data_interval)
historical_data = historical_data[["Open", "High", "Low", "Close"]]
historical_data = historical_data.dropna()  # Critical fix

# Train-test split
split = int(split_ratio * len(historical_data))
train_data = historical_data[:split]
test_data = historical_data[split:]

# Scaling
scaler = StandardScaler()
train_normal = scaler.fit_transform(train_data)
test_normal = scaler.transform(test_data)

# Separate scaler for close prices
close_scaler = StandardScaler()
close_scaler.fit(train_data[['Close']])

# Tensor conversion
train_tensor = torch.tensor(train_normal, dtype=torch.float32)
test_tensor = torch.tensor(test_normal, dtype=torch.float32)

# Sequence creation
def create_seq(input_tensor, window_size):
    x, y = [], []
    for i in range(len(input_tensor) - window_size):
        x.append(input_tensor[i:i+window_size])
        y.append(input_tensor[i+window_size][3])
    return torch.stack(x), torch.tensor(y).unsqueeze(1)

x_train, y_train = create_seq(train_tensor, window)
x_test, y_test = create_seq(test_tensor, window)

# Model definition
class LSTM_Model(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(4, hidden_size, batch_first=True, 
                           dropout=dropout, num_layers=num_layers)
        self.linear = nn.Linear(hidden_size, 1)
        
        # Weight initialization
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)

    def forward(self, X):
        X, _ = self.lstm(X)
        X = X[:, -1, :] 
        return self.linear(X)

# Model setup
lstm = LSTM_Model(hidden_size).to(device)
optimizer = torch.optim.AdamW(lstm.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=5, verbose=True
)

# DataLoader
train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_dataset = TensorDataset(x_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Training loop
train_losses = []
test_losses = []
best_loss = float('inf')
patience = 10
patience_counter = 0

for epoch in range(epochs):
    lstm.train()
    total_train_loss = 0

    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        y_hat = lstm(x_batch)
        loss = loss_fn(y_hat, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    # Evaluation
    lstm.eval()
    total_test_loss = 0
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            y_hat = lstm(x_batch)
            loss = loss_fn(y_hat, y_batch)
            total_test_loss += loss.item()
    
    # Calculate losses
    avg_train_loss = total_train_loss / len(train_loader)
    avg_test_loss = total_test_loss / len(test_loader)
    train_losses.append(avg_train_loss)
    test_losses.append(avg_test_loss)
    
    # Learning rate scheduling
    scheduler.step(avg_test_loss)
    
    # Early stopping
    if avg_test_loss < best_loss:
        best_loss = avg_test_loss
        patience_counter = 0
        torch.save(lstm.state_dict(), 'best_model.pth')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f}, Test Loss: {avg_test_loss:.6f}")

# Load best model
lstm.load_state_dict(torch.load('best_model.pth'))

# Prediction
lstm.eval()
with torch.no_grad():
    x_test_device = x_test.to(device)
    predictions = lstm(x_test_device)
    predictions = predictions.cpu().numpy()

# Inverse transform using close scaler
predicted_prices = close_scaler.inverse_transform(predictions).flatten()
actual_prices = test_data['Close'].values[window:]

# Plotting
test_dates = test_data.index[window:]
plt.figure(figsize=(12, 6))
plt.plot(test_dates, actual_prices, label="Actual Price")
plt.plot(test_dates, predicted_prices, label="Predicted Price", alpha=0.7)
plt.title("AAPL: Actual vs Predicted Close Prices")
plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()