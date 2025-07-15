import yfinance as yf

import pandas as pd
import numpy as np

from datetime import date

#Parameters
symbol = "AAPL"

data_period = "max"
data_interval = "1d" #How precise is the data being measured is

ticker = yf.Ticker(symbol)

historical_data = ticker.history(period=data_period, interval=data_interval)
historical_data = historical_data[["Open", "High", "Low", "Close", "Volume"]]

historical_data.reset_index(inplace=True)

historical_data.to_csv("Dataset/AAPL.csv", index=False)