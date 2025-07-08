import yfinance as yf

import pandas as pd
import numpy as np

from datetime import date, datetime, timedelta

#Parameters
symbol = "AAPL"

data_period = "max"
data_interval = "1d" #How precise is the data being measured is

csv_path = "Dataset/APPL.csv"

timezone = "US/Eastern"

ticker = yf.Ticker(symbol)

def load_data(path):
    return pd.read_csv(path, parse_dates=["Date"])

def update_csv(path):
    data = load_data(path)
    #Converts items in the date category to datetime objects, instead of strings
    existing_dates = set(data["Date"])

    end_date = data["Date"][len(data) - 1] # Last day of stock

    import pytz

    current_date = datetime.now(pytz.timezone(timezone))

    # print(start_date)
    # print(end_date)
    # print(current_date)

    #The amount of time between the last date in the file and now
    time_gap_delta = current_date - end_date

    time_gap = time_gap_delta.days # The number of days as a number

    data = data[["Open", "High", "Low", "Close"]]
    data.reset_index(inplace=True)

    if time_gap <= 0:
        return

    if time_gap > 0:
        new_data = ticker.history(period=f"{time_gap}d", interval=data_interval)

        if not new_data.empty:
            new_data = new_data[["Open", "High", "Low", "Close"]].copy()
            new_data.reset_index(inplace=True)

            # Filter out any rows with a date already in the file
            new_data = new_data[~new_data["Date"].isin(existing_dates)]

            if not new_data.empty:
                new_data.to_csv(path, mode='a', header=False, index=False)

# This aims to put a sentimentally value into each available date
# This is so that each date can use news stories during time gaps
def map_trade_gaps(data):
# Dates historical data. (Input)
    available_dates = data["Date"]

    # Dates for each sentimentallity value (output)
    # output_dates = []

    # print(available_dates)

    # length = len(data) - 1

    mapped_dates = []

    mapped_dates.append([available_dates.iloc[0], []])

    for i in range(len(available_dates) - 1):
        current_date = available_dates[i]
        next_date = available_dates[i + 1]

        gap = (next_date - current_date).days - 1
        
        gap_dates = []
        if gap > 0:
            for j in range(1, gap + 1):
                gap_dates.append(current_date + timedelta(days=j))
        mapped_dates.append([next_date, gap_dates])

    return mapped_dates

historical_data = load_data(csv_path)

mapped_dates = map_trade_gaps(historical_data)

update_csv(csv_path)
