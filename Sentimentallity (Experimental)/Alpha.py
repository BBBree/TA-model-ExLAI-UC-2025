import requests
from datetime import datetime

# replace the "demo" apikey below with your own key from https://www.alphavantage.co/support/#api-key

limit = 1000
symbol = "AAPL"
api_key = ""

url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&limit={limit}&apikey={api_key}"
r = requests.get(url)
data = r.json()

dat = []
for article in data.get("feed", []):
    # Parse date
    raw_date = article.get("time_published", "")
    date = datetime.strptime(raw_date, "%Y%m%dT%H%M%S").date() if raw_date else "Unknown date"
    print(date)
    dat.append(date)

print(len(dat))
