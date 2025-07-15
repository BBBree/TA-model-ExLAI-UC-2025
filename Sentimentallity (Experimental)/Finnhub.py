import finnhub

api_key = ""

client = finnhub.Client(api_key=api_key)

from datetime import datetime, timedelta

from_date = (datetime.now() - timedelta(days=1000)).strftime('%Y-%m-%d')
to_date = datetime.now().strftime('%Y-%m-%d')

news = client.company_news('AAPL', _from=from_date, to=to_date)

for article in news:
    print(article["datetime"], article["headline"])



 