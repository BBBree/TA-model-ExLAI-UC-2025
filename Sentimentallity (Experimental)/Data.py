import nasdaqdatalink

nasdaqdatalink.ApiConfig.api_key = ""

data = nasdaqdatalink.get("NS1/FINSENTS_WEB_NEWS_SENTIMENT_SAMPLE")
print(data.head())
