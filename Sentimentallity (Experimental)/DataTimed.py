import feedparser
from newspaper import Article
from datetime import datetime, timezone

rss_url = "https://feeds.finance.yahoo.com/rss/2.0/headline?s=AAPL&region=US&lang=en-US"

feed = feedparser.parse(rss_url)

"""
article = Article(url)

article.download()
article.parse()

print(article.title)
print(article.publish_date)
print(article.text)
"""

for entry in feed.entries:
    url = entry.link

    try:
        article = Article(url)
        article.download()
        article.parse()

        # print("Title:", article.title)
        print("Publish date:", article.publish_date)
        # print("Text (preview):", article.text[:200], "\n")  # Print a preview
    except Exception as e:
        print(f"Failed to process {url}: {e}")