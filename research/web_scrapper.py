from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup


class NewsSource:
    """Base class for news sources using RSS feeds"""
    label = None

    def __init__(self, name: str, url: str):
        self.name = name
        self.url = url
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                          'Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
        }

    def fetch(self, num_articles: int = 10):
        """Fetch articles from RSS source"""
        articles = []
        try:
            response = requests.get(self.url, headers=self.headers, timeout=15)
            response.raise_for_status()

            # Use 'xml' parser for RSS feeds
            soup = BeautifulSoup(response.content, 'xml')

            # In RSS, articles are inside <item> tags
            items = soup.find_all('item')[:num_articles]

            for item in items:
                title = item.find('title')
                description = item.find('description')
                link = item.find('link')

                if title and description:
                    # Clean up the description text
                    clean_text = BeautifulSoup(description.text, 'html.parser').get_text(strip=True)

                    # Ensure the text isn't empty after cleaning
                    if clean_text:
                        articles.append({
                            'source': self.name,
                            'title': title.text.strip(),
                            'text': clean_text,
                            'url': link.text.strip() if link else self.url,
                            'label': self.label,
                            'scraped_date': datetime.now().isoformat()
                        })

            print(f"Fetched {len(articles)} articles from {self.name}")

        except requests.exceptions.HTTPError as e:
            print(f"Blocked or Error fetching from {self.name}: {e.response.status_code}")
        except Exception as e:
            print(f"Error fetching from {self.name}: {str(e)}")

        return articles


class RealNewsSource(NewsSource):
    label = 1


class FakeNewsSource(NewsSource):
    label = 0


class WebNewsCollector:
    """Collect news from multiple sources"""

    REAL_NEWS_SOURCES = {
        'BBC_World': 'http://feeds.bbci.co.uk/news/world/rss.xml',
        'BBC_Tech': 'http://feeds.bbci.co.uk/news/technology/rss.xml',
        'NYT_World': 'https://rss.nytimes.com/services/xml/rss/nyt/World.xml',
        'NYT_US': 'https://rss.nytimes.com/services/xml/rss/nyt/US.xml',
        'NPR': 'https://feeds.npr.org/1001/rss.xml',
        'Guardian_World': 'https://www.theguardian.com/world/rss',
        'Guardian_US': 'https://www.theguardian.com/us-news/rss',
        'AlJazeera': 'https://www.aljazeera.com/xml/rss/all.xml',
        'WashingtonPost': 'https://feeds.washingtonpost.com/rss/world',
        'WallStreetJournal': 'https://feeds.a.dj.com/rss/RSSWorldNews.xml',
        'ABC_News': 'https://abcnews.go.com/abcnews/topstories',
        'CBS_News': 'https://www.cbsnews.com/latest/rss/main',
        'SkyNews': 'https://feeds.skynews.com/feeds/rss/world.xml',
        'NYTimes': 'http://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml',
        'FoxNews': 'http://feeds.foxnews.com/foxnews/latest',
        'NBCNews': 'http://feeds.nbcnews.com/feeds/topstories',
        'DailyMail': 'http://www.dailymail.co.uk/articles.rss',
        'Independent': 'http://www.independent.co.uk/rss',
        'NyPost': 'https://nypost.com/feed/',
        'Express': 'https://www.express.co.uk/posts/rss/77/news',
        'FinancialTimes': 'https://www.ft.com/world?format=rss',
        'NationalPublicRadio': 'https://feeds.npr.org/1004/rss.xml',
        'TheWashingtonPost': 'https://feeds.washingtonpost.com/rss/world',
        'TheChristianScienceMonitor': 'https://rss.csmonitor.com/feeds/world',
        'RFI': 'https://rfi.fr/en/international/rss',
        'Time': 'https://feeds.feedburner.com/time/world',
        'NBC_News': 'https://feeds.nbcnews.com/nbcnews/public/news',
        'TheGlobeAndMail': 'https://www.theglobeandmail.com/arc/outboundfeeds/rss/category/world/',
        'SBS_News': 'https://www.sbs.com.au/news/topic/world/feed',
        'GlobalNews': 'https://globalnews.ca/world/feed/',
        'ThePrint': 'https://theprint.in/category/world/feed/',
        'CNBC': 'https://www.cnbc.com/id/100727362/device/rss/rss.html',
        'DerSpiegel': 'https://www.spiegel.de/international/index.rss',
        'TheSydneyMorningHerald': 'https://www.smh.com.au/rss/world.xml',
        'LosAngelesTimes': 'https://www.latimes.com/world-nation/rss2.0.xml#nt=1col-7030col1',
        'NDTV': 'https://feeds.feedburner.com/ndtvnews-world-news',
        'TribuneIndia': 'https://publish.tribuneindia.com/newscategory/world/feed/',
        'TheHinduBusinessline': 'https://www.thehindubusinessline.com/news/world/feeder/default.rss',
        'BreakingNews': 'https://feeds.breakingnews.ie/bnworld',
        'ScrippsNews': 'https://www.scrippsnews.com/world.rss',
        '9News': 'https://www.9news.com.au/world/rss',
        'MaroccoWorldNews': 'https://www.moroccoworldnews.com/international/feed/',
        'DefenceBlog': 'https://defence-blog.com/feed/',
        'FOWorld': 'https://www.fairobserver.com/category/world-news/feed/',
        'TheWeek': 'https://www.theweek.in/news/world.feeds.rss.xml',
        'CNBCTV': 'https://www.cnbctv18.com/commonfeeds/v1/cne/rss/world.xml',
        'PatricaNews': 'https://cms.patrika.com/googlefeed/blog/category/world-news',
        'DailyMirror': 'https://www.mirror.co.uk/news/world-news/?service=rss',
        'DailyExpress': 'https://feeds.feedburner.com/daily-express-world-news',
        'USNN': 'https://www.usnn.news/feed/',
        'IFP': 'https://ifpnews.com/feed/',
        'BuzzFeed': 'https://www.buzzfeed.com/in/world.xml'
    }

    FAKE_NEWS_SOURCES = {
        'TheOnion': 'https://www.theonion.com/rss',
        'BabylonBee': 'https://babylonbee.com/feed',
        'DailyMash': 'https://www.thedailymash.co.uk/feed',
        'Beaverton': 'https://www.thebeaverton.com/feed/',
        'Chaser': 'https://chaser.com.au/feed/',
        'TheBabylonBee': 'https://babylonbee.com/feed',
        'TheDuffelBlog': 'https://duffelblog.com/feed',
        'TheChaser': 'https://chaser.com.au/feed',
        'WeeklyWorldNews': 'https://weeklyworldnews.com/feed',
        'NewsBiscuit': 'https://newsbiscuit.com/blog-feed.xml',
        'GomerBlog': 'https://gomerblog.com/feed',
        'EmpireNews': 'https://empirenews.net/feed',
        'GlossyNews': 'http://glossynews.com/feed/',
        'StaveFakeBallmer': 'https://stevefakeballmer.wordpress.com/feed/',
        'ThePokeNews': 'https://www.thepoke.com/category/news/rss',
        'TheDailyMash': 'https://www.thedailymash.co.uk/news/feed',
        'TheSpoof': 'https://www.thespoof.com/rss/feeds/world/rss.xml',
    }

    def __init__(self, output_dir: str = "configs/web_scraped_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def collect_real_news(self, num_per_source: int = 10):
        print(f"\n[Collecting Real News] {len(self.REAL_NEWS_SOURCES)} sources")
        all_articles = []
        for source_name, url in self.REAL_NEWS_SOURCES.items():
            source = RealNewsSource(source_name, url)
            articles = source.fetch(num_per_source)
            all_articles.extend(articles)

        df = pd.DataFrame(all_articles)
        print(f"Collected {len(df)} real news articles")
        return df

    def collect_fake_news(self, num_per_source: int = 10):
        print(f"\n[Collecting Fake News] {len(self.FAKE_NEWS_SOURCES)} sources")
        all_articles = []
        for source_name, url in self.FAKE_NEWS_SOURCES.items():
            source = FakeNewsSource(source_name, url)
            articles = source.fetch(num_per_source)
            all_articles.extend(articles)

        df = pd.DataFrame(all_articles)
        print(f"Collected {len(df)} fake news articles")
        return df

    def collect_all(self, num_per_source: int = 10):
        real_df = self.collect_real_news(num_per_source)
        fake_df = self.collect_fake_news(num_per_source)

        combined_df = pd.concat([real_df, fake_df], ignore_index=True)
        if not combined_df.empty:
            combined_df = combined_df.sample(frac=1).reset_index(drop=True)
        return combined_df

    def save_scraped_data(self, df: pd.DataFrame, filename: str = "web_scraped_news.csv"):
        if df.empty:
            print("No data to save!")
            return None

        filepath = self.output_dir / filename
        df.to_csv(filepath, index=False)
        print(f"Saved to {filepath}")
        return str(filepath)


def create_simulated_web_data(num_real: int = 50, num_fake: int = 50):
    """Create simulated web scraped data for testing"""

    real_news_samples = [
        "Latest scientific breakthrough in renewable energy discovered",
        "Government announces new healthcare initiatives",
        "Economic report shows positive growth",
        "International climate summit reaches agreement",
        "Major technological advancement in AI research",
        "Health officials recommend new vaccination guidelines",
        "Stock market reaches record highs",
        "New infrastructure project begins construction",
        "University research finds cure for disease",
        "World leaders meet for trade negotiations",
    ]

    fake_news_samples = [
        "Celebrities secretly controlling world governments",
        "Miracle cure suppressed by pharmaceutical companies",
        "Government plot to implant microchips",
        "Aliens found in secret underground base",
        "Conspiracy: Moon landing was faked",
        "Bill Gates plans mass vaccination plot",
        "Hidden truth about 5G towers revealed",
        "Secret society controls all media",
        "Toxic ingredient in drinking water",
        "Government experiment on citizens exposed",
    ]

    data = []

    # Real news
    for i in range(num_real):
        sample = np.random.choice(real_news_samples)
        data.append({
            'source': 'Real News Source',
            'title': sample,
            'text': f"{sample}. " * np.random.randint(2, 5),
            'url': 'https://real-news-source.com',
            'label': 1,
            'scraped_date': datetime.now().isoformat()
        })

    # Fake news
    for i in range(num_fake):
        sample = np.random.choice(fake_news_samples)
        data.append({
            'source': 'Fake News Source',
            'title': sample,
            'text': f"{sample}. " * np.random.randint(2, 5),
            'url': 'https://fake-news-source.com',
            'label': 0,
            'scraped_date': datetime.now().isoformat()
        })

    df = pd.DataFrame(data)
    return df.sample(frac=1).reset_index(drop=True)
