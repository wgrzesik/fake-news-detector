import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from newspaper import Article, ArticleException, Config as NewspaperConfig


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

    @staticmethod
    def _fetch_full_text(url: str, timeout: int = 15) -> str:
        """Follow article URL and extract the full body text using newspaper3k."""
        try:
            config = NewspaperConfig()
            config.browser_user_agent = (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/120.0.0.0 Safari/537.36'
            )
            config.request_timeout = timeout
            config.fetch_images = False

            article = Article(url, config=config)
            article.download()
            article.parse()
            text = (article.text or "").strip()
            if len(text) > 100:
                return text
        except (ArticleException, Exception) as e:
            print(f"  Could not fetch full text from {url}: {e}")
        return ""

    def fetch(self, num_articles: int = 10, fetch_full_text: bool = False,
              delay: float = 0.5):
        """Fetch articles from RSS source.

        Args:
            num_articles: max items to retrieve from the RSS feed.
            fetch_full_text: if True, follow each article link and scrape
                             the full body text (saved in 'full_text' column).
            delay: seconds to wait between full-text requests (rate-limiting).
        """
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
                        article_url = link.text.strip() if link else self.url

                        entry = {
                            'source': self.name,
                            'title': title.text.strip(),
                            'text': clean_text,
                            'url': article_url,
                            'label': self.label,
                            'scraped_date': datetime.now().isoformat()
                        }

                        # Optionally fetch the full article body
                        if fetch_full_text:
                            full_text = self._fetch_full_text(article_url)
                            entry['full_text'] = full_text
                            if delay > 0:
                                time.sleep(delay)

                        articles.append(entry)

            print(f"Fetched {len(articles)} articles from {self.name}"
                  + (f" (full text: {sum(1 for a in articles if a.get('full_text'))})"
                     if fetch_full_text else ""))

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

    def collect_real_news(self, num_per_source: int = 10,
                          fetch_full_text: bool = False):
        print(f"\n[Collecting Real News] {len(self.REAL_NEWS_SOURCES)} sources")
        all_articles = []
        for source_name, url in self.REAL_NEWS_SOURCES.items():
            source = RealNewsSource(source_name, url)
            articles = source.fetch(num_per_source,
                                    fetch_full_text=fetch_full_text)
            all_articles.extend(articles)

        df = pd.DataFrame(all_articles)
        print(f"Collected {len(df)} real news articles")
        return df

    def collect_fake_news(self, num_per_source: int = 10,
                          fetch_full_text: bool = False):
        print(f"\n[Collecting Fake News] {len(self.FAKE_NEWS_SOURCES)} sources")
        all_articles = []
        for source_name, url in self.FAKE_NEWS_SOURCES.items():
            source = FakeNewsSource(source_name, url)
            articles = source.fetch(num_per_source,
                                    fetch_full_text=fetch_full_text)
            all_articles.extend(articles)

        df = pd.DataFrame(all_articles)
        print(f"Collected {len(df)} fake news articles")
        return df

    def collect_all(self, num_per_source: int = 10,
                    fetch_full_text: bool = False):
        real_df = self.collect_real_news(num_per_source,
                                         fetch_full_text=fetch_full_text)
        fake_df = self.collect_fake_news(num_per_source,
                                          fetch_full_text=fetch_full_text)

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

    def build_full_article_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build a DataFrame for full-article evaluation.

        Takes the combined DataFrame (which must contain a 'full_text' column)
        and returns a new DataFrame where 'text' is replaced by the full article
        body. Rows where the full article could not be scraped are dropped.
        """
        if 'full_text' not in df.columns:
            print("[Warning] No 'full_text' column — run collect with fetch_full_text=True")
            return pd.DataFrame()

        full_df = df[df['full_text'].astype(str).str.len() > 100].copy()
        # Keep original short text in a separate column for reference
        full_df = full_df.rename(columns={'text': 'short_text', 'full_text': 'text'})
        full_df = full_df.reset_index(drop=True)
        print(f"Full-article dataset: {len(full_df)} articles "
              f"(dropped {len(df) - len(full_df)} without full text)")
        return full_df

    def save_both_variants(self, df: pd.DataFrame,
                           short_filename: str = "web_scraped_news.csv",
                           full_filename: str = "web_scraped_news_full.csv"):
        """Save both the short-text and full-article CSVs from one combined DataFrame.

        Args:
            df: DataFrame with 'text' (description) and optionally 'full_text' columns.
            short_filename: filename for the short-text (description) CSV.
            full_filename: filename for the full-article CSV.
        """
        # Short tex
        short_cols = ['source', 'title', 'text', 'url', 'label', 'scraped_date']
        short_df = df[[c for c in short_cols if c in df.columns]].copy()
        self.save_scraped_data(short_df, filename=short_filename)

        # Full articles
        if 'full_text' in df.columns:
            full_df = self.build_full_article_df(df)
            if not full_df.empty:
                save_cols = ['source', 'title', 'text', 'short_text', 'url', 'label', 'scraped_date']
                full_df = full_df[[c for c in save_cols if c in full_df.columns]]
                self.save_scraped_data(full_df, filename=full_filename)
            else:
                print(f"[Warning] No full articles to save to {full_filename}")
        else:
            print(f"[Warning] 'full_text' column not present — skipping {full_filename}")


def create_simulated_web_data(num_real: int = 50, num_fake: int = 50,
                              include_full_text: bool = False):
    """Create simulated web scraped data for testing.

    Args:
        num_real: number of simulated real-news samples.
        num_fake: number of simulated fake-news samples.
        include_full_text: if True, generate a 'full_text' column with
                           longer simulated article bodies.
    """

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
        entry = {
            'source': 'Real News Source',
            'title': sample,
            'text': f"{sample}. " * np.random.randint(2, 5),
            'url': 'https://real-news-source.com',
            'label': 1,
            'scraped_date': datetime.now().isoformat()
        }
        if include_full_text:
            entry['full_text'] = f"{sample}. " * np.random.randint(15, 40)
        data.append(entry)

    # Fake news
    for i in range(num_fake):
        sample = np.random.choice(fake_news_samples)
        entry = {
            'source': 'Fake News Source',
            'title': sample,
            'text': f"{sample}. " * np.random.randint(2, 5),
            'url': 'https://fake-news-source.com',
            'label': 0,
            'scraped_date': datetime.now().isoformat()
        }
        if include_full_text:
            entry['full_text'] = f"{sample}. " * np.random.randint(15, 40)
        data.append(entry)

    df = pd.DataFrame(data)
    return df.sample(frac=1).reset_index(drop=True)
