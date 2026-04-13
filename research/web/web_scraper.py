import random
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import cloudscraper
import feedparser
import pandas as pd
import requests
from bs4 import BeautifulSoup
from newspaper import Article, Config as NewspaperConfig

from research.web.sources_config import (
    FAKE_NEWS_SOURCES,
    GENERIC_HOMEPAGE_SELECTORS,
    GENERIC_SELECTORS,
    HOMEPAGE_SELECTORS,
    REAL_NEWS_SOURCES,
    SITE_SELECTORS,
    USER_AGENTS,
)


class NewsSource:
    label = None

    @staticmethod
    def _is_valid_rss(content: str) -> bool:
        head = content.strip()[:500].lower()
        return any(tag in head for tag in ("<?xml", "<rss", "<feed", "<rdf:rdf"))

    @staticmethod
    def _is_blocked_page(html: str) -> bool:
        lower = html[:2000].lower()
        return any(x in lower for x in [
            "cloudflare",
            "attention required",
            "captcha",
            "access denied",
            "bot detection",
        ])

    @staticmethod
    def _retry(fn, retries=3, delay=2):
        for i in range(retries):
            try:
                return fn()
            except Exception:
                if i == retries - 1:
                    raise
                time.sleep(delay * (i + 1))
        return None

    @classmethod
    def _build_headers(cls, url: str) -> dict:
        parsed = urlparse(url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        return {
            "User-Agent": random.choice(USER_AGENTS),
            "Referer": origin + "/",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }

    @classmethod
    def _fetch_html(cls, url: str, timeout: int = 15,
                    use_cloudscraper: bool = False) -> str:
        headers = cls._build_headers(url)

        def _do_request():
            if use_cloudscraper:
                scraper = cloudscraper.create_scraper(
                    browser={"browser": "chrome", "platform": "windows",
                             "mobile": False}
                )
                return scraper.get(url, headers=headers, timeout=timeout)
            else:
                session = requests.Session()
                parsed = urlparse(url)
                home = f"{parsed.scheme}://{parsed.netloc}"
                try:
                    session.get(home, headers=headers, timeout=timeout,
                                allow_redirects=True)
                except Exception:
                    pass
                return session.get(url, headers=headers, timeout=timeout,
                                   allow_redirects=True)

        resp = cls._retry(_do_request)
        resp.raise_for_status()
        html = resp.text

        if cls._is_blocked_page(html):
            raise RuntimeError(f"Blocked by anti-bot: {url}")

        return html


    @classmethod
    def _get_selectors_for_url(cls, url: str) -> list[str] | None:
        domain = urlparse(url).netloc.lower()
        for key, sels in SITE_SELECTORS.items():
            if key in domain:
                return sels
        return None

    @classmethod
    def _extract_with_selectors(cls, html: str,
                                selectors: list[str]) -> str:
        soup = BeautifulSoup(html, "html.parser")
        for sel in selectors:
            paragraphs = soup.select(sel)
            if paragraphs:
                text = "\n".join(p.get_text(strip=True) for p in paragraphs)
                text = text.strip()
                if len(text) > 100:
                    return text
        return ""

    @classmethod
    def _extract_text(cls, html: str, url: str = "") -> str:
        """Extract article text from HTML using a cascade of strategies."""
        # Site-specific selectors
        if url:
            site_sels = cls._get_selectors_for_url(url)
            if site_sels:
                text = cls._extract_with_selectors(html, site_sels)
                if text:
                    return text

        # Generic selectors
        text = cls._extract_with_selectors(html, GENERIC_SELECTORS)
        if text:
            return text

        # Raw <p> fallback
        soup = BeautifulSoup(html, "html.parser")
        paragraphs = soup.find_all("p")
        text = "\n".join(p.get_text(strip=True) for p in paragraphs)
        return text.strip() if len(text.strip()) > 100 else ""

    @classmethod
    def _parse_with_newspaper(cls, url: str) -> str:
        """Use newspaper3k as a last-resort extraction strategy."""
        try:
            config = NewspaperConfig()
            config.browser_user_agent = random.choice(USER_AGENTS)
            config.request_timeout = 15
            config.fetch_images = False
            article = Article(url, config=config)
            article.download()
            article.parse()
            text = (article.text or "").strip()
            return text if len(text) > 100 else ""
        except Exception:
            return ""

    @classmethod
    def _extract_feed_full_text(cls, entry, url: str) -> str:
        """Strategy 0: extract from RSS entry content (no HTTP needed)."""
        html_candidates: list[str] = []

        if hasattr(entry, "content") and entry.content:
            for block in entry.content:
                value = block.get("value", "")
                if value and len(value) > 200:
                    html_candidates.append(value)

        summary = getattr(entry, "summary", "") or ""
        if len(summary) > 500:
            html_candidates.append(summary)

        for html in html_candidates:
            text = cls._extract_text(html, url)
            if text:
                return text

        return ""

    @classmethod
    def _fetch_full_text(cls, url: str, feed_entry=None) -> str:
        """Multi-strategy full-text extraction pipeline.

        Order:
        0. RSS entry content (free, no HTTP)
        1. Site-specific selectors (session → cloudscraper)
        2. Generic selectors (session → cloudscraper)
        3. newspaper3k
        """
        # Strategy 0 — RSS entry content
        if feed_entry is not None:
            try:
                text = cls._extract_feed_full_text(feed_entry, url)
                if text:
                    return text
            except Exception:
                pass

        # Strategy 1+2 — fetch HTML, extract with selectors
        for use_cs in (False, True):
            try:
                html = cls._fetch_html(url, use_cloudscraper=use_cs)
                text = cls._extract_text(html, url)
                if text:
                    return text
            except Exception:
                continue

        # Strategy 3 — newspaper3k
        text = cls._parse_with_newspaper(url)
        if text:
            return text

        return ""

    @classmethod
    def _fetch_feed_xml(cls, feed_url: str):
        """Re-fetch RSS with session/cloudscraper when feedparser is blocked."""
        for use_cs in (False, True):
            try:
                raw = cls._fetch_html(feed_url, use_cloudscraper=use_cs)
                if not raw or not cls._is_valid_rss(raw):
                    continue
                feed = feedparser.parse(raw)
                if feed.entries:
                    return feed
            except Exception:
                continue
        return None

    @classmethod
    def _scrape_article_urls(cls, site_url: str,
                             num: int = 10) -> list[tuple[str, str]]:
        """Scrape (url, title) pairs from a site's homepage when RSS fails."""
        parsed = urlparse(site_url)
        base_domain = parsed.netloc.lower()
        homepage = f"{parsed.scheme}://{parsed.netloc}"

        selectors: list[str] = GENERIC_HOMEPAGE_SELECTORS
        for key, sels in HOMEPAGE_SELECTORS.items():
            if key in base_domain:
                selectors = sels
                break

        for use_cs in (False, True):
            try:
                html = cls._fetch_html(homepage, use_cloudscraper=use_cs)
                if not html:
                    continue

                soup = BeautifulSoup(html, "html.parser")
                seen: set[str] = set()
                results: list[tuple[str, str]] = []

                for sel in selectors:
                    for a_tag in soup.select(sel):
                        href = a_tag.get("href", "")
                        title = a_tag.get_text(strip=True)
                        if not href or not title or len(title) < 10:
                            continue
                        if href.startswith("/"):
                            href = homepage + href
                        if href in seen:
                            continue
                        if base_domain not in urlparse(href).netloc.lower():
                            continue
                        seen.add(href)
                        results.append((href, title))
                        if len(results) >= num:
                            return results
                    if results:
                        break

                if results:
                    return results
            except Exception:
                continue
        return []

    def __init__(self, name: str, url: str):
        self.name = name
        self.url = url

    def fetch(self, num_articles: int = 10,
              fetch_full_text: bool = False,
              delay: float = 0.5) -> list[dict]:
        """Fetch articles with 3-tier fallback:
        1. feedparser with its own HTTP
        2. Re-fetch feed XML with session/cloudscraper → re-parse
        3. Scrape article links from the site's homepage
        """
        articles: list[dict] = []

        try:
            feed = feedparser.parse(self.url)

            if feed.bozo and not feed.entries:
                print(f"[{self.name}] RSS blocked — retrying with "
                      f"session/cloudscraper…")
                feed = self._fetch_feed_xml(self.url)

            if feed and feed.entries:
                for entry in feed.entries[:num_articles]:
                    title = entry.get("title")
                    link = entry.get("link")
                    if not title or not link:
                        continue

                    summary = BeautifulSoup(
                        entry.get("summary", ""), "html.parser"
                    ).get_text(strip=True)

                    article = {
                        "source": self.name,
                        "title": title.strip(),
                        "text": summary if summary else title.strip(),
                        "url": link.strip(),
                        "label": self.label,
                        "scraped_date": datetime.now().isoformat(),
                    }

                    if fetch_full_text:
                        full = self._fetch_full_text(link.strip(),
                                                     feed_entry=entry)
                        if full:
                            article["full_text"] = full
                        if delay > 0:
                            time.sleep(delay)

                    articles.append(article)

            else:
                print(f"[{self.name}] RSS unavailable — scraping homepage…")
                urls = self._scrape_article_urls(self.url, num=num_articles)
                for url, title in urls:
                    body = self._fetch_full_text(url)
                    summary = (body[:300] + "…") if len(body) > 300 else body

                    article = {
                        "source": self.name,
                        "title": title,
                        "text": summary if summary else title,
                        "url": url,
                        "label": self.label,
                        "scraped_date": datetime.now().isoformat(),
                    }
                    if fetch_full_text and body:
                        article["full_text"] = body
                    articles.append(article)
                    if delay > 0:
                        time.sleep(delay)

            full_count = sum(1 for a in articles if a.get("full_text")) \
                if fetch_full_text else 0
            print(
                f"Fetched {len(articles)} from {self.name}"
                + (f" (full text: {full_count})" if fetch_full_text else "")
            )

        except Exception as e:
            print(f"Error fetching {self.name}: {e}")

        return articles


class RealNewsSource(NewsSource):
    label = 1


class FakeNewsSource(NewsSource):
    label = 0


class WebNewsCollector:
    """Collect news from multiple sources."""

    def __init__(self, output_dir: str = "configs/web_scraped_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def collect(self, sources: dict, source_class, num: int = 10,
                full_text: bool = False) -> pd.DataFrame:
        all_articles: list[dict] = []
        for name, url in sources.items():
            src = source_class(name, url)
            articles = src.fetch(num, fetch_full_text=full_text)
            all_articles.extend(articles)
        return pd.DataFrame(all_articles)

    def collect_real_news(self, num_per_source: int = 10,
                          fetch_full_text: bool = False) -> pd.DataFrame:
        print(f"\n[Collecting Real News] {len(REAL_NEWS_SOURCES)} sources")
        df = self.collect(REAL_NEWS_SOURCES, RealNewsSource,
                          num_per_source, fetch_full_text)
        print(f"Collected {len(df)} real news articles")
        return df

    def collect_fake_news(self, num_per_source: int = 10,
                          fetch_full_text: bool = False) -> pd.DataFrame:
        print(f"\n[Collecting Fake News] {len(FAKE_NEWS_SOURCES)} sources")
        df = self.collect(FAKE_NEWS_SOURCES, FakeNewsSource,
                          num_per_source, fetch_full_text)
        print(f"Collected {len(df)} fake news articles")
        return df

    def collect_all(self, num_per_source: int = 10,
                    fetch_full_text: bool = False) -> pd.DataFrame:
        real_df = self.collect_real_news(num_per_source, fetch_full_text)
        fake_df = self.collect_fake_news(num_per_source, fetch_full_text)
        combined = pd.concat([real_df, fake_df], ignore_index=True)
        if not combined.empty:
            combined = combined.sample(frac=1).reset_index(drop=True)
        return combined

    def save_scraped_data(self, df: pd.DataFrame,
                          filename: str = "web_scraped_news.csv") -> str | None:
        if df.empty:
            print("No data to save!")
            return None
        filepath = self.output_dir / filename
        df.to_csv(filepath, index=False)
        print(f"Saved to {filepath}")
        return str(filepath)

    def build_full_article_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replace 'text' with 'full_text'; drop rows without full text."""
        if "full_text" not in df.columns:
            print("[Warning] No 'full_text' column — "
                  "run collect with fetch_full_text=True")
            return pd.DataFrame()

        full_df = df[df["full_text"].astype(str).str.len() > 100].copy()
        full_df = full_df.rename(
            columns={"text": "short_text", "full_text": "text"}
        )
        full_df = full_df.reset_index(drop=True)
        print(f"Full-article dataset: {len(full_df)} articles "
              f"(dropped {len(df) - len(full_df)} without full text)")
        return full_df

    def save_both_variants(self, df: pd.DataFrame,
                           short_filename: str = "web_scraped_news.csv",
                           full_filename: str = "web_scraped_news_full.csv"):
        """Save short-text and full-article CSVs."""
        short_cols = ["source", "title", "text", "url", "label", "scraped_date"]
        short_df = df[[c for c in short_cols if c in df.columns]].copy()
        self.save_scraped_data(short_df, filename=short_filename)

        if "full_text" in df.columns:
            full_df = self.build_full_article_df(df)
            if not full_df.empty:
                save_cols = ["source", "title", "text", "short_text",
                             "url", "label", "scraped_date"]
                full_df = full_df[[c for c in save_cols if c in full_df.columns]]
                self.save_scraped_data(full_df, filename=full_filename)
            else:
                print(f"[Warning] No full articles to save to {full_filename}")
        else:
            print(f"[Warning] 'full_text' column not present — "
                  f"skipping {full_filename}")
