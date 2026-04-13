"""
Configuration constants for web news scraping.

Contains news source URLs, CSS selectors for article extraction,
homepage selectors, and user-agent strings.
"""

USER_AGENTS: list[str] = [
    (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4_1) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) "
        "Version/17.4.1 Safari/605.1.15"
    ),
    (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
]

SITE_SELECTORS: dict[str, list[str]] = {
    "theonion.com": [
        "div[data-testid='post-content'] p",
        "div.post-content p",
    ],
    "babylonbee.com": [
        "div.entry-content p",
        "article .body p",
    ],
    "thedailymash.co.uk": [
        "div.entry-content p",
        "article .content p",
    ],
    "thebeaverton.com": [
        "div.entry-content p",
    ],
    "duffelblog.com": [
        "div.entry-content p",
    ],
    "chaser.com.au": [
        "div.entry-content p",
        "article .post-content p",
    ],
    "weeklyworldnews.com": [
        "div.entry-content p",
    ],
    "newsbiscuit.com": [
        "div.post-content p",
        "article .entry-content p",
        "article p",
    ],
    "gomerblog.com": [
        "div.entry-content p",
    ],
    "empirenews.net": [
        "div.entry-content p",
    ],
    "glossynews.com": [
        "div.entry-content p",
        "div.post-content p",
    ],
    "thepoke.com": [
        "div.entry-content p",
        "article p",
    ],
    "thespoof.com": [
        "div.entry-content p",
        "div#content p",
        "article p",
    ],
}

GENERIC_SELECTORS: list[str] = [
    "article .entry-content p",
    "article .post-content p",
    "div.entry-content p",
    "div.post-content p",
    "article p",
    "main p",
]

HOMEPAGE_SELECTORS: dict[str, list[str]] = {
    "theonion.com": [
        "div[data-testid] h2 a[href]",
        "article h2 a[href]",
        "h4 a[href]",
    ],
    "babylonbee.com": [
        ".entry-title a[href]",
        "article h2 a[href]",
        "h2 a[href]",
    ],
    "thedailymash.co.uk": [
        ".entry-title a[href]",
        "article h2 a[href]",
    ],
    "chaser.com.au": [
        ".entry-title a[href]",
        "article h2 a[href]",
    ],
    "duffelblog.com": [
        ".entry-title a[href]",
        "article h2 a[href]",
    ],
    "weeklyworldnews.com": [
        ".entry-title a[href]",
        "article h2 a[href]",
    ],
    "newsbiscuit.com": [
        ".entry-title a[href]",
        "article h2 a[href]",
        "h2 a[href]",
    ],
    "thepoke.com": [
        ".entry-title a[href]",
        "article h2 a[href]",
    ],
    "thespoof.com": [
        "article h2 a[href]",
        "h3 a[href]",
    ],
    "stevefakeballmer.wordpress.com": [
        ".entry-title a[href]",
        "article h2 a[href]",
    ],
}

GENERIC_HOMEPAGE_SELECTORS: list[str] = [
    "article h2 a[href]",
    "article h3 a[href]",
    ".entry-title a[href]",
    ".post-title a[href]",
    "h2.entry-title a[href]",
    "h2 a[href]",
    "h3 a[href]",
]

REAL_NEWS_SOURCES: dict[str, str] = {
    "BBC_World": "http://feeds.bbci.co.uk/news/world/rss.xml",
    "BBC_Tech": "http://feeds.bbci.co.uk/news/technology/rss.xml",
    "NYT_World": "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
    "NYT_US": "https://rss.nytimes.com/services/xml/rss/nyt/US.xml",
    "NPR": "https://feeds.npr.org/1001/rss.xml",
    "Guardian_World": "https://www.theguardian.com/world/rss",
    "Guardian_US": "https://www.theguardian.com/us-news/rss",
    "AlJazeera": "https://www.aljazeera.com/xml/rss/all.xml",
    "WashingtonPost": "https://feeds.washingtonpost.com/rss/world",
    "WallStreetJournal": "https://feeds.a.dj.com/rss/RSSWorldNews.xml",
    "ABC_News": "https://abcnews.go.com/abcnews/topstories",
    "CBS_News": "https://www.cbsnews.com/latest/rss/main",
    "SkyNews": "https://feeds.skynews.com/feeds/rss/world.xml",
    "NYTimes": "http://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",
    "FoxNews": "http://feeds.foxnews.com/foxnews/latest",
    "NBCNews": "http://feeds.nbcnews.com/feeds/topstories",
    "DailyMail": "http://www.dailymail.co.uk/articles.rss",
    "Independent": "http://www.independent.co.uk/rss",
    "NyPost": "https://nypost.com/feed/",
    "Express": "https://www.express.co.uk/posts/rss/77/news",
    "FinancialTimes": "https://www.ft.com/world?format=rss",
    "NationalPublicRadio": "https://feeds.npr.org/1004/rss.xml",
    "TheWashingtonPost": "https://feeds.washingtonpost.com/rss/world",
    "TheChristianScienceMonitor": "https://rss.csmonitor.com/feeds/world",
    "RFI": "https://rfi.fr/en/international/rss",
    "Time": "https://feeds.feedburner.com/time/world",
    "NBC_News": "https://feeds.nbcnews.com/nbcnews/public/news",
    "TheGlobeAndMail": "https://www.theglobeandmail.com/arc/outboundfeeds/rss/category/world/",
    "SBS_News": "https://www.sbs.com.au/news/topic/world/feed",
    "GlobalNews": "https://globalnews.ca/world/feed/",
    "ThePrint": "https://theprint.in/category/world/feed/",
    "CNBC": "https://www.cnbc.com/id/100727362/device/rss/rss.html",
    "DerSpiegel": "https://www.spiegel.de/international/index.rss",
    "TheSydneyMorningHerald": "https://www.smh.com.au/rss/world.xml",
    "LosAngelesTimes": "https://www.latimes.com/world-nation/rss2.0.xml#nt=1col-7030col1",
    "NDTV": "https://feeds.feedburner.com/ndtvnews-world-news",
    "TribuneIndia": "https://publish.tribuneindia.com/newscategory/world/feed/",
    "TheHinduBusinessline": "https://www.thehindubusinessline.com/news/world/feeder/default.rss",
    "BreakingNews": "https://feeds.breakingnews.ie/bnworld",
    "ScrippsNews": "https://www.scrippsnews.com/world.rss",
    "9News": "https://www.9news.com.au/world/rss",
    "MaroccoWorldNews": "https://www.moroccoworldnews.com/international/feed/",
    "DefenceBlog": "https://defence-blog.com/feed/",
    "FOWorld": "https://www.fairobserver.com/category/world-news/feed/",
    "TheWeek": "https://www.theweek.in/news/world.feeds.rss.xml",
    "CNBCTV": "https://www.cnbctv18.com/commonfeeds/v1/cne/rss/world.xml",
    "PatricaNews": "https://cms.patrika.com/googlefeed/blog/category/world-news",
    "DailyMirror": "https://www.mirror.co.uk/news/world-news/?service=rss",
    "DailyExpress": "https://feeds.feedburner.com/daily-express-world-news",
    "USNN": "https://www.usnn.news/feed/",
    "IFP": "https://ifpnews.com/feed/",
    "BuzzFeed": "https://www.buzzfeed.com/in/world.xml",
}

FAKE_NEWS_SOURCES: dict[str, str] = {
    "TheOnion": "https://www.theonion.com/rss",
    "BabylonBee": "https://babylonbee.com/feed",
    "DailyMash": "https://www.thedailymash.co.uk/feed/",
    "Beaverton": "https://www.thebeaverton.com/feed/",
    "Chaser": "https://chaser.com.au/feed/",
    "DuffelBlog": "https://duffelblog.com/feed",
    "GomerBlog": "https://gomerblog.com/feed",
    "EmpireNews": "https://empirenews.net/feed",
    "GlossyNews": "http://glossynews.com/feed/",
    "FakeBallmer": "https://stevefakeballmer.wordpress.com/feed/",
    "ThePoke": "https://www.thepoke.com/category/news/rss",
    "TheSpoof": "https://www.thespoof.com/rss/feeds/world/rss.xml",
}

