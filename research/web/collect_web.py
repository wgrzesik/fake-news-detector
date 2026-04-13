import argparse

import pandas as pd
from omegaconf import OmegaConf

from research.web.sources_config import FAKE_NEWS_SOURCES, REAL_NEWS_SOURCES
from research.web.web_scraper import WebNewsCollector


def safe_concat(df_list):
    df_list = [df for df in df_list if df is not None and not df.empty]
    if not df_list:
        return pd.DataFrame()
    return pd.concat(df_list, ignore_index=True)


def remove_duplicates(df):
    if df.empty:
        return df
    return df.drop_duplicates(subset=["url"]).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(
        description="Collect web-scraped news data based on config."
    )
    parser.add_argument("--num-samples-per-class", type=int, default=50)
    args = parser.parse_args()

    try:
        cfg = OmegaConf.load("research/configs/config.yaml")

        num_real = cfg.web_scraping.num_real_samples
        num_fake = cfg.web_scraping.num_fake_samples
        output_dir = cfg.web_scraping.output_dir
        scrape_full_articles = cfg.web_scraping.get("scrape_full_articles", True)
        full_article_filename = cfg.web_scraping.get(
            "full_article_filename", "web_scraped_news_full.csv"
        )

    except Exception as e:
        print(f"[WARNING] Config load failed: {e}")
        print("Falling back to defaults")

        num_real = args.num_samples_per_class
        num_fake = args.num_samples_per_class
        output_dir = "research/configs/web_scraped_data"
        scrape_full_articles = True
        full_article_filename = "web_scraped_news_full.csv"

    print("\n[SCRAPING MODE]")

    collector = WebNewsCollector(output_dir=output_dir)

    num_per_real = max(1, num_real // max(1, len(REAL_NEWS_SOURCES)))
    num_per_fake = max(1, num_fake // max(1, len(FAKE_NEWS_SOURCES)))

    print(f"Real per source: {num_per_real}")
    print(f"Fake per source: {num_per_fake}")

    real_df = collector.collect_real_news(
        num_per_source=num_per_real,
        fetch_full_text=scrape_full_articles,
    )

    fake_df = collector.collect_fake_news(
        num_per_source=num_per_fake,
        fetch_full_text=scrape_full_articles,
    )

    web_data = safe_concat([real_df, fake_df])

    # Remove duplicates
    before = len(web_data)
    web_data = remove_duplicates(web_data)
    after = len(web_data)
    print(f"Removed duplicates: {before - after}")

    # Shuffle
    if not web_data.empty:
        web_data = web_data.sample(frac=1).reset_index(drop=True)

    if web_data.empty:
        print("[ERROR] No data collected. Exiting.")
        return

    if scrape_full_articles and "full_text" in web_data.columns:
        success = web_data["full_text"].astype(str).str.len() > 100
        print(f"Full text success rate: {success.mean():.2%}")

    print(f"\n[FINAL DATASET] {len(web_data)} samples")

    if scrape_full_articles:
        collector.save_both_variants(
            web_data,
            short_filename="web_scraped_news.csv",
            full_filename=full_article_filename,
        )
    else:
        collector.save_scraped_data(web_data)

    print("\nDONE")


if __name__ == "__main__":
    main()
