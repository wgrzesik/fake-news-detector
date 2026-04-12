import argparse

import pandas as pd
from omegaconf import OmegaConf

from research.web.web_scraper import WebNewsCollector, create_simulated_web_data


def main():
    parser = argparse.ArgumentParser(description='Collect web-scraped news data based on config.')
    parser.add_argument(
        '--num-samples-per-class',
        type=int,
        default=50,
        help='Fallback samples per class if config fails'
    )
    args = parser.parse_args()

    # Load configuration
    try:
        cfg = OmegaConf.load("research/configs/config.yaml")
        use_simulated = cfg.web_scraping.use_simulated_data
        num_real = cfg.web_scraping.num_real_samples
        num_fake = cfg.web_scraping.num_fake_samples
        output_dir = cfg.web_scraping.output_dir
        scrape_full_articles = cfg.web_scraping.get('scrape_full_articles', True)
        full_article_filename = cfg.web_scraping.get(
            'full_article_filename', 'web_scraped_news_full.csv'
        )
    except Exception as e:
        print(f"Could not load config: {e}. Defaulting to simulated data.")
        use_simulated = True
        num_real = args.num_samples_per_class
        num_fake = args.num_samples_per_class
        output_dir = "research/configs/web_scraped_data"
        scrape_full_articles = True
        full_article_filename = "web_scraped_news_full.csv"

    # Generate or Collect Data based on config
    if use_simulated:
        print(f"\n[Creating {num_real + num_fake} simulated web-scraped test samples]")
        web_data = create_simulated_web_data(
            num_real=num_real,
            num_fake=num_fake,
            include_full_text=scrape_full_articles
        )

        collector = WebNewsCollector(output_dir=output_dir)

        if scrape_full_articles:
            collector.save_both_variants(
                web_data,
                short_filename="web_scraped_news.csv",
                full_filename=full_article_filename
            )
        else:
            collector.save_scraped_data(web_data)

    else:
        print("\n[Running Web Scraper for REAL data]")
        collector = WebNewsCollector(output_dir=output_dir)

        # Calculate per-source limits to match your desired total
        num_per_real = max(1, num_real // len(collector.REAL_NEWS_SOURCES))
        num_per_fake = max(1, num_fake // len(collector.FAKE_NEWS_SOURCES))

        real_df = collector.collect_real_news(
            num_per_source=num_per_real,
            fetch_full_text=scrape_full_articles
        )
        fake_df = collector.collect_fake_news(
            num_per_source=num_per_fake,
            fetch_full_text=scrape_full_articles
        )

        # Combine and shuffle
        web_data = pd.concat([real_df, fake_df], ignore_index=True).sample(frac=1).reset_index(drop=True)

        if scrape_full_articles:
            collector.save_both_variants(
                web_data,
                short_filename="web_scraped_news.csv",
                full_filename=full_article_filename
            )
        else:
            collector.save_scraped_data(web_data)

    print(f"Prepared {len(web_data)} web samples\n")


if __name__ == "__main__":
    main()
