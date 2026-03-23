import argparse

import pandas as pd
from omegaconf import OmegaConf

from research.web_scrapper import WebNewsCollector, create_simulated_web_data


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
        # Note: adjust the path if your config is located elsewhere
        cfg = OmegaConf.load("research/configs/config.yaml")
        use_simulated = cfg.web_scraping.use_simulated_data
        num_real = cfg.web_scraping.num_real_samples
        num_fake = cfg.web_scraping.num_fake_samples
        output_dir = cfg.web_scraping.output_dir
    except Exception as e:
        print(f"Could not load config: {e}. Defaulting to simulated data.")
        use_simulated = True
        num_real = args.num_samples_per_class
        num_fake = args.num_samples_per_class
        output_dir = "research/configs/web_scraped_data"

    # Generate or Collect Data based on config
    if use_simulated:
        print(f"\n[Creating {num_real + num_fake} simulated web-scraped test samples]")
        web_data = create_simulated_web_data(num_real=num_real, num_fake=num_fake)

        # Initialize collector just to use its saving method
        collector = WebNewsCollector(output_dir=output_dir)
        collector.save_scraped_data(web_data)

    else:
        print("\n[Running Web Scraper for REAL data]")
        collector = WebNewsCollector(output_dir=output_dir)

        # Calculate per-source limits to match your desired total
        num_per_real = max(1, num_real // len(collector.REAL_NEWS_SOURCES))
        num_per_fake = max(1, num_fake // len(collector.FAKE_NEWS_SOURCES))

        real_df = collector.collect_real_news(num_per_source=num_per_real)
        fake_df = collector.collect_fake_news(num_per_source=num_per_fake)

        # Combine, shuffle, and save
        web_data = pd.concat([real_df, fake_df], ignore_index=True).sample(frac=1).reset_index(drop=True)
        collector.save_scraped_data(web_data)

    print(f"Prepared {len(web_data)} web samples\n")


if __name__ == "__main__":
    main()
