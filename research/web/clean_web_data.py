"""
Clean web-scraped news data and split into per-text-type CSVs.

Removes:
- Non-English articles (using langdetect)
- Unicode artifacts (ZWNBSP, zero-width chars, replacement chars, etc.)
- Duplicate articles (by URL and by text content)
- Articles with insufficient text length

Produces three CSVs in the output directory (default: web_scraped_data/processed/):
- web_title.csv       (article titles)
- web_text.csv        (full article text)
- web_short_text.csv  (RSS summary / short descriptions)

Usage:
    python -m research.web.clean_web_data [--input PATH] [--output-dir DIR] [--min-length 100]
"""

import argparse
import html
import re
import unicodedata
from pathlib import Path

import pandas as pd
from langdetect import detect, LangDetectException
from langdetect import DetectorFactory
from tqdm import tqdm

# Make langdetect deterministic
DetectorFactory.seed = 0

# Unicode characters to remove
UNICODE_ARTIFACTS = re.compile(
    r'[\ufeff'        # BOM / ZWNBSP (Zero Width No-Break Space)
    r'\u200b'         # Zero Width Space
    r'\u200c'         # Zero Width Non-Joiner
    r'\u200d'         # Zero Width Joiner
    r'\u200e'         # Left-to-Right Mark
    r'\u200f'         # Right-to-Left Mark
    r'\u00a0'         # Non-Breaking Space (replace with regular space)
    r'\u00ad'         # Soft Hyphen
    r'\u2060'         # Word Joiner
    r'\ufffd'         # Replacement Character
    r'\u2028'         # Line Separator
    r'\u2029'         # Paragraph Separator
    r'\x00-\x08'      # Control characters
    r'\x0b\x0c'       # Vertical tab, form feed
    r'\x0e-\x1f'      # More control characters
    r']+'
)

# Minimum text length for reliable language detection
MIN_LANGDETECT_LENGTH = 100


def remove_unicode_artifacts(text: str) -> str:
    """
    Remove invisible Unicode characters and normalize text.

    Handles:
    - ZWNBSP (Zero Width No-Break Space) - \ufeff
    - Zero-width characters (\u200b, \u200c, \u200d)
    - Replacement character (\ufffd)
    - Non-breaking spaces (\u00a0)
    - Various control characters
    - HTML entities
    """
    if not isinstance(text, str):
        return ""

    # Decode HTML entities (e.g., &amp; -> &, &nbsp; -> space)
    text = html.unescape(text)

    # Normalize Unicode (NFKC converts compatibility characters)
    text = unicodedata.normalize('NFKC', text)

    # Remove unicode artifacts
    text = UNICODE_ARTIFACTS.sub(' ', text)

    # Replace multiple spaces/newlines with single space
    text = re.sub(r'\s+', ' ', text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def is_english(text: str, min_length: int = MIN_LANGDETECT_LENGTH) -> bool:
    """
    Check if text is written in English.

    Args:
        text: The text to check.
        min_length: Minimum character length for reliable detection.
                   Shorter texts default to True (kept).

    Returns:
        True if text is English or too short to detect reliably.
        False if text is detected as non-English.
    """
    if not isinstance(text, str) or len(text.strip()) < min_length:
        # Too short to detect reliably - keep it
        return True

    try:
        lang = detect(text)
        return lang == 'en'
    except LangDetectException:
        # Detection failed (e.g., no features) - keep it
        return True


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate articles by URL and by text content.

    Returns:
        DataFrame with duplicates removed.
    """
    before = len(df)

    # Remove URL duplicates first
    if 'url' in df.columns:
        df = df.drop_duplicates(subset=['url'], keep='first')

    # Remove text content duplicates
    if 'text' in df.columns:
        df = df.drop_duplicates(subset=['text'], keep='first')

    after = len(df)
    removed = before - after

    if removed > 0:
        print(f"  Removed {removed} duplicate articles")

    return df.reset_index(drop=True)


def filter_by_length(df: pd.DataFrame, min_length: int = MIN_LANGDETECT_LENGTH) -> pd.DataFrame:
    """
    Remove articles with 'text' column shorter than min_length characters.

    Note: Only filters on 'text' column. 'short_text' and 'title' columns
    are not affected by this length constraint.
    """
    before = len(df)

    df = df[df['text'].astype(str).str.len() >= min_length]

    after = len(df)
    removed = before - after

    if removed > 0:
        print(f"  Removed {removed} articles with text < {min_length} characters")

    return df.reset_index(drop=True)


def clean_web_data(
    df: pd.DataFrame,
    min_text_length: int = 100,
    filter_non_english: bool = True,
    remove_dups: bool = True,
    show_progress: bool = True
) -> pd.DataFrame:
    """
    Full cleaning pipeline for web-scraped news data.

    Steps:
    1. Drop rows with missing text or label
    2. Clean Unicode artifacts from text, title, and short_text
    3. Remove duplicates (URL + content)
    4. Filter articles where 'text' column < min_text_length
       (Note: 'title' and 'short_text' have no length constraints)
    5. Filter non-English articles

    Args:
        df: Input DataFrame with 'text' and 'label' columns.
        min_text_length: Minimum 'text' column length to keep (default: 100).
                        Does NOT apply to 'title' or 'short_text' columns.
        filter_non_english: Whether to remove non-English articles.
        remove_dups: Whether to remove duplicate articles.
        show_progress: Whether to show progress bars.

    Returns:
        Cleaned DataFrame.
    """
    print(f"\n{'='*60}")
    print("CLEANING WEB-SCRAPED DATA")
    print(f"{'='*60}")
    print(f"Initial samples: {len(df)}")

    # Step 1: Drop missing values
    before = len(df)
    df = df.dropna(subset=['text', 'label'])
    removed = before - len(df)
    if removed > 0:
        print(f"  Removed {removed} rows with missing text/label")

    # Step 2: Clean Unicode artifacts from text
    print("\n[Step 1/4] Cleaning Unicode artifacts...")
    tqdm.pandas(desc="Cleaning text", disable=not show_progress)
    df['text'] = df['text'].progress_apply(remove_unicode_artifacts)

    # Also clean title if present
    if 'title' in df.columns:
        df['title'] = df['title'].apply(remove_unicode_artifacts)

    # Also clean short_text if present
    if 'short_text' in df.columns:
        df['short_text'] = df['short_text'].apply(remove_unicode_artifacts)

    # Step 3: Remove duplicates
    if remove_dups:
        print("\n[Step 2/4] Removing duplicates...")
        df = remove_duplicates(df)

    # Step 4: Filter by length
    print(f"\n[Step 3/4] Filtering by minimum length ({min_text_length} chars)...")
    df = filter_by_length(df, min_length=min_text_length)

    # Step 5: Filter non-English articles
    if filter_non_english:
        print("\n[Step 4/4] Filtering non-English articles...")
        before = len(df)

        tqdm.pandas(desc="Detecting language", disable=not show_progress)
        is_eng_mask = df['text'].progress_apply(is_english)

        df = df[is_eng_mask]
        removed = before - len(df)

        if removed > 0:
            print(f"  Removed {removed} non-English articles")

    df = df.reset_index(drop=True)

    print(f"\n{'='*60}")
    print(f"CLEANING COMPLETE")
    print(f"Final samples: {len(df)}")
    print(f"{'='*60}\n")

    return df


def print_stats(df: pd.DataFrame, title: str = "Dataset Statistics"):
    """Print statistics about the dataset."""
    print(f"\n{title}")
    print("-" * 40)
    print(f"Total samples: {len(df)}")

    if 'label' in df.columns:
        label_counts = df['label'].value_counts()
        print(f"Real news (1): {label_counts.get(1, 0)}")
        print(f"Fake news (0): {label_counts.get(0, 0)}")

    if 'text' in df.columns:
        text_lengths = df['text'].astype(str).str.len()
        print(f"Avg text length: {text_lengths.mean():.0f} chars")
        print(f"Min text length: {text_lengths.min()} chars")
        print(f"Max text length: {text_lengths.max()} chars")

    if 'source' in df.columns:
        print(f"Unique sources: {df['source'].nunique()}")


def split_by_text_type(
    df: pd.DataFrame,
    output_dir: Path,
    min_title_length: int = 10,
    min_short_text_length: int = 30,
) -> dict[str, Path]:
    """Split a cleaned DataFrame into three per-text-type CSVs.

    Produces:
      - web_title.csv       ('title' column becomes 'text')
      - web_text.csv         ('text' column stays as 'text')  — full article
      - web_short_text.csv   ('short_text' column becomes 'text')

    Each file contains only 'text' and 'label' columns.
    Rows whose text is too short or missing are dropped.

    Returns a dict  {text_type: output_path}.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Mapping: text_type -> (source_column, minimum_length)
    variants: dict[str, tuple[str, int]] = {
        "title": ("title", min_title_length),
        "text": ("text", 100),
        "short_text": ("short_text", min_short_text_length),
    }

    saved: dict[str, Path] = {}

    for text_type, (src_col, min_len) in variants.items():
        if src_col not in df.columns:
            print(f"  [Skip] Column '{src_col}' not found — skipping {text_type}")
            continue

        sub = df[[src_col, "label"]].copy()
        sub = sub.rename(columns={src_col: "text"})
        sub = sub.dropna(subset=["text"])
        sub = sub[sub["text"].astype(str).str.len() >= min_len]
        sub = sub.reset_index(drop=True)

        out_path = output_dir / f"web_{text_type}.csv"
        sub.to_csv(out_path, index=False)
        saved[text_type] = out_path
        print(f"  {text_type:12s} -> {out_path}  ({len(sub)} samples)")

    return saved


def main():
    parser = argparse.ArgumentParser(
        description="Clean web-scraped news data."
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="research/configs/web_scraped_data/web_scraped_news.csv",
        help="Path to input CSV file"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="research/configs/web_scraped_data/processed",
        help="Directory for per-text-type output CSVs"
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=100,
        help="Minimum full-text length to keep (default: 100)"
    )
    parser.add_argument(
        "--skip-language-filter",
        action="store_true",
        help="Skip filtering non-English articles"
    )
    parser.add_argument(
        "--skip-dedup",
        action="store_true",
        help="Skip removing duplicates"
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars"
    )

    args = parser.parse_args()

    # Resolve input path
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}")
        return 1

    output_dir = Path(args.output_dir)

    # Load data
    print(f"\nLoading data from: {input_path}")
    df = pd.read_csv(input_path)
    print_stats(df, "Input Dataset Statistics")

    # Clean data (operates on 'text' column — full article body)
    df_clean = clean_web_data(
        df,
        min_text_length=args.min_length,
        filter_non_english=not args.skip_language_filter,
        remove_dups=not args.skip_dedup,
        show_progress=not args.no_progress
    )

    print_stats(df_clean, "Cleaned Dataset Statistics")

    # Split into per-text-type CSVs and save to processed/
    print(f"\nSplitting into per-text-type CSVs → {output_dir}")
    saved = split_by_text_type(df_clean, output_dir)

    if not saved:
        print("[ERROR] No output files produced.")
        return 1

    print(f"\nDone — {len(saved)} files written to {output_dir}")
    return 0


if __name__ == "__main__":
    exit(main())



