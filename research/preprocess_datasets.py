import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

BASE = Path(__file__).parent
RANDOM_STATE = 42

REUTERS_PREFIX = re.compile(r"[A-Z][A-Za-z\s/]+ \(Reuters\)\s*-\s*")
REUTERS_TAG = re.compile(r"\(Reuters\)")
VIDEO_TAG = re.compile(r"[\[\(]video[\]\)]", re.IGNORECASE)


def save_splits(df: pd.DataFrame, out_dir: Path, train_ratio=0.8, test_ratio=0.1):
    """Shuffle and split df into train/test/val CSVs (80/10/10)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    df = df.dropna(subset=["text"]).reset_index(drop=True)
    train, temp = train_test_split(df, test_size=1 - train_ratio, random_state=RANDOM_STATE, stratify=df["label"])
    relative_test = test_ratio / (1 - train_ratio)
    test, val = train_test_split(temp, test_size=1 - relative_test, random_state=RANDOM_STATE, stratify=temp["label"])
    for name, split in [("train", train), ("test", test), ("val", val)]:
        path = out_dir / f"{name}.csv"
        split.to_csv(path, index=False)
        print(f"  {path.name}: {len(split)} rows")


def process_isot():
    print("\nISOT")
    true_df = pd.read_csv(BASE / "ISOT" / "True.csv")
    fake_df = pd.read_csv(BASE / "ISOT" / "Fake.csv")
    true_df["label"] = 0
    fake_df["label"] = 1
    df = pd.concat([true_df, fake_df], ignore_index=True)[["text", "label"]]

    texts = df["text"].astype(str)
    n_reuters = int(texts.str.contains(r"\(Reuters\)", na=False).sum())
    n_video = int(texts.str.contains(r"[\[\(]video[\]\)]", case=False, na=False).sum())
    df["text"] = df["text"].apply(clean_reuters)
    print(f"  cleaned {n_reuters} Reuters + {n_video} [video] tags before splitting")

    save_splits(df, BASE / "ISOT")


def process_liar():
    print("\nLIAR")
    # LIAR TSV: col 1 = label, col 2 = statement (0-indexed)
    FAKE_LABELS = {"pants-fire", "false", "barely-true"}
    REAL_LABELS = {"half-true", "mostly-true", "true"}

    out_dir = BASE / "LIAR"
    out_dir.mkdir(parents=True, exist_ok=True)

    for src_name, dst_name in [("train.tsv", "train.csv"), ("test.tsv", "test.csv"), ("valid.tsv", "val.csv")]:
        df = pd.read_csv(BASE / "LIAR" / src_name, sep="\t", header=None)
        df = df.rename(columns={1: "orig_label", 2: "text"})
        df = df[df["orig_label"].isin(FAKE_LABELS | REAL_LABELS)].copy()
        df["label"] = df["orig_label"].apply(lambda x: 1 if x in FAKE_LABELS else 0)
        df = df[["text", "label"]].dropna(subset=["text"])
        dst_path = out_dir / dst_name
        df.to_csv(dst_path, index=False)
        print(f"  {dst_name}: {len(df)} rows")


def process_welfake():
    print("\nWELFake")
    df = pd.read_csv(BASE / "WELFake" / "data.csv")
    # Keep only text and label columns
    df = df[["text", "label"]]
    save_splits(df, BASE / "WELFake")


def clean_reuters(text: str) -> str:
    """Remove Reuters agency tags and [video] tags from ISOT text."""
    if not isinstance(text, str):
        return text
    text = REUTERS_PREFIX.sub("", text)
    text = REUTERS_TAG.sub("", text)
    text = VIDEO_TAG.sub("", text)
    return text.strip()


if __name__ == "__main__":
    process_isot()
    process_liar()
    process_welfake()
    print("\nAll done!")

