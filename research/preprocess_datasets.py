import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

BASE = Path(__file__).parent
RANDOM_STATE = 42


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


def run_clean_isot():
    print("\nCleaning ISOT (removing Reuters leakage)")
    import importlib.util, sys
    # Update ISOT_DIR in clean_isot module before running
    spec = importlib.util.spec_from_file_location("clean_isot", BASE / "clean_isot.py")
    mod = importlib.util.module_from_spec(spec)
    # Override the directory to point to local ISOT folder
    mod.ISOT_DIR = BASE / "ISOT"
    spec.loader.exec_module(mod)
    # Re-override after exec (module-level assignment may have reset it)
    mod.ISOT_DIR = BASE / "ISOT"
    for split in ["train", "test", "val"]:
        path = mod.ISOT_DIR / f"{split}.csv"
        if path.exists():
            stats = mod.clean_file(path)
            print(f"  {stats['file']}: cleaned {stats['reuters']} Reuters + {stats['video']} [video] tags (out of {stats['total']} rows)")


if __name__ == "__main__":
    process_isot()
    process_liar()
    process_welfake()
    run_clean_isot()
    print("\nAll done!")

