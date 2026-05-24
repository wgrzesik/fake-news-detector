import argparse
import glob
import json
import os
import shutil
from pathlib import Path

import pandas as pd


OUTPUT_DIRS = [
    "saved_models",
    "experiments",
    "outputs",
    "research/configs/web_scraped_data",
]

OUTPUT_FILES = [
    "mlflow.db",
    "optuna.db",
]

# Directories that contain INPUT data (uploaded once to Drive)
# key  = path inside the project
# value = path inside the Drive results folder
INPUT_DIRS = {
    "research/configs/datasets/processed":   "datasets/processed",
    "research/configs/datasets/embeddings":  "datasets/embeddings",
    "research/configs/web_scraped_data": "datasets/web_scraped_data",
}


def _force_symlink(target: str, link: str) -> None:
    """Create a symlink, removing whatever is in the way."""
    if os.path.islink(link):
        os.remove(link)
    elif os.path.isdir(link):
        shutil.rmtree(link)
    elif os.path.isfile(link):
        os.remove(link)
    # Make sure parent dir exists
    os.makedirs(os.path.dirname(link), exist_ok=True)
    os.symlink(target, link)


def _discover_web_scraped_data(drive: Path) -> None:
    """Copy web_scraped_data CSVs from the INPUT location to the OUTPUT location.

    Users upload CSVs to ``<drive>/datasets/web_scraped_data/`` (INPUT_DIRS).
    ``evaluate_web.py`` reads from ``<drive>/research/configs/web_scraped_data/``
    (OUTPUT_DIRS).  This step bridges the two by copying any CSVs across,
    including the ``processed/`` subdirectory.
    """
    source_dir = drive / "datasets" / "web_scraped_data"
    target_dir = drive / "research" / "configs" / "web_scraped_data"

    if not source_dir.is_dir() or not any(source_dir.glob("*.csv")):
        print(f"No web-scraped CSV files found at: {source_dir}")
        print(f"Upload web_scraped_news.csv to:")
        print(f"{source_dir}")
        return

    target_dir.mkdir(parents=True, exist_ok=True)

    # Copy root-level CSVs
    csvs = sorted(source_dir.glob("*.csv"))
    print(f"Copying web data from {source_dir} → {target_dir}")
    for csv_path in csvs:
        dest = target_dir / csv_path.name
        shutil.copy2(str(csv_path), str(dest))
        try:
            row_count = len(pd.read_csv(dest))
            print(f"  {csv_path.name}  ({row_count} rows)")
        except Exception:
            print(f"  {csv_path.name}  (copied, could not preview)")

    # Copy processed/ subdirectory CSVs
    processed_src = source_dir / "processed"
    processed_dst = target_dir / "processed"
    if processed_src.is_dir() and any(processed_src.glob("*.csv")):
        processed_dst.mkdir(parents=True, exist_ok=True)
        proc_csvs = sorted(processed_src.glob("*.csv"))
        print(f"Copying processed web data from {processed_src} → {processed_dst}")
        for csv_path in proc_csvs:
            dest = processed_dst / csv_path.name
            shutil.copy2(str(csv_path), str(dest))
            try:
                row_count = len(pd.read_csv(dest))
                print(f"{csv_path.name}  ({row_count} rows)")
            except Exception:
                print(f"{csv_path.name}  (copied, could not preview)")


def setup(project_root: str, drive_root: str) -> None:
    project = Path(project_root).resolve()
    drive   = Path(drive_root).resolve()

    print(f"\n{'='*60}")
    print(f"Colab Setup")
    print(f"Project: {project}")
    print(f"Drive: {drive}")
    print(f"{'='*60}\n")

    print("[1/5] Linking OUTPUT directories (project -> Drive) …")
    for name in OUTPUT_DIRS:
        drive_path   = str(drive / name)
        project_path = str(project / name)
        os.makedirs(drive_path, exist_ok=True)
        _force_symlink(drive_path, project_path)
        print(f"{name} -> {drive_path}")

    print("\n[2/5] Linking OUTPUT files (project -> Drive) …")
    for name in OUTPUT_FILES:
        drive_path   = str(drive / name)
        project_path = str(project / name)
        if not os.path.exists(drive_path):
            open(drive_path, "w").close()
        _force_symlink(drive_path, project_path)
        print(f"{name} -> {drive_path}")

    print("\n[3/5] Linking INPUT data (Drive -> project) …")
    for project_rel, drive_rel in INPUT_DIRS.items():
        drive_path   = str(drive / drive_rel)
        project_path = str(project / project_rel)

        if not os.path.exists(drive_path):
            print(f"SKIP  {drive_rel}  (not found on Drive - upload it first)")
            continue

        _force_symlink(drive_path, project_path)
        print(f"{project_rel} <- {drive_path}")

    print(f"\n[4/5] Discovering web-scraped data …")
    _discover_web_scraped_data(drive)

    print(f"\n[5/5] Checking resume state (checkpoints & completed experiments) …")
    _print_resume_status(drive)

    print(f"\n{'='*60}")
    print("Setup complete - all paths are ready.")
    print(f"{'='*60}\n")


def _print_resume_status(drive: Path) -> None:
    """Detect and report what will be resumed on the next training run.

    - Completed experiments  → read from experiments/metrics/training_results.csv
    - In-progress transformers → look for saved_models/**/checkpoints/ckpt_latest/meta.json
    """
    # Completed experiments
    results_csv = drive / "experiments" / "metrics" / "training_results.csv"
    completed = 0
    if results_csv.exists():
        try:
            df = pd.read_csv(results_csv)
            completed = df["experiment_key"].nunique()
            print(f"Completed experiments: {completed}")
            # Show top 3 by f1
            top = df.nlargest(min(3, len(df)), "f1_score")
            for _, row in top.iterrows():
                print(f"{row['experiment_key']}  F1={row['f1_score']:.4f}")
        except Exception as e:
            print(f"Could not read results CSV: {e}")
    else:
        print("  ℹ️  No completed experiments yet (first run).")

    # In-progress transformer checkpoints
    ckpt_pattern = str(drive / "saved_models" / "**" / "checkpoints" / "ckpt_latest" / "meta.json")
    meta_files = glob.glob(ckpt_pattern, recursive=True)

    if meta_files:
        print(f"In-progress transformer checkpoints: {len(meta_files)}")
        for meta_path in sorted(meta_files):
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                epoch = meta.get("epoch", "?")
                total = meta.get("num_epochs", "?")
                name = meta.get("model_name", "unknown")
                dataset = meta.get("dataset_name", "unknown")
                loss = meta.get("epoch_loss", 0)
                print(f"{dataset}/{name}  epoch {int(epoch)+1}/{total}  loss={loss:.4f}")
            except Exception:
                print(f"{meta_path} (could not parse)")
    else:
        print("No in-progress transformer checkpoints.")

    # Web scraped data
    web_dir = drive / "research" / "configs" / "web_scraped_data"
    web_csvs = sorted(web_dir.glob("*.csv")) if web_dir.exists() else []
    if web_csvs:
        print(f"Web scraped data files: {len(web_csvs)}")
        for csv_path in web_csvs:
            try:
                df = pd.read_csv(csv_path)
                print(f"{csv_path.name}  ({len(df)} samples)")
            except Exception:
                print(f"{csv_path.name}  (could not read)")
    else:
        print("No web scraped data yet (run collect_web first).")

    # Processed web data
    processed_dir = web_dir / "processed" if web_dir.exists() else None
    if processed_dir and processed_dir.exists():
        proc_csvs = sorted(processed_dir.glob("*.csv"))
        if proc_csvs:
            print(f"Processed web data files: {len(proc_csvs)}")
            for csv_path in proc_csvs:
                try:
                    df = pd.read_csv(csv_path)
                    print(f"{csv_path.name}  ({len(df)} samples)")
                except Exception:
                    print(f"{csv_path.name}  (could not read)")
        else:
            print("No processed web data yet (run clean_web_data first).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Colab environment setup")
    parser.add_argument(
        "--project",
        default=os.getcwd(),
        help="Absolute path to the cloned project root (default: cwd)",
    )
    parser.add_argument(
        "--drive",
        default="/content/drive/MyDrive/fake-news-results",
        help="Google Drive folder for persistent data",
    )
    args = parser.parse_args()
    setup(args.project, args.drive)
