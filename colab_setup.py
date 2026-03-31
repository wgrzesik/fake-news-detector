import argparse
import os
import shutil
from pathlib import Path


OUTPUT_DIRS = [
    "saved_models",
    "experiments",
    "mlruns",
    "mlartifacts",
    "outputs",
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


def setup(project_root: str, drive_root: str) -> None:
    project = Path(project_root).resolve()
    drive   = Path(drive_root).resolve()

    print(f"\n{'='*60}")
    print(f"  Colab Setup")
    print(f"  Project : {project}")
    print(f"  Drive   : {drive}")
    print(f"{'='*60}\n")

    print("[1/3] Linking OUTPUT directories (project -> Drive) …")
    for name in OUTPUT_DIRS:
        drive_path   = str(drive / name)
        project_path = str(project / name)
        os.makedirs(drive_path, exist_ok=True)
        _force_symlink(drive_path, project_path)
        print(f"{name} -> {drive_path}")

    print("\n[2/3] Linking OUTPUT files (project -> Drive) …")
    for name in OUTPUT_FILES:
        drive_path   = str(drive / name)
        project_path = str(project / name)
        if not os.path.exists(drive_path):
            open(drive_path, "w").close()
        _force_symlink(drive_path, project_path)
        print(f"{name} -> {drive_path}")

    print("\n[3/3] Linking INPUT data (Drive -> project) …")
    for project_rel, drive_rel in INPUT_DIRS.items():
        drive_path   = str(drive / drive_rel)
        project_path = str(project / project_rel)

        if not os.path.exists(drive_path):
            print(f"SKIP  {drive_rel}  (not found on Drive - upload it first)")
            continue

        _force_symlink(drive_path, project_path)
        print(f"{project_rel} <- {drive_path}")

    print(f"\n{'='*60}")
    print("Setup complete - all paths are ready.")
    print(f"{'='*60}\n")


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
