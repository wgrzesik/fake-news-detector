import json
import os
from pathlib import Path
from typing import Dict, Any


def load_colab_config() -> Dict[str, Any]:
    """
    Load Colab configuration from secure file.

    Usage:
        config = load_colab_config()
        REPO_URL = config['github']['repo_url']
        BRANCH = config['github']['branch']
        DRIVE_ROOT = config['colab']['drive_root']
        PROJECT_DIR = config['colab']['project_dir']

    Returns:
        Dictionary with keys: github, colab

    Raises:
        FileNotFoundError: If colab_secrets.json doesn't exist
    """

    # Look for secrets file in project root
    project_root = Path(__file__).parent
    secrets_file = project_root / "colab_secrets.json"

    if not secrets_file.exists():
        raise FileNotFoundError(
            f"[ERROR] colab_secrets.json not found at {secrets_file}\n"
            f"Create it from colab_secrets.json.template:\n"
            f"  1. Copy colab_secrets.json.template to colab_secrets.json\n"
            f"  2. Fill in your GitHub token (GitHub → Settings → Developer settings → Personal access tokens)\n"
            f"  3. DO NOT commit colab_secrets.json to git (it's in .gitignore)\n"
        )

    with open(secrets_file, 'r') as f:
        config = json.load(f)

    # Validate required fields
    required_github_fields = ['username', 'token', 'branch']
    required_colab_fields = ['drive_root', 'project_dir']

    for field in required_github_fields:
        if field not in config.get('github', {}):
            raise ValueError(f"Missing github.{field} in colab_secrets.json")

    for field in required_colab_fields:
        if field not in config.get('colab', {}):
            raise ValueError(f"Missing colab.{field} in colab_secrets.json")

    # Check for placeholder token
    if config['github']['token'] == 'YOUR_GITHUB_TOKEN_HERE':
        raise ValueError(
            "[ERROR] GitHub token not set!\n"
            "Edit colab_secrets.json and replace 'YOUR_GITHUB_TOKEN_HERE' with your actual token.\n"
            "Get token from: https://github.com/settings/tokens"
        )

    # Construct repo URL if needed
    username = config['github']['username']
    token = config['github']['token']
    config['github']['repo_url'] = (
        f"https://{username}:{token}@github.com/{username}/fake-news-detector.git"
    )

    return config


if __name__ == "__main__":
    try:
        config = load_colab_config()
        print("[OK] Configuration loaded successfully!")
        print(f"    GitHub: {config['github']['username']}")
        print(f"    Branch: {config['github']['branch']}")
        print(f"    Drive root: {config['colab']['drive_root']}")
        print(f"    Project dir: {config['colab']['project_dir']}")
    except Exception as e:
        print(f"[ERROR] {e}")

