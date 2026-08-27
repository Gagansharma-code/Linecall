"""Download a pinned Roboflow Universe tennis-ball detection dataset.

Reads workspace/project/version/format from configs/dataset.yaml and writes
the export under data/raw/<project>-v<version>/ plus a manifest.json that
records exactly what was pulled. Safe to re-run: a matching manifest skips
the network call.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from roboflow import Roboflow

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "dataset.yaml"
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "raw"

REQUIRED_CONFIG_KEYS = ("workspace", "project", "version", "format")


class MissingAPIKeyError(Exception):
    """Raised when ROBOFLOW_API_KEY is not set in the environment."""


def load_dataset_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    missing = [key for key in REQUIRED_CONFIG_KEYS if key not in raw]
    if missing:
        raise ValueError(f"dataset config {path} is missing keys: {', '.join(missing)}")
    return {
        "workspace": str(raw["workspace"]),
        "project": str(raw["project"]),
        "version": int(raw["version"]),
        "format": str(raw["format"]),
    }


def require_api_key() -> str:
    key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not key:
        raise MissingAPIKeyError(
            "ROBOFLOW_API_KEY is not set. Copy .env.example to .env and add "
            "your Roboflow API key (free tier)."
        )
    return key


def destination_dir(data_root: Path, project: str, version: int) -> Path:
    return data_root / f"{project}-v{version}"


def compute_fingerprint(dataset_dir: Path) -> str:
    """SHA-256 of the sorted relative paths and file sizes (not file contents)."""
    lines: list[str] = []
    for path in sorted(dataset_dir.rglob("*")):
        if not path.is_file() or path.name == "manifest.json":
            continue
        relative = path.relative_to(dataset_dir).as_posix()
        lines.append(f"{relative}\t{path.stat().st_size}")
    payload = "\n".join(lines).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_manifest(dataset_dir: Path, config: dict[str, Any]) -> Path:
    manifest = {
        "workspace": config["workspace"],
        "project": config["project"],
        "version": config["version"],
        "format": config["format"],
        "downloaded_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sha256": compute_fingerprint(dataset_dir),
    }
    path = dataset_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def manifest_matches(dataset_dir: Path, config: dict[str, Any]) -> bool:
    path = dataset_dir / "manifest.json"
    if not dataset_dir.is_dir() or not path.is_file():
        return False
    try:
        recorded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return (
        recorded.get("workspace") == config["workspace"]
        and recorded.get("project") == config["project"]
        and int(recorded.get("version", -1)) == config["version"]
        and recorded.get("format") == config["format"]
    )


def download_dataset(config: dict[str, Any], dest: Path, api_key: str) -> str:
    """Pull the pinned dataset into dest. Returns 'skipped' or 'downloaded'."""
    if manifest_matches(dest, config):
        print(
            f"Dataset already present at {dest} "
            f"({config['workspace']}/{config['project']} v{config['version']}). "
            "Skipping download."
        )
        return "skipped"

    dest.parent.mkdir(parents=True, exist_ok=True)
    client = Roboflow(api_key=api_key)
    project = client.workspace(config["workspace"]).project(config["project"])
    version = project.version(config["version"])
    version.download(
        model_format=config["format"],
        location=str(dest),
        overwrite=True,
    )
    write_manifest(dest, config)
    print(
        f"Downloaded {config['workspace']}/{config['project']} "
        f"v{config['version']} ({config['format']}) to {dest}"
    )
    return "downloaded"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download the pinned tennis-ball detection dataset from Roboflow."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to dataset.yaml (default: configs/dataset.yaml)",
    )
    args = parser.parse_args(argv)

    load_dotenv(REPO_ROOT / ".env")
    try:
        api_key = require_api_key()
    except MissingAPIKeyError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    config = load_dataset_config(args.config)
    dest = destination_dir(DEFAULT_DATA_ROOT, config["project"], config["version"])
    download_dataset(config, dest, api_key)
    return 0


if __name__ == "__main__":
    sys.exit(main())
