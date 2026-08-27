# LineCall

LineCall is a production tennis-ball line-calling system: a Raspberry Pi 5 will eventually run real-time detection and decide whether a ball landed in or out. The repo is currently in the dataset and model-training phase — this ticket only scaffolds the Python project and a reproducible download of a pinned public training set from Roboflow Universe.

## Setup

Requires Python 3.11+. From the repo root:

```bash
python -m venv .venv
```

Activate the venv (Windows: `.venv\Scripts\activate`, Linux/macOS: `source .venv/bin/activate`), then:

```bash
pip install -e ".[dev]"
cp .env.example .env
```

Open `.env` and set `ROBOFLOW_API_KEY` to a key from your own Roboflow account (free tier). Then pull the pinned dataset:

```bash
python scripts/download_dataset.py
```

The export lands in `data/raw/tennis-ball-detection-v6/` with a `manifest.json` fingerprint. Re-running the script is a no-op if that directory already matches the pinned workspace/project/version.

```bash
pytest
```

## What this repo does not do yet

There is no model training, no inference, no camera or Raspberry Pi capture code, and no networking or payment logic. Those are separate tickets. Do not treat this scaffold as a finished detection pipeline.
