# Decisions Log

One entry per decision. Status is either **Settled** (build against it) or **Open** (needs a deliberate call before the phase/ticket that touches it — don't let it get decided by default).

## Settled

- **Language & tooling** — Python 3.11+, packaged via `pyproject.toml`. Fast to iterate on while the model is still being tuned against real footage; the hot paths (capture, inference) are C/C++ under the hood regardless of the glue language.
- **Detection approach** — a trained model (YOLO), not classical background-subtraction/thresholding. Confirmed by every real competitor researched: SwingVision, PlayReplay, and Foxtenn are all ML- or purpose-built-sensor based, none rely on simple thresholding for the actual ball.
- **Model & export path** — YOLOv8n (or newer Ultralytics nano variant), fine-tuned as a single-class "tennis ball" detector, exported to NCNN for on-device inference. Chosen for Ultralytics' train→export tooling maturity, not because it's the most accurate architecture available.
- **Seed dataset** — Roboflow Universe `viren-dhanwani/tennis-ball-detection`, pinned to a specific version and downloaded via `scripts/download_dataset.py`, which records a `manifest.json` fingerprint so every model checkpoint is traceable to exact training data. Own-footage fine-tuning (from the actual baseline/net-post-mounted angle) comes later, once real camera hardware exists — public sets are shot from broadcast angles that won't fully transfer.
- **MLOps scope for now** — lightweight JSON run manifests (dataset version, config, git commit, metrics), no DVC/MLflow/W&B server, no Docker. Right-sized for a two-person team at Phase A; revisit only if/when fleet scale actually demands it.
- **CV debugging tooling** — Rerun (rerun.io) for frame-by-frame detection visualization in the replay harness, matching what the closest funded competitor (PlayReplay) uses internally for the same purpose.
- **Testing & quality gates** — every ticket from A2 onward must ship unit tests, pass `ruff` (lint + format), pass `mypy`, and pass CI before being considered done. Established once, in A2, rather than left to each ticket's discretion.
- **Repo & attribution** — `github.com/Gagansharma-code/Linecall`. Only the user and Claude push to the remote; Cursor edits files but does not commit or push. No `Co-Authored-By` trailers — commit history carries the user's name/email only.

## Open — needs a deliberate call before the phase that touches it

- **Camera placement** (blocks Phase C / Stage 02). Two real options, both defensible:
  - *Opposite-baseline* (original doc's design) — wide 24m stereo baseline, best depth precision along the court's length, needs two independently-mounted poles and a wireless sync link between them.
  - *Net-post-mounted, multiple cameras per post* — what the two real, budget-tier, certified competitors actually ship (PlayReplay: 2 cams/post × 2 posts; Baseline Vision: single net-post unit). Faster one-person install, better near-net coverage, but a shorter effective baseline and the same single-mount limitation the original doc explicitly designed around.
  - Decide with real trajectory-accuracy data from Phase C testing, not before — but don't let Phase C code get written against an unstated default.
- **Business model** (blocks Phase F). Two real options:
  - *Coin-operated, pay-per-session, unstaffed public court* (original doc's premise) — genuinely novel; no well-funded competitor has publicly validated this exact model in tennis. Closest precedent is the padel industry's newer smart-lock unmanned-access pattern, and even that's usually sold through a club, not a raw public park.
  - *B2B facility annual subscription* — what every funded tennis competitor actually sells (PlayReplay: $3,500–4,500/court/year to the facility, not per-session to the player).
  - Does not block any CV or hardware work before Phase F.
- **Per-viewpoint redundancy** (revisit alongside camera placement). Current 2-node design has exactly one camera per end — a single point of failure per view. Foxtenn (5 sensors covering every ball) and PlayReplay (2 cameras per post) both build in redundancy this design currently lacks.
