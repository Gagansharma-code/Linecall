# Roadmap

Each ticket is written as a standalone Cursor prompt (senior-eng-writes-spec / junior-eng-executes workflow) and reviewed before the next one is issued. See `DECISIONS.md` for what's settled vs. still open at each phase boundary.

## Phase A — Detection model + validation harness (current)
No camera/hardware, no networking, no payment code. Goal: prove a nano YOLO model exported to NCNN can find the ball reliably, on recorded footage, before spending on hardware.

- [x] **A1** — repo scaffold, environment, dataset acquisition
- [x] **A2** — testing, linting, type-checking, and CI infrastructure
- [x] **A3** — collect + ingest real amateur public-court footage (extraction done: 5 clips, ~63 min, 3 venues/lighting conditions → 1000 candidate frames at `C:\linecall-data\candidate_frames`; annotated via Roboflow, class `tennis ball`. Final: 298 approved / 194 rejected / 507 unannotated — accepted with known residual noise rather than exhaustively re-verified, see `DECISIONS.md`'s auto-labeling finding. Good enough to unblock A4; A7 is the real quality check.)
- [x] **A4** — YOLOv8n fine-tuning pipeline + run manifest. Verified end-to-end on this machine (CPU only, no CUDA): merged 877 images (578 broadcast + 299 amateur), fine-tuned YOLOv8n with a custom Albumentations pipeline (RandomShadow/RandomBrightnessContrast/MotionBlur) confirmed active via Ultralytics' own pipeline log, not just echoed config. A full 30-epoch run measured at ~2 hours on this CPU — too long for a pipeline-validation run, so the verification run used 8 epochs (27 min): precision 0.591, recall 0.458, mAP50 0.481, mAP50-95 0.163. **These numbers are not a real accuracy result** — 8 epochs proves the pipeline works, it does not represent a trained model. A real training run (30+ epochs, likely needing GPU access given the CPU pace) is still needed before A7's benchmark means anything.
- [x] **A5** — NCNN export + parity check. Verified on this machine: exported the A4 checkpoint via Ultralytics' NCNN path, then compared PyTorch vs. NCNN detections on 25 sampled test images — mean matched IoU 0.9576, mean confidence diff 0.0138, PASS against defined export-fidelity bars (reproduced independently, byte-identical results). Also fixed a real gap: `mypy` was scanning Ultralytics' auto-generated `model_ncnn.py` under `runs/` (gitignored, so CI never saw it, but it broke locally once a real export existed) — `runs/` and `data/` added to mypy's exclude pattern.
- [x] **A6** — Rerun-backed replay harness. Verified on this machine: real run against actual amateur footage through the NCNN export, and independently confirmed via Rerun's own CLI (`rrd verify` + `rrd stats`, not just file size) that the output `.rrd` genuinely contains the logged video frames, detection boxes, and confidence timeline it's supposed to.
- [x] **A7** — benchmark harness with per-source reporting. **This is the number Phase A was built to produce.** Measured on the current 8-epoch smoke-test checkpoint (not a real training run): broadcast mAP50 0.6193 / recall 0.560 vs. amateur mAP50 0.3019 / recall 0.279 — a −0.317 mAP50 gap, amateur worse. Recall dropping by half specifically indicates the model is genuinely missing real balls in amateur frames more often, not just measurement noise from the amateur test set's own residual label imperfections (see A3's auto-labeling finding — some of these 30 ground-truth boxes could themselves be imperfect, but that alone doesn't explain a recall collapse this large). **By this roadmap's own exit criterion, Phase A is not yet done: A7 does not show the model detecting well on amateur footage.** Expected, given this checkpoint has had 8 epochs of CPU training, not a real one — see `DECISIONS.md`'s GPU-access note. Re-run A7 after a real training run before revisiting this conclusion.
- [ ] **A8** — active-learning flagging stub for low-confidence detections (optional / lower priority now — the more urgent gap is a real training run, not more tooling)

Every ticket from A2 onward is expected to come with unit tests and pass lint/type-check/CI before it's considered done — A2 is what makes that enforceable instead of aspirational.

**A3's original concern (broadcast-only seed data) is now addressed as far as data and tooling go** — real amateur public-court footage exists, annotated, and folded into training, and A7 now proves it's actually being measured, not just present. **What's not resolved: the model itself doesn't yet detect amateur footage well.** Nothing moves to Phase B until a real training run (not an 8-epoch smoke test) closes enough of this gap, re-measured by re-running A7 — that's the actual remaining blocker on Phase A, and it's a training/compute question now, not a data or tooling one.

## Phase B — Single camera, live, real-time
Needs: 1× Raspberry Pi 5 + 1× IMX296 Global Shutter camera.
`picamera2` capture → Kalman/optical-flow tracker every frame, YOLO keyframe detector every 8–10th frame on a cropped search window. Exit: stable x/y/t stream at full frame rate on the actual Pi 5, not a laptop.

## Phase C — Two cameras + geometry
Needs: second Pi 5 + camera rig, and the **camera placement decision** from `DECISIONS.md` resolved first.
UDP coordinate link + clock-offset handshake, automatic court-line calibration, `cv2.triangulatePoints` + `scipy.optimize.curve_fit` trajectory fit. Exit: a 3D bounce point from two synchronized 2D streams, with a measured sync error.

## Phase D — The call itself
Point-in-polygon test against calibrated geometry, tolerance margin sized from measured (not guessed) sync error, SQLite logging, confidence score, "close call" fallback, LED/tone output. Exit: a full point, end to end, produces a stored, explainable call.

## Phase E — Field validation
Outdoor testing against chalk-marked or independently-filmed ground truth, error measured in centimeters. This number — not a guess — sets the production tolerance margin. **Nothing in Phase F starts until this passes.**

## Phase F — Business/payment layer
Needs: the **business model decision** from `DECISIONS.md` resolved first.
Manual start/stop switch → Stripe Payment Link (tested solo, no hardware) → webhook-to-MQTT bridge → fleet heartbeat/status dashboard once there's more than one court.
