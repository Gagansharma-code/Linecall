# Roadmap

Each ticket is written as a standalone Cursor prompt (senior-eng-writes-spec / junior-eng-executes workflow) and reviewed before the next one is issued. See `DECISIONS.md` for what's settled vs. still open at each phase boundary.

## Phase A — Detection model + validation harness (current)
No camera/hardware, no networking, no payment code. Goal: prove a nano YOLO model exported to NCNN can find the ball reliably, on recorded footage, before spending on hardware.

- [x] **A1** — repo scaffold, environment, dataset acquisition
- [x] **A2** — testing, linting, type-checking, and CI infrastructure
- [x] **A3** — collect + ingest real amateur public-court footage (extraction done: 5 clips, ~63 min, 3 venues/lighting conditions → 1000 candidate frames at `C:\linecall-data\candidate_frames`; annotated via Roboflow, class `tennis ball`. Final: 298 approved / 194 rejected / 507 unannotated — accepted with known residual noise rather than exhaustively re-verified, see `DECISIONS.md`'s auto-labeling finding. Good enough to unblock A4; A7 is the real quality check.)
- [x] **A4** — YOLOv8n fine-tuning pipeline + run manifest. Verified end-to-end on this machine (CPU only, no CUDA): merged 877 images (578 broadcast + 299 amateur), fine-tuned YOLOv8n with a custom Albumentations pipeline (RandomShadow/RandomBrightnessContrast/MotionBlur) confirmed active via Ultralytics' own pipeline log, not just echoed config. A full 30-epoch run measured at ~2 hours on this CPU — too long for a pipeline-validation run, so the verification run used 8 epochs (27 min): precision 0.591, recall 0.458, mAP50 0.481, mAP50-95 0.163. **These numbers are not a real accuracy result** — 8 epochs proves the pipeline works, it does not represent a trained model. A real training run (30+ epochs, likely needing GPU access given the CPU pace) is still needed before A7's benchmark means anything.
- [ ] **A5** — NCNN export + parity check against the PyTorch model
- [ ] **A6** — Rerun-backed replay harness
- [ ] **A7** — benchmark harness + report, broken out by data source (broadcast vs. amateur), not one blended number
- [ ] **A8** — active-learning flagging stub for low-confidence detections

Every ticket from A2 onward is expected to come with unit tests and pass lint/type-check/CI before it's considered done — A2 is what makes that enforceable instead of aspirational.

**A3's original concern (broadcast-only seed data) is now addressed:** real amateur public-court footage exists, annotated, and ready to fold into training. **Phase A still isn't fully done, and nothing moves to Phase B, until A7 actually shows the model detecting well on the amateur/held-out set** — not just the broadcast one. That's now purely a training + measurement question (A4–A7), not a data-collection one.

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
