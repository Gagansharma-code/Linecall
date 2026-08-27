# Roadmap

Each ticket is written as a standalone Cursor prompt (senior-eng-writes-spec / junior-eng-executes workflow) and reviewed before the next one is issued. See `DECISIONS.md` for what's settled vs. still open at each phase boundary.

## Phase A — Detection model + validation harness (current)
No camera/hardware, no networking, no payment code. Goal: prove a nano YOLO model exported to NCNN can find the ball reliably, on recorded footage, before spending on hardware.

- [ ] **A1** — repo scaffold, environment, dataset acquisition
- [ ] **A2** — testing, linting, type-checking, and CI infrastructure
- [ ] **A3** — collect + ingest real amateur public-court footage (see below — blocks A4's validity, not its construction)
- [ ] **A4** — YOLOv8n fine-tuning pipeline + run manifest (trains on seed + amateur data, with lighting/shadow/blur augmentation)
- [ ] **A5** — NCNN export + parity check against the PyTorch model
- [ ] **A6** — Rerun-backed replay harness
- [ ] **A7** — benchmark harness + report, broken out by data source (broadcast vs. amateur), not one blended number
- [ ] **A8** — active-learning flagging stub for low-confidence detections

Every ticket from A2 onward is expected to come with unit tests and pass lint/type-check/CI before it's considered done — A2 is what makes that enforceable instead of aspirational.

**A3 is a physical-world task, not a Cursor ticket:** the only seed data pulled so far (`viren-dhanwani/tennis-ball-detection`, A1) was verified by direct inspection to be 100% professional broadcast footage — perfect stadium lighting, pristine lines, elevated wide camera angle — across every filename group in it, not a mix. It's fine for validating that the training/export/harness pipeline works mechanically, but a model trained on it alone has no evidence it generalizes to a public court's harsh sun, dim evening light, or worn markings. A4 can still be built and run against the seed set to prove the pipeline works, but **Phase A isn't considered done, and nothing should move to Phase B, until real amateur-court footage has been folded into training and A7 shows it's actually being detected well** — not just the broadcast set. Action: record a few minutes of real play at a public court on a phone (doesn't need the final Pi/camera rig) — ideally one clip in harsh midday sun, one overcast or evening, and a court with normal wear rather than fresh paint.

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
