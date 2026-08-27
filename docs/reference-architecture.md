# Line-Call Pipeline

**Reference architecture — coin-operated public install**

A two-camera design for calling tennis balls in or out at a public court — permanently mounted, paid for by QR code or Apple Pay, and asleep until someone taps to play. Built around one rule that now covers the business too: the AI decides from a live feed, and the software only has to talk to a person at the two moments that matter — paying, and playing.

| Camera nodes | Capture rate | Pay → live | Call latency | Idle power |
|---|---|---|---|---|
| 2, opposite ends | 120–150 fps | < 5 s | < 700 ms | cameras off |

> **Status note (see `DECISIONS.md`):** this document is the original exploratory research/architecture pass and is kept verbatim for reference. Camera placement (stage 02, opposite-baseline here) and the coin-operated business model (stages 09–13) are both flagged as **open decisions** in `DECISIONS.md` after competitive research — don't treat everything below as settled.

---

## 00 · Governing idea
**Decouple seeing from keeping**

Every design decision below follows from one split: **detection happens on the live frame, in memory, on the node that captured it** — the call is made and logged before anything is compressed. What gets written to disk afterward exists for the player to replay a rally, not for the algorithm. Once you stop asking storage to also serve pixel-perfect ball detection, the two problems — "see a 5-pixel ball at 250 km/h" and "don't fill an SD card in one afternoon" — stop fighting each other.

*This edition trades Hawk-Eye's camera count for something a single player can carry, set up alone, and trust for a friendly match — not for a challenge system. Two nodes, opposite baselines, no installer, no court crew.*

---

## 01 · Capture layer
**One Pi 5 + GS camera per viewpoint**

Each camera is its own edge node: a Raspberry Pi 5 with the IMX296 Global Shutter camera, running a cropped capture window rather than the full sensor. Published tracking systems get reliable ball detection at 720p–1280×720 and 25–30 fps — your sensor's native 1456×1088 already exceeds that. The lever worth pulling isn't resolution, it's **frame rate**: more samples per second means less motion blur on a ball that can cross the frame in a few dozen milliseconds, and a shorter gap for the trajectory fit in stage 05 to interpolate across.

Crop each node's capture to the court boundary plus a small margin — the AAAI low-cost tracking system did exactly this to cut both data volume and downstream processing. A smaller frame at the sensor's max readout speed also buys you a higher achievable frame rate, since the GS sensor trades resolution for speed.

| Setting | Recommendation | Why |
|---|---|---|
| Crop region | court + ~1 m margin | discard pixels no ball can ever occupy |
| Frame rate | 120–150 fps | freeze a 250 km/h serve to <60 cm of travel per frame |
| Color | mono / luma-only | ball shape & motion don't need chroma; halves raw bandwidth |
| Exposure | short, fixed | GS already kills rolling-shutter skew — keep it that way with a short shutter to also kill blur |

---

## 02 · Camera placement
**Two nodes, opposite baselines, elevated**

With two nodes, put them behind each baseline, opposite each other, raised on a pole or tripod — roughly 3 m up, tilted down the full length of the court. That gives you the longest possible stereo baseline (the ~24 m of the court itself), which is actually the geometry triangulation wants: two views separated by a wide angle resolve depth far better than two cameras bunched close together. It's a different trade than a single net-post unit with two lenses a few centimetres apart — a real consumer product on the market today ([Baseline Vision](https://www.baselinevision.com/product)) takes that close-stereo, single-mount approach instead, and leans on "predictive calculations" to cover what the short baseline and net-height vantage can't see cleanly. Opposite-baseline placement gets you the wide-baseline depth accuracy that approach trades away, at the cost of needing two separate units and a wireless link between them.

The honest trade-off versus more cameras: with only two views, if a player's body blocks the ball from *both* ends at once — rare, but it happens right around the net on a passing shot — you lose triangulation entirely for that instant. **Elevation is your main defense**: a camera looking down at ~15–20° over a 1.8 m player has a much narrower "shadow" than one at head height, so raising both units as high as the pole/tripod allows meaningfully cuts how often this happens. Stage 04 covers the fallback for when it does anyway.

> **Fig. 1** — Both nodes look the full length of the court from opposite, elevated ends. The wide 24 m baseline favors depth accuracy; the height keeps a player's shadow narrow in each view rather than blocking it outright. *(Side-view diagram: one elevated camera pole behind each baseline, tilted inward, fields of view overlapping across the full court; the two nodes link wirelessly to each other overhead.)*

---

## 03 · Synchronization
**No cable across the court — so Wi-Fi has to be good enough**

Because this version gets permanently mounted rather than carried court to court, running a buried or aerial cable between the two poles at install time is a one-time cost, not a per-session hassle — if the budget allows it, that buys back tournament-grade PTP sync and the tighter accuracy that comes with it. If the install budget favors two independent solar/mains poles with no cable run between them, a dedicated point-to-point wireless bridge (aimed once, unlike a phone's Wi-Fi) is the fallback — better than ordinary Wi-Fi jitter, though still not cable-grade.

Do the arithmetic on what that costs you: a recreational serve or groundstroke at 70 m/s (250 km/h) moves ≈ 70 mm per millisecond. A few milliseconds of sync error is a few centimetres of positional uncertainty on the reconstructed bounce — noticeably looser than a tournament-grade rig, but on a par with what recreational players already accept from their own eyes on a fast, close call. Have each node estimate the other's clock offset and drift at start-of-session (a handful of round-trip timing pings is enough, the same idea NTP uses) and re-check periodically through the session rather than assuming the offset holds for hours.

> Rule of thumb: **position error ≈ ball speed × sync error**. At recreational serve speeds, budget for roughly 2–5 cm of Wi-Fi-driven uncertainty — call it out in the app's confidence score rather than pretending the call is millimeter-exact.

---

## 04 · Edge detection
**Send coordinates, not video — and predict through gaps**

Each node runs its own lightweight detector on the frames as they arrive: background subtraction plus blob/contour filtering (fast, and what most published low-cost systems use), or a small CNN if you need robustness against shadows, court-color balls, or variable outdoor light. The output per frame is tiny — a pixel coordinate, a confidence score, and the synchronized timestamp — and that's the only thing that crosses the Wi-Fi link to whichever node is acting as aggregator. This is also what makes the occlusion case from stage 02 survivable: when one node briefly loses the ball behind a player, it simply reports "no detection" for those frames instead of a wrong one, and the fusion stage falls back to extrapolating from the other node's clean view plus the flight's known physics — a lower-confidence estimate rather than a blind guess, and the same "predict through the gap" idea the Baseline Vision product markets for doubles play.

---

## 05 · Fusion & trajectory
**2D detections → one 3D flight path**

One of the two nodes doubles as the aggregator (a Pi 5 has plenty of headroom for this once it's not also running a hardware encoder — Pi 5 has no dedicated hardware video encoder, so all video encoding is software-based anyway): it receives the other node's timestamped 2D detections over Wi-Fi and triangulates each instant into a 3D ball position, using a one-time camera calibration for each node's pose. That calibration doesn't need a tape measure — solve it automatically from the court's own painted lines, detected the moment the unit is switched on, the same court-model calibration technique used in broadcast tennis analysis. The player's entire "setup" is placing two poles at opposite baselines and turning them on.

*The 24 m opposite-baseline separation is a wide stereo baseline, which is good news for depth precision along the court's length — the axis a short net-post baseline struggles with most. The trade is weaker precision very close to either camera's own baseline, where the two viewing angles become more similar; the physics-based fit below is what compensates.*

Raw triangulated points are noisy. Fit a physics-based curve — a parabolic or drag-corrected trajectory model — through the recent points, the same technique used in both the Stanford and AAAI systems. This does two things at once: it smooths sensor noise, and it lets you find the bounce instant *between* two captured frames rather than being limited to whatever frame happened to catch it — sub-frame precision from curve-fitting, not from throwing more raw resolution at the problem.

---

## 06 · The call
**A light, a tone, and a phone — no umpire needed**

The bounce coordinate is compared against the calibrated line geometry with a tolerance band sized to what stage 03 actually measured that session — not a fixed number, since Wi-Fi conditions vary court to court. For a recreational match, the output should be as unobtrusive as a real line judge: an LED flash and a short tone on the unit itself for "out," silence for "in," which is exactly the pattern the closest real consumer product on the market uses. A companion phone app carries the detail — the call, a confidence indicator, and a quick replay clip — for the rare moment someone wants to check rather than play on.

Keep the "don't know" case, but make it fit casual play: when confidence is low (the occlusion case from stage 04, or a serve near the edge of both nodes' tolerance), the honest move is a distinct "close call" signal rather than a confident-sounding wrong answer — friends can finish the point by eye, the same way they would without the system at all.

---

## 07 · Storage & archive
**A season's worth of sessions on one SD card**

Because the call is already made by stage 06, the archive's job shifts from "feed an algorithm" to "let a player scroll back through their session afterward." That's a much lower bar than tournament footage, and it's where nearly all the storage savings come from — no central database or facility server needed, just local storage on the aggregator node (or synced to the player's phone over the same Wi-Fi link already in place).

| What | Lives where | Kept how long |
|---|---|---|
| Raw high-fps frames | RAM / local buffer, per node | seconds — discarded after detection |
| Trajectory + call log | on-device storage | indefinitely — kilobytes per point |
| Compressed point clip | on-device / synced to app | full session, ROI-weighted H.265 |
| Full-quality replay clip | synced to app on request | only "close call" points |

For the clips you do keep, compress asymmetrically: background-subtract the same way stage 04 already does, then encode the static court at a heavy compression ratio and the ball/player region at higher quality. Published ROI-compression work reports 5–15× smaller files this way versus uniform compression, at visually near-identical quality — which matters more here than in a facility install, since the whole unit runs off a battery and a modest onboard card, not a server rack.

> **Fig. 2** — Video pixels never leave node A. Only coordinates cross the Wi-Fi link to node B, which fuses them, sounds the call in under a second, and writes a compressed clip locally, off the critical path. *(Diagram: node A → Wi-Fi (xy + t) → node B/aggregator [fuse + fit + call engine] → LED+tone output, and → clip+log → on-device storage / app.)*

---

## 08 · Field-ready design
**What breaks between a backyard test and any court, any Saturday**

- **Wi-Fi that survives a busy club.** Two units talking on a court next to a dozen other people's phones and hotspots need automatic channel selection and a private link (their own hotspot or direct device-to-device pairing) rather than assuming the venue's public Wi-Fi is clean enough for the sync budget in stage 03.
- **Calibrate at install, then keep checking.** A permanently mounted pole doesn't get bumped as often as a portable one, but wind, a stray racket, or a maintenance visit can still nudge it a few degrees — re-run the automatic court-line calibration from stage 05 on a schedule (weekly is reasonable) and flag anything that's drifted rather than trusting the day-one calibration forever.
- **Mains power with a battery backup, not a battery alone.** A fixed installation can run a proper power line to each pole; keep a small backup battery behind it so a brief outage doesn't kill a session someone already paid for, and so the device can report the outage instead of just going dark.
- **Graceful one-node failure.** If node B goes offline mid-session, node A alone can still estimate calls with the physics-fallback method from stage 04 — lower confidence, clearly labeled, better than the system going dark.
- **Weatherproof, since it's outdoors by default.** Sun glare and shifting shadows hit background-subtraction detectors hardest, and a device left on a court needs basic splash/dust resistance — this is exactly why the commercial unit above ships with an IP rating.
- **Give the installer a confirmation screen, not a guess.** Whoever mounts the poles should get an on-the-spot check — both units see the whole court and each other, calibration succeeded — before they leave the site, catching a badly aimed camera in ten seconds instead of after a week of wrong calls nobody reported.

---

## 09 · How to build it, in order
**The software splits into four systems — build them in this sequence**

Everything from here down is the layer that turns the camera pipeline (stages 00–08, already designed) into a business. In plain terms, there are four pieces of software, and they genuinely are easiest to build in this order — each one only makes sense once the one before it works.

| Order | Build | What "done" looks like |
|---|---|---|
| 1 | The calling software (stages 00–08) | Runs standing next to the court, on a bench, calling balls in/out reliably — no payment involved yet. |
| 2 | A manual start/stop switch | Pressing a button (or hitting an endpoint on the Pi) starts a timed session and puts the device back to sleep after — proves the idle/active behavior stage 12 formalizes. |
| 3 | A working payment link | Scanning a QR code with your own phone, paying $1 to yourself, and seeing that payment land in your Stripe dashboard — no hardware involved yet. |
| 4 | The bridge between them | That same $1 payment automatically triggers step 2's start switch on the actual device, within a few seconds, with no person in the loop. |

*Steps 1–3 can be built and tested completely independently, even by different people, before anyone connects them. Step 4 is usually the smallest amount of code and the biggest confidence boost — it's the moment the "product" exists.*

---

## 10 · Taking the payment
**Don't build a payment system — rent one**

The single most important simplification available to you: **never write code that touches a card number.** Accepting payments directly carries real legal and security weight (PCI compliance, fraud liability, encryption standards), and a payment processor like Stripe has already solved it. What you build instead is a link.

Create a **Stripe Payment Link** — a hosted checkout page Stripe builds and hosts for you — for each court, with the court's ID baked into the link (something like `pay.example.com/court-07`). Print that as a QR code and mount it next to the court. A player points their phone's ordinary camera at it — no app to install — and lands on a page that already knows how to show an Apple Pay button on an iPhone or a Google Pay button on Android, because Stripe's checkout page detects the device automatically. They pick a duration, pay, done. Every card number, every fraud check, every dispute — that's Stripe's problem, not yours.

> What your software actually has to know: which QR code belongs to which physical court, and what to do the moment Stripe says "this one's paid." Everything upstream of that is off-the-shelf.

---

## 11 · The bridge: turning "paid" into "on"
**A small always-listening line between your server and the pole**

When a payment clears, Stripe doesn't call the court directly — it calls **your** server, through something called a webhook: essentially, Stripe hits a web address you control and says "court-07 just paid for 30 minutes." That's a small piece of server code you write once; it doesn't need to be fancy, and it can run on the cheapest always-on cloud hosting available, since its whole job is: receive that message, look up which physical device is "court-07," and tell it to start.

"Tell it to start" needs its own always-open channel, because a public court usually can't be reached by a normal web request the way a phone can — it's not sitting waiting for one. The standard tool for this is **MQTT**, a lightweight messaging protocol built exactly for "cloud talks to a small, often-offline device." Each court's Pi keeps one quiet, persistent connection open to your server; your server publishes a short message like `{"cmd":"start","minutes":30}` to that court's channel, the Pi receives it in about a second, starts the session, and publishes a reply — `"started"` or `"failed: camera offline"` — so you always know what actually happened, not just what you asked to happen.

Because public courts often have weak or no usable Wi-Fi, give each unit its own cellular data connection (a small SIM-based modem, the same approach vending machines and payment terminals use) rather than depending on the venue's network. It's the one part of this system that isn't optional if you want it to work reliably on a random public court.

> **Fig. 3** — Money and hardware never talk directly; your server is the only thing that has to trust both sides, and it's the smallest piece of custom code in the whole system. *(Flow: player's phone → scan+pay → Stripe checkout → webhook → your server → MQTT "start" → court-07 Pi (cellular) → wakes on-court display; Pi sends heartbeat + "started" ack back to your server.)*

---

## 12 · Asleep until needed
**The Pi only ever has three states**

The device software at each court is simpler than it sounds once you draw it as a loop rather than a list of features. It sits in **SLEEP** — cameras off, detection loop not running, just listening on its MQTT channel — until a start command arrives with a paid duration. It moves to **ACTIVE**: cameras on, the full stages 00–08 pipeline running, a countdown ticking. When time is running low it moves briefly to **WARNING** — a light or a tone plus a fresh QR code on-screen to extend — and either an extend payment arrives (back to ACTIVE with more time) or the clock runs out and it drops back to SLEEP.

> `SLEEP → (paid) → ACTIVE → (time low) → WARNING → (extend or expire) → ACTIVE / SLEEP`

Keeping this as an explicit, small state machine — rather than scattered flags in the code — is what makes the device predictable to operate and easy to debug months later when you're not the one standing next to it.

---

## 13 · Running a fleet you can't see
**A heartbeat per court, and one screen for all of them**

Once there's more than one court, the operator's real day-to-day problem stops being "does the AI call the ball correctly" — you solved that in stages 00–08 — and becomes "is court-07 even switched on right now." Each Pi should send a short heartbeat (say, once a minute) to your server: alive, camera image looks normal, battery/power fine. Your server needs nothing more sophisticated than a page listing every court with a green or red dot and today's revenue — that single page is most of what "running the business" looks like day to day, and it's what tells you a court needs a service visit before a customer's bad experience does.

- **Missed heartbeats mean something specific.** No signal for a few minutes is different from "camera sees a fully black frame" (lens covered — possible vandalism) or "enclosure opened" (a tamper switch, if you fit one) — route these to different alerts, not one generic "offline."
- **A failed payment shouldn't need a phone call.** If the start command fails to reach the court after a successful payment (bad signal, device asleep for maintenance), have your server auto-refund through Stripe rather than leaving a stranger out $8 with no recourse.
- **New courts should be a config change, not new code.** Once the four systems in stage 09 exist, adding court-08 should mean shipping a unit, printing a QR code, and adding one row to a database — if it means touching the Pi's software or the server's logic, that's worth fixing before you scale further.

---

## 14 · Language & core libraries
**Building stages 00–08 — the calling software only, nothing from the payment layer**

Write it in **Python**, not C++, and don't second-guess that early. The parts that actually need to be fast — frame capture, image preprocessing, and neural-network inference itself — are already implemented in optimized C/C++ underneath the libraries below; Python is just the glue calling into them, so the interpreter overhead barely matters here. You get much faster iteration while you're still retraining and tuning against real footage, which you will be doing constantly in the first few months. Reach for C++ later, only for a specific hot loop you've measured and confirmed Python is the bottleneck for — not as a default.

| Job | Library | Why this one |
|---|---|---|
| Camera capture | `picamera2` | Raspberry Pi's own actively-maintained Python bindings for libcamera — the direct, supported path to the GS camera on Pi 5 |
| Ball detection (training + export) | `Ultralytics YOLO` | the most mature training/export toolchain for a small custom object detector — see stage 15 |
| On-device inference | `NCNN runtime` | the fastest measured export format for YOLO on Pi 5 CPU — see stage 15's benchmark |
| Image ops & between-frame tracking | `OpenCV (opencv-python)` | cropping, optical flow, and drawing — still doing real work, just not the primary detector anymore |
| Trajectory math | `NumPy + SciPy` | vectorized array math and curve fitting (`scipy.optimize`) for the physics fit in stage 05 |
| Tracking / smoothing | `filterpy` | a ready-made Kalman filter implementation rather than hand-rolling the linear algebra |
| Dataset annotation | `Roboflow` | hosts existing public tennis-ball datasets and lets you label your own footage in the same format |
| On-device storage | `sqlite3` | built into Python's standard library — see stage 18 |

---

## 15 · Choosing the detection model
**Revised after checking what actual competitors do — a trained detector, not background subtraction**

*This stage originally recommended starting with pure classical computer vision. Pulling apart how real competitors and the open-source community actually build this changed that recommendation — the reasoning below replaces it.*

Every serious tennis ball tracker, commercial or open-source, runs a trained model, not a background-subtraction threshold. SwingVision's own engineers say plainly that their app "is basically not possible without" Apple's Neural Engine, and that they "had to innovate a lot to make these models as lean as possible" — a company betting its whole product on deep learning, not classical CV. Every public GitHub project that reproduces Hawk-Eye-style tracking — [ArtLabss/tennis-tracking](https://github.com/ArtLabss/tennis-tracking), [nikhilgrad/Tennis-Ball-Tracker](https://github.com/nikhilgrad/Tennis-Ball-Tracker), [abhroroy365/Tennis-Tracker](https://github.com/abhroroy365/Tennis-Tracker) — uses either TrackNet or a fine-tuned YOLO model for the ball specifically, because a small, fast, blurry ball moving in front of variable lighting, shadows, and a moving player is exactly the case classical thresholding handles worst. That's a direct answer to the concern the earlier version of this stage raised: the risk isn't deep learning, it's deep learning you haven't fitted to your actual hardware.

So the model is **YOLO — specifically a nano-sized variant (YOLOv8n or newer), fine-tuned as a single-class "tennis ball" detector** — over TrackNet, for one practical reason: Ultralytics' YOLO has by far the most mature training and export tooling, with first-party paths to ONNX, NCNN, and OpenVINO. That tooling maturity is what actually gets a model off a laptop and onto a Pi; the ArtLabss project above, built on TrackNet, is a useful cautionary data point here — it needs a GPU and takes 28 minutes to process 15 seconds of footage, which tells you a research-grade model dropped in as-is, with no export or speed work, is nowhere close to real-time regardless of which architecture it started from.

> The hard number that has to shape this design: Ultralytics' own published Pi 5 benchmarks show a nano YOLO model at standard 640px input tops out around **15 fps** on the Pi 5 CPU even with the fastest export format (NCNN) — nowhere near the 120–150 fps this design's camera capture runs at. A model that accurate isn't automatically a model that's fast enough; both have to be engineered for, separately.

Three moves close that gap, all standard practice rather than exotic: shrink the input the detector actually looks at — you've already cropped to the court in stage 01, and once the tracker below has locked onto the ball you can run YOLO on a small search-window crop around its predicted position rather than the full court, which cuts the pixel count (and inference time) by an order of magnitude or more; run the heavy model as a **keyframe detector** rather than on every frame — detect at whatever rate the Pi 5 can sustain (roughly the 15 fps ballpark above, faster once cropped) and use a cheap classical tracker (OpenCV optical flow, or the Kalman filter already in this design) to carry the position forward at the full 120–150 fps between detections, exactly the interpolation strategy the nikhilgrad project above already uses to cover frames its YOLO model misses; and treat a small AI accelerator (a Coral USB Accelerator or a Hailo module, $60–$130) as a normal production line item rather than an exotic add-on if testing shows the CPU-only path still isn't enough — SwingVision's whole architecture is a bet on exactly this kind of dedicated inference hardware, just built into the iPhone instead of bolted onto a Pi.

> **Fig. 5** — The tracker never runs a neural network; YOLO only runs occasionally, on a small crop, to correct drift — this is how a Pi 5-class CPU gets both full-rate tracking and deep-learning-grade robustness. *(Diagram: every captured frame (120–150fps) → optical flow + Kalman predict → smoothed x,y,t output; every 8–10th frame also branches to a YOLO keyframe detector on a small crop (~15fps+), which re-anchors the tracker's position.)*

| Piece | Choice | Source / evidence |
|---|---|---|
| Starting dataset | public Roboflow tennis-ball sets | single-class sets already exist (one has 578 images, 5k+ downloads) — a tractable starting point, not a from-scratch data project |
| Fine-tuning data | your own rig's footage | public sets are shot from broadcast camera angles; your baseline-mounted view is different enough to need its own annotated frames too |
| Base model | YOLOv8n (or newer nano) | smallest Ultralytics variant — accuracy you can spend on a bigger model later, speed you can't get back |
| Export target | NCNN | fastest measured format for YOLO on Pi 5 CPU in Ultralytics' own benchmark |

*Training itself is close to a one-liner with Ultralytics' tooling (`yolo train model=yolov8n.pt data=tennis-ball.yaml epochs=100`), which is exactly why the tooling choice matters more than it might seem — the model architecture decision above is really a decision about which ecosystem gets you from "trained" to "running on the actual device" with the least custom engineering.*

---

## 16 · Talking between the two camera nodes
**Not MQTT for this part — that's the wrong tool here**

The payment layer's MQTT link (stage 11) is built for occasional, low-frequency control messages — "start," "stop" — and a broker in the middle adds a small amount of overhead that's irrelevant at that rate. The detection stream between the two camera nodes is the opposite: a steady 120–150 messages a second, each one tiny (an x/y pixel coordinate, a confidence value, a timestamp), where every extra millisecond of latency directly weakens stage 05's triangulation. Use a plain **UDP socket** with a small fixed-size packet instead — no broker, no connection handshake, minimal overhead, and losing an occasional packet (which UDP allows) is harmless here since the Kalman filter in stage 15 already expects to smooth over the odd missing detection.

Keep the two data paths conceptually separate even if they end up on the same device: a low-rate, reliable, broker-based channel for session control, and a high-rate, best-effort, direct channel for the real-time coordinate stream. Reaching for one general-purpose messaging system to do both jobs is a common source of exactly the kind of latency this design is trying to avoid.

---

## 17 · Trajectory fitting & the decision engine
**Off-the-shelf math, a hand-written rule**

The aggregator's job (stage 05) is arithmetic, not machine learning: triangulate each pair of simultaneous 2D detections into a 3D point using standard multi-view geometry (OpenCV's `cv2.triangulatePoints` does this directly), then fit a short physics curve — a parabola for the vertical axis is a fine starting approximation, refined later with a drag term if accuracy testing calls for it — through the last several points with `scipy.optimize.curve_fit`. That fit is what lets you find the bounce instant between two captured frames, not just at whichever frame happened to catch it.

The call itself should be the most boring code in the entire system: a point-in-polygon test (is the bounce coordinate inside the court's in-bounds region, using the calibrated line geometry from stage 05) with a tolerance margin sized to whatever your own accuracy testing measures — not a guess, and not something a model learns implicitly. Keeping this rule explicit and inspectable matters more here than almost anywhere else in the system: it's the one output a player can dispute, so it needs to be the one part you can always explain in one sentence.

---

## 18 · Choosing a database
**SQLite — and stop there for this scope**

For the IN/OUT software alone, on one device, with one writer and a handful of rows per rally, a database server is the wrong amount of machinery. **SQLite** is the right choice: it's a single file, ships built into Python's standard library, needs no server process running in the background on a resource-constrained Pi, and handles this write volume without breaking a sweat. Save a client-server database (Postgres, MySQL) for the day you're aggregating call data across many courts centrally — that's the business layer from the earlier sections, explicitly out of scope here.

| Table | Holds | Roughly |
|---|---|---|
| `calibration` | each node's solved camera pose & the court's line geometry | a few rows, rewritten on re-calibration |
| `points` | one row per rally: bounce x/y, call, confidence, clip file path | a few hundred rows per session |
| `sessions` | start/end time, node health at session start | one row per session |

Video clips themselves are just files on disk, referenced by path from the `points` table — never store the video blob inside the database itself, that defeats the point of using something this lightweight.

---

## 19 · Testing it without standing on a court
**Build the replay harness before you trust any tuning change**

The single most useful piece of infrastructure you can build early, and the one that's easiest to skip: a small script that feeds *recorded* video through the exact same detection code the live system uses, frame by frame, as if it were a live camera. Record a handful of reference clips once — clean serves, a ball near the line, a player briefly blocking the view, a shadow crossing the court — and every time you tune a threshold or touch the detection code, run it against that same fixed set and compare the calls to what you know actually happened. Without this, "does my change help or hurt" turns into standing on a real court re-testing by eye, which is slow, weather-dependent, and doesn't catch regressions in the cases you're not actively thinking about that day.

Validate against objective ground truth before trusting any call in a real session: known ball drops onto marked spots, chalk-marked bounce points, or a phone slow-motion video of the same point shot independently — the same validation approach used in the published smartphone-based tennis tracking research. Track your measured error in centimeters over time; that number is what should actually decide the tolerance margin in stage 17, not a guess.

---

## 20 · Build order for this software specifically
**Six milestones, each one testable on its own**

| Milestone | What you can prove at the end of it |
|---|---|
| 1 | A trained YOLO ball model, exported to NCNN, running against recorded clips through the replay harness — the ball is found reliably frame to frame, on the actual export format you'll deploy. |
| 2 | One camera, live, with the keyframe-detector-plus-tracker loop from stage 15 — smooth, stable positions in real time at full frame rate on the actual Pi 5, not just in a benchmark. |
| 3 | Two cameras, UDP link, clock-offset handshake — both nodes agree on "the same instant" to within a measured, known error. |
| 4 | Triangulation + trajectory fit producing a 3D bounce point from two synchronized 2D detections. |
| 5 | The line-call rule, SQLite logging, and a confidence score — a full point, end to end, produces a stored, explainable call. |
| 6 | Field validation against marked ground truth, with a measured error in centimeters — the number that finally sets the production tolerance margin. |

*Only after milestone 6 holds up under real outdoor light and real players does it make sense to connect this to the payment layer in stages 09–13 — wiring money to a call engine you haven't yet measured is the wrong order to build in.*

---

## Sources consulted for this design

1. [How Sony's Hawk-Eye electronic line-calling system works — CNBC](https://www.cnbc.com/2023/09/09/how-sonys-hawk-eye-works-at-the-us-open.html)
2. [Low-cost tennis line-call system with ball tracking — Stanford CS231A](https://web.stanford.edu/class/cs231a/prev_projects_2016/final_report_v2.pdf)
3. [Real Time Tennis Match Tracking with Low Cost Equipment — AAAI](https://cdn.aaai.org/ocs/17690/17690-77727-1-PB.pdf)
4. [TrackNet: A Deep Learning Network for Tracking High-speed and Tiny Objects in Sports Applications — arXiv](https://ar5iv.labs.arxiv.org/html/1907.03698)
5. [Anatomy of a Computer Vision App — Tennis Camera](https://medium.com/@tenniscamera/anatomy-of-a-computer-vision-app-5159d22e8c8c)
6. [Frame Accurate Video Synchronization Using Multi Raspberry Pi Camera Module](https://www.academia.edu/38499974/Frame_Accurate_Video_Synchronization_Using_Multi_Raspberry_Pi_Camera_Module_docx)
7. [A Novel Approach to Video Compression using Region of Interest (ROI) Method on Video Surveillance Systems](https://thesai.org/Downloads/Volume13No6/Paper_17-A_Novel_Approach_to_Video_Compression.pdf)
8. [Baseline Vision — portable net-post tennis line-calling camera (commercial product)](https://www.baselinevision.com/product)
9. [Baseline Vision review — setup, predictive calculations, specs](https://www.tennisleo.com/baseline-vision-review/)
10. [Tennis Ball Tracking: 3D Trajectory Estimation using Smartphone Videos — Stanford EE367](https://web.stanford.edu/class/ee367/Winter2018/fazio_fisher_fujinami_ee367_win18_report.pdf)
11. [Fully automatic algorithm for tennis court line detection (open-source reference implementation)](https://github.com/gchlebus/tennis-court-detection)
12. [Unattended payment terminals: here's what to know — Stripe](https://stripe.com/resources/more/unattended-payment-terminals)
13. [Apple Pay QR code setup guide via hosted checkout links — EZQR](https://ez-qr.com/blog/apple-pay-qr-code-complete-2026-guide)
14. [Remote command concepts for IoT devices (MQTT command/response pattern) — AWS IoT Core docs](https://docs.aws.amazon.com/iot/latest/developerguide/iot-remote-command-concepts.html)
15. [Building vball-net: a lightweight ball tracker, 200+ FPS on CPU (TrackNet-derived)](https://medium.com/@asigatchov/building-vball-net-a-lightweight-volleyball-ball-tracker-200-fps-on-cpu-20f5724c0c18)
16. [Testing object detection (YOLO, MobileNet, etc.) with picamera2 on Pi 5 — Jeff Geerling](https://www.jeffgeerling.com/blog/2024/testing-object-detection-yolo-mobilenet-etc-picamera2-on-pi-5/)
17. [Real-time Ball Detection and Tracking using Raspberry Pi (HSV color-threshold + contour method)](https://www.academia.edu/105636795/Real_time_Ball_Detection_and_Tracking_using_Raspberry_PI)
18. [ArtLabss/tennis-tracking — open-source monocular "HawkEye" for tennis (TrackNet + ResNet50 + court detection)](https://github.com/ArtLabss/tennis-tracking)
19. [nikhilgrad/Tennis-Ball-Tracker — custom YOLOv8 ball detector with interpolation](https://github.com/nikhilgrad/Tennis-Ball-Tracker)
20. [abhroroy365/Tennis-Tracker — YOLOv8 players, YOLOv5 ball, ResNet34 court keypoints](https://github.com/abhroroy365/Tennis-Tracker)
21. [Public tennis ball detection dataset (Roboflow Universe)](https://universe.roboflow.com/viren-dhanwani/tennis-ball-detection)
22. [YOLO on Raspberry Pi: setup & measured benchmarks — Ultralytics docs](https://docs.ultralytics.com/guides/raspberry-pi)
23. [WO2017008218A1 — Hawk-Eye single-camera identification method for tennis (patent)](https://patents.google.com/patent/WO2017008218A1/en)
24. [Machine vision cameras provide vision for the PlaySight tennis analysis system](https://www.vision-systems.com/cameras-accessories/article/16746305/machine-vision-cameras-provide-vision-for-tennis-analysis-system)
25. [Behind the Design: SwingVision — Apple Developer](https://developer.apple.com/news/?id=0pg4dthn)
26. [UPA partners with PlayReplay for electronic line-calling technology](https://www.playreplay.io/news/upa-partners-with-playreplay-for-electronic-line-calling-technology)
