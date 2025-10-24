# Retail Edge InsightFace (Hackathon-Ready)

An end-to-end **Edge AI retail ad engine** that detects people, estimates **age / gender / expression** (via **InsightFace**),
and selects a **contextual ad** to display — all **on-device**. Includes a simple rules engine, privacy guardrails,
and optional hooks for **MemryX** acceleration (use DFP if available).

## Quick Start

```bash
# 1) Python >= 3.9 recommended
python -m venv .venv && source .venv/bin/activate

# 2) Install deps
pip install -r requirements.txt

# 3) (Optional) Pre-download InsightFace model weights to avoid first-run delay:
# python -c "from insightface.app import FaceAnalysis; a=FaceAnalysis(name='buffalo_l'); a.prepare(ctx_id=0, det_size=(640,640))"

# 4) Run webcam demo
python main.py --source 0
# or run a test video
python main.py --source sample.mp4
```

## Optional: Enable MemryX Acceleration
If you have MemryX SDK + DFP(s):
1. Put your DFP(s) under `models/` (e.g., `models/yolo_person.dfp`, `models/scrfd_face.dfp`, `models/face_attrs.dfp`).
2. Set the environment variable `USE_MEMRYX=1`.
3. Run: `USE_MEMRYX=1 python main.py --source 0`

The code will try to import MemryX Python APIs and use them if available. Otherwise it falls back to a CPU-friendly path
(OpenCV HOG-based person detector). This means **the app still runs** even without MX hardware.

> Note: Exact MemryX APIs can vary by SDK version. The adapter in `apps/person_detect.py` shows how to plug in your
> AsyncAccl/DFP calls. Replace the TODOs with the exact calls from your SDK/example if needed.

## Privacy / Ethics
- No faces or images are saved; only ephemeral counts/labels are kept in memory.
- You can enable face blurring overlay via `--blur-faces`.
- Children targeting is disabled by default (see `config/personas.yaml: guardrails.ignore_children=true`).

## Project Layout
```
retail-edge-insightface/
  apps/
    person_detect.py        # Person detector: MemryX adapter OR CPU fallback (HOG)
    face_attrs.py           # InsightFace SCRFD + attributes (age/gender/expression)
  core/
    fusion.py               # Merges attributes + context (time/people_count) → persona
    ad_engine.py            # Rule-based engine using config/personas.yaml
    tracking.py             # Lightweight IOU tracker
    utils.py                # Helpers (timers, drawing)
  ui/
    display.py              # Video overlay + ad display window
    assets/ads/             # Put your ad images/videos here
  config/
    personas.yaml           # Rules/thresholds (edit to tune behavior)
  models/                   # (Optional) MemryX DFP files
  main.py                   # Orchestrates the pipeline
  run.sh                    # Install & run helper
  requirements.txt
  README.md
```

## Tips for the Demo
- Put 2–4 ad media files in `ui/assets/ads/` (e.g., `coffee_promo.jpg`, `family_combo.jpg`, `sports_snacks.jpg`, `takeaway_deal.jpg`).
- Use good lighting and place the camera around chest height for better face detection.
- Try walking groups vs. solo, change expressions, and outfits (hoodie/sporty vs. formal) to trigger different ads.
