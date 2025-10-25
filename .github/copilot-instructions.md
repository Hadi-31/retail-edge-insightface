## Retail Edge InsightFace — Copilot Instructions

This repository is a compact on-device retail ad engine (person detection → face attributes → persona fusion → rule-based ad selection → display). Below are focused, actionable notes to help an AI coding agent be productive immediately.

- Entry point: `main.py` orchestrates the pipeline. Key components are instantiated there:
  - `apps/person_detect.py` — PersonDetector (MemryX DFP adapter or CPU HOG fallback).
  - `apps/face_attrs.py` — FaceAttrs (uses DeepFace for age/gender/emotion on a head-ROI).
  - `core/tracking.py` — IOUTracker (assigns persistent numeric IDs).
  - `core/fusion.py` — fuse(frame, tracked, face_infos) → (context, enriched_persons).
  - `core/ad_engine.py` — AdEngine.choose(context) returns `(ad_path, reason)`.
  - `ui/display.py` — rendering, ad caching, optional face blurring.

- Runtime / dev workflow (how to run):
  - Create venv and install: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt` (also `run.sh` automates this).
  - Optional: pre-download InsightFace model to avoid first-run delay (see README snippet using `FaceAnalysis(name='buffalo_l')`).
  - Run demo: `python main.py --source 0` or `python main.py --source sample.mp4`.
  - MemryX acceleration: set `USE_MEMRYX=1` and provide DFP(s) under `models/` or via `MX_DFP_PERSON` env var. Example:
    `USE_MEMRYX=1 MX_DFP_PERSON=models/yolo_person.dfp python main.py --source 0`

- Important env vars and tunables (search for these in the code):
  - `USE_MEMRYX`, `MX_DFP_PERSON` — MemryX adapter toggle and DFP path (`apps/person_detect.py`).
  - `LITE_MODE`, `FRAME_SKIP` / `FRAME_SKIP` -> `FSKIP` in `main.py`, `PERSON_IN_SZ`, `FACE_DET_SIZE` — performance knobs.
  - `ROI_HEAD_RATIO`, `DF_DETECTOR`, `DF_ENFORCE` — used by `FaceAttrs` to control head-ROI and DeepFace backend.
  - `MIN_SCORE`, `NMS_IOU` — person detector thresholds.

- Codebase conventions and gotchas (concrete, discoverable patterns):
  - Alignment by index: many components expect aligned lists — e.g. `IOUTracker.update(dets)` produces `tracked` which is zipped with `face_attr.analyze(frame, boxes)`; `FaceAttrs.analyze` returns a list the same length as `person_boxes` (see `apps/face_attrs.py`). Maintain this contract when modifying flows.
  - AdEngine uses a rule YAML at `config/personas.yaml`. Rules are evaluated in order; `AdEngine` enforces cooldowns per `person.id` and a guardrail `ignore_children` (so children-only scenes return no ad). See `core/ad_engine.py` for exact match semantics.
  - MemryX adapter is defensive: missing SDK/DFP falls back to CPU path. You can reproduce failures by toggling `USE_MEMRYX`.
  - DeepFace return type: code handles both `dict` and `list[dict]` return shapes (see `apps/face_attrs.py`). Be careful when upgrading DeepFace versions; tests assume the existing handling.
  - Display caches ad images and is robust to headless environments (it swallows display errors; see `ui/display.py`).

- Integration points to review for changes:
  - `models/` — optional DFP files for accelerator paths. Keep DFP naming consistent with `MX_DFP_PERSON` if you change the adapter.
  - `ui/assets/ads/` — where ad images/videos live; `Display` resizes and overlays them.
  - `config/personas.yaml` — authoritative rule set; editing here changes behavior without touching code.

- Debugging tips and tests to try (practical steps):
  - Reproduce MemryX fallback: unset `USE_MEMRYX` or point `MX_DFP_PERSON` to a missing file to test CPU HOG path.
  - Verify tracker/ad cooldown: run demo, observe `AdEngine.mark()` behavior in `core/ad_engine.py`; check state keys in `AdEngine.state` while debugging.
  - Face analysis speed: to test head-ROI behavior change `ROI_HEAD_RATIO` and `DF_DETECTOR` in env.

- Files that contain the most useful examples/patterns (reference):
  - `main.py` — orchestration, env defaults (FSKIP, PERSON_IN_SZ, FACE_DET_SIZE).
  - `apps/person_detect.py` — MemryX adapter + CPU fallback and robust output parsing.
  - `apps/face_attrs.py` — ROI crop and DeepFace result handling.
  - `core/ad_engine.py`, `core/fusion.py`, `core/tracking.py`, `core/utils.py` — business logic and helper primitives.
  - `ui/display.py` — rendering and caching behavior.

- Safety / privacy notes (encoded in the code):
  - Faces/images are not persisted to disk by default; overlays and ephemeral counters are used. `--blur-faces` toggles blurring in `Display`.

If any part of the pipeline, env variable list, or config semantics are incomplete or you'd like more examples (unit-test stubs, suggested configuration variants, or step‑by‑step debugging scripts), tell me which area to expand and I'll iterate.
