import argparse, os, cv2, yaml
from apps.person_detect import PersonDetector
from apps.face_attrs import FaceAttrs
from core.tracking import IOUTracker
from core.ad_engine import AdEngine
from core import utils as U
from core.fusion import fuse
from ui.display import Display
from core.heatmap_tracker import HeatmapTracker
from core.heatmap_aggregator import combine_reports
import numpy as np

# -----------------------------
# Performance / Lite mode flags
# -----------------------------
FSKIP = int(os.getenv("FRAME_SKIP", "0"))          # عدد الإطارات التي تُتخطّى بين كل معالجة
LITE_MODE = os.getenv("LITE_MODE", "0") == "1"     # وضع خفيف (اختياري)

# -----------------------------
# Heatmap controls (env vars)
# -----------------------------
DWELL_THRESH = float(os.getenv("DWELL_THRESH", "5"))     # يحسب زيارة بعد 5 ثواني وقوف
HOT_THRESH   = float(os.getenv("HOT_THRESH", "10"))      # نقطة ساخنة بعد 10 ثواني
HEAT_OUT_DIR = os.getenv("HEAT_OUT_DIR", "heatmap_reports")
MASTER_HEAT_FILE = os.getenv("MASTER_HEAT_FILE", None)   # المسار النهائي للتقرير المركزي

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=str, default="0", help="webcam index or video path (default 0)")
    ap.add_argument("--config", type=str, default="config/personas.yaml")
    ap.add_argument("--blur-faces", action="store_true")
    ap.add_argument("--min-person-conf", type=float, default=None)
    return ap.parse_args()

def main():
    args = parse_args()
    src = int(args.source) if args.source.isdigit() else args.source

    # Components
    person_det = PersonDetector()
    # DeepFace version – FaceAttrs uses internal ROI optimization
    face_attr = FaceAttrs()
    tracker = IOUTracker(iou_thresh=0.4, max_age=30)
    ad_engine = AdEngine(args.config)
    disp = Display(blur_faces=args.blur_faces)

    # Load config for thresholds
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    min_person_conf = args.min_person_conf or cfg['global']['min_person_conf']
    min_face_conf = cfg['global']['min_face_conf']

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"Failed to open source: {args.source}")
        return

    # Initialize heatmap tracker using the first frame's shape
    heatmap = None
    ret, test_frame = cap.read()
    if ret and test_frame is not None:
        heatmap = HeatmapTracker(
            test_frame.shape,
            cam_id=str(args.source),
            dwell_thresh=DWELL_THRESH,
            hot_thresh=HOT_THRESH,
            out_dir=HEAT_OUT_DIR
        )
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        except Exception:
            pass
    else:
        print("[Heatmap] Warning: could not initialize heatmap (no test frame).")

    frame_index = 0

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        # -----------------------------
        # 1) Frame skipping (لتخفيف الحمل)
        # -----------------------------
        if FSKIP and (frame_index % (FSKIP + 1) != 0):
            try:
                cv2.imshow("Retail Edge", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('q'):
                    break
            except Exception:
                pass
            frame_index += 1
            continue

        # -----------------------------
        # 2) Person detection
        # -----------------------------
        dets = person_det.infer(frame) or []
        dets = [d for d in dets if d.get('conf', 1.0) >= min_person_conf]

        # Tracking
        tracked = tracker.update(dets)
        boxes = [t['box'] for t in tracked]

        # -----------------------------
        # 3) Heatmap update
        # -----------------------------
        if heatmap is not None:
            heatmap.update(frame, tracked)

        # -----------------------------
        # 4) Face analysis
        # -----------------------------
        face_infos = face_attr.analyze(frame, boxes)

        # Ensure alignment between tracking and faces
        if len(face_infos) != len(tracked):
            pad = len(tracked) - len(face_infos)
            face_infos += [{'has_face': False, 'age': None, 'gender': None,
                            'expression': None, 'face_box': None, 'is_child': False}] * max(0, pad)
            face_infos = face_infos[:len(tracked)]

        # Fuse all attributes
        ctx, enriched = fuse(frame, tracked, face_infos)

        # Choose ad
        ad_path, reason = ad_engine.choose(ctx)
        disp.set_ad(ad_path, reason)

        # Draw labels
        for p in enriched:
            if LITE_MODE:
                label = f"ID {p['id']} | {p['clothing_style']}"
            else:
                label = f"ID {p['id']} | {p['gender'] or 'NA'} | {p['age'] or 'NA'} | {p['expression']} | {p['clothing_style']}"
            U.draw_box(frame, p['box'], (0, 255, 0), label=label)

        # Render UI and heatmap overlay
        frame = disp.render(frame, enriched, face_infos, U)
        if heatmap is not None:
            frame = heatmap.render(frame)

        # Display
        try:
            cv2.imshow("Retail Edge", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break
        except Exception:
            pass

        frame_index += 1

    # -----------------------------
    # Save heatmap + aggregate master report
    # -----------------------------
    try:
        if heatmap is not None:
            report_path = heatmap.save_report()
            combine_reports(HEAT_OUT_DIR, MASTER_HEAT_FILE)
    except Exception as e:
        print(f"[Heatmap] Failed to save/aggregate report: {e}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
