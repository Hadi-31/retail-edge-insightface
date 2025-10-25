import argparse, os, cv2, yaml
from apps.person_detect import PersonDetector
from apps.face_attrs import FaceAttrs
from core.tracking import IOUTracker
from core.ad_engine import AdEngine
from core import utils as U
from core.fusion import fuse
from ui.display import Display
from core.heatmap_tracker import HeatmapTracker
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
    # مع DeepFace الحالية، لا نحتاج det_size هنا (FaceAttrs يستخدم ROI داخليًا)
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

    # Initialize heatmap tracker using the very first frame's shape
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
        # إعادة مؤشر الفيديو للبداية (لو كان ملف)
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
        # 2) Person detection (بدون أي تصغير هنا)
        # -----------------------------
        dets = person_det.infer(frame) or []
        dets = [d for d in dets if d.get('conf', 1.0) >= min_person_conf]

        # تتبّع
        tracked = tracker.update(dets)
        boxes = [t['box'] for t in tracked]

        # -----------------------------
        # 3) Heatmap update (قبل أي رسم)
        # -----------------------------
        if heatmap is not None:
            heatmap.update(frame, tracked)

        # -----------------------------
        # 4) Face analysis (ROI داخل FaceAttrs لسرعة أعلى)
        # -----------------------------
        face_infos = face_attr.analyze(frame, boxes)

        # تأكد المحاذاة بين tracked و face_infos
        if len(face_infos) != len(tracked):
            pad = len(tracked) - len(face_infos)
            face_infos += [{'has_face': False, 'age': None, 'gender': None,
                            'expression': None, 'face_box': None, 'is_child': False}] * max(0, pad)
            face_infos = face_infos[:len(tracked)]

        # دمج السياق
        ctx, enriched = fuse(frame, tracked, face_infos)

        # اختيار الإعلان
        ad_path, reason = ad_engine.choose(ctx)
        disp.set_ad(ad_path, reason)

        # رسم المعلومات
        for p in enriched:
            if LITE_MODE:
                label = f"ID {p['id']} | {p['clothing_style']}"
            else:
                label = f"ID {p['id']} | {p['gender'] or 'NA'} | {p['age'] or 'NA'} | {p['expression']} | {p['clothing_style']}"
            U.draw_box(frame, p['box'], (0, 255, 0), label=label)

        # UI render
        frame = disp.render(frame, enriched, face_infos, U)

        # Heatmap overlay
        if heatmap is not None:
            frame = heatmap.render(frame)

        # عرض
        try:
            cv2.imshow("Retail Edge", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break
        except Exception:
            pass

        frame_index += 1

    # حفظ تقرير الحرارة عند الخروج
    try:
        if heatmap is not None:
            heatmap.save_report()
    except Exception as e:
        print(f"[Heatmap] Failed to save report: {e}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
