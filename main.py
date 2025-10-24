import argparse, os, cv2, yaml
from apps.person_detect import PersonDetector
from apps.face_attrs import FaceAttrs
from core.tracking import IOUTracker
from core.ad_engine import AdEngine
from core import utils as U
from core.fusion import fuse
from ui.display import Display
import numpy as np

# -----------------------------
# Performance / Lite mode flags
# -----------------------------
FSKIP = int(os.getenv("FRAME_SKIP", "0"))            # عدد الإطارات التي تُتخطّى بين كل معالجة
LITE_MODE = os.getenv("LITE_MODE", "0") == "1"       # وضع خفيف (اختياري)
# ملاحظة: PERSON_IN_SZ يُستخدم داخل PersonDetector فقط إذا رغبت، لا نستخدمه هنا الآن

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
    # مع نسخة DeepFace الحالية، لا نحتاج det_size هنا
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

    frame_index = 0

    while True:
        ok, frame = cap.read()
        if not ok:
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
        #    التصغير إن لزم يتم داخل PersonDetector فقط
        # -----------------------------
        dets = person_det.infer(frame) or []

        # فلترة حسب الثقة
        dets = [d for d in dets if d.get('conf', 1.0) >= min_person_conf]

        # تتبّع
        tracked = tracker.update(dets)
        boxes = [t['box'] for t in tracked]

        # -----------------------------
        # 3) Face analysis (ROI داخل FaceAttrs لسرعة أعلى)
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

        frame = disp.render(frame, enriched, face_infos, U)

        # عرض
        try:
            cv2.imshow("Retail Edge", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break
        except Exception:
            pass

        frame_index += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
