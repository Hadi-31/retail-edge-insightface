import argparse, os, cv2, yaml
from apps.person_detect import PersonDetector
from apps.face_attrs import FaceAttrs
from core.tracking import IOUTracker
from core.ad_engine import AdEngine
from core import utils as U
from core.fusion import fuse
from ui.display import Display
import numpy as np

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

    while True:
        ok, frame = cap.read()
        if not ok: break

        dets = person_det.infer(frame) or []
        dets = [d for d in dets if d.get('conf', 1.0) >= min_person_conf]
        tracked = tracker.update(dets)
        boxes = [t['box'] for t in tracked]

        # Face analysis
        face_infos = face_attr.analyze(frame, boxes)

        # Fuse context
        ctx, enriched = fuse(frame, tracked, face_infos)

        # Choose ad
        ad_path, reason = ad_engine.choose(ctx)
        disp.set_ad(ad_path, reason)

        # Render
        for p in enriched:
            label = f"ID {p['id']} | {p['gender'] or 'NA'} | {p['age'] or 'NA'} | {p['expression']} | {p['clothing_style']}"
            U.draw_box(frame, p['box'], (0,255,0), label=label)
        frame = disp.render(frame, enriched, face_infos, U)

        cv2.imshow("Retail Edge InsightFace", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
