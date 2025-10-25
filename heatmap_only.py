# heatmap_only.py
import os, cv2, argparse, time, numpy as np
from apps.person_detect import PersonDetector
from core.tracking import IOUTracker
from core.heatmap_tracker import HeatmapTracker

# --------- Env controls ----------
FSKIP         = int(os.getenv("FRAME_SKIP", "0"))       # skip N frames between processing
MIN_SCORE     = float(os.getenv("MIN_SCORE", "0.30"))   # person conf threshold
DWELL_THRESH  = float(os.getenv("DWELL_THRESH", "5"))   # secs → count a visit
HOT_THRESH    = float(os.getenv("HOT_THRESH", "10"))    # secs → mark as hotspot
HEAT_OUT_DIR  = os.getenv("HEAT_OUT_DIR", "heatmap_reports")
DRAW_BOXES    = os.getenv("DRAW_BOXES", "1") == "1"     # outline tracked persons

def parse_args():
    ap = argparse.ArgumentParser("Heatmap-only demo (person tracking + heatmap + reports)")
    ap.add_argument("--source", type=str, default="0", help="webcam index or video path (default 0)")
    ap.add_argument("--no-display", action="store_true", help="run headless (no window), just save report/image")
    return ap.parse_args()

def save_heatmap_png(hmap_tracker, out_dir, cam_id):
    os.makedirs(out_dir, exist_ok=True)
    hmap = hmap_tracker.heatmap
    heat_norm = cv2.normalize(hmap, None, 0, 255, cv2.NORM_MINMAX)
    heat_color = cv2.applyColorMap(heat_norm.astype(np.uint8), cv2.COLORMAP_JET)
    out_path = os.path.join(out_dir, f"{cam_id}_heatmap.png")
    cv2.imwrite(out_path, heat_color)
    print(f"[HeatmapOnly] Saved heatmap image: {out_path}")

def main():
    args = parse_args()
    src = int(args.source) if args.source.isdigit() else args.source
    cam_id = str(args.source)

    # Components
    person_det = PersonDetector()                 # uses MemryX if USE_MEMRYX=1 & DFP provided
    tracker    = IOUTracker(iou_thresh=0.4, max_age=30)

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[HeatmapOnly] Failed to open source: {args.source}")
        return

    # Initialize heatmap with first frame geometry
    heatmap = None
    ret, test_frame = cap.read()
    if not ret or test_frame is None:
        print("[HeatmapOnly] No first frame — aborting.")
        return
    heatmap = HeatmapTracker(
        test_frame.shape,
        cam_id=cam_id,
        dwell_thresh=DWELL_THRESH,
        hot_thresh=HOT_THRESH,
        out_dir=HEAT_OUT_DIR
    )
    # rewind if file source
    try: cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    except: pass

    frame_index = 0
    win_name = f"HeatmapOnly [{cam_id}]"

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        # Frame skip
        if FSKIP and (frame_index % (FSKIP + 1) != 0):
            if not args.no_display:
                try:
                    cv2.imshow(win_name, frame)
                    if (cv2.waitKey(1) & 0xFF) in (27, ord('q')): break
                except: pass
            frame_index += 1
            continue

        # Person detection
        dets = person_det.infer(frame) or []
        dets = [d for d in dets if d.get('conf', 1.0) >= MIN_SCORE]

        # Tracking
        tracked = tracker.update(dets)

        # Heatmap update
        heatmap.update(frame, tracked)

        # Optional: draw boxes
        if DRAW_BOXES:
            for t in tracked:
                x1,y1,x2,y2 = map(int, t['box'])
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(frame, f"ID {t['id']}", (x1, max(0, y1-6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

        # Overlay current heatmap
        overlay = heatmap.render(frame)

        # Show
        if not args.no_display:
            try:
                cv2.imshow(win_name, overlay)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q')):
                    break
            except:
                pass

        frame_index += 1

    # Save reports / image
    try:
        report_path = heatmap.save_report()
        save_heatmap_png(heatmap, HEAT_OUT_DIR, cam_id)
    except Exception as e:
        print(f"[HeatmapOnly] Failed to save outputs: {e}")

    cap.release()
    if not args.no_display:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
