import cv2, numpy as np, time, json, os
from collections import defaultdict

class HeatmapTracker:
    """
    Heatmap tracker:
      - يسجل زمن الوقوف عند كل موقع
      - يرسم خريطة حرارة (heatmap overlay)
      - يولد تقرير JSON بالمناطق الساخنة لكل كاميرا
    """

    def __init__(self, frame_shape, cam_id="cam1", dwell_thresh=5.0, hot_thresh=10.0, out_dir="heatmap_reports"):
        h, w = frame_shape[:2]
        self.cam_id = cam_id
        self.heatmap = np.zeros((h, w), dtype=np.float32)
        self.last_pos = {}
        self.stay_start = {}
        self.dwell_thresh = dwell_thresh
        self.hot_thresh = hot_thresh
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        # grid بحجم (50px) لتجميع الإحصاءات
        self.cell = 50
        self.stats = defaultdict(lambda: {"visits": 0, "hot": 0, "avg_dwell": 0.0, "visit_weight": 0})

    def update(self, frame, tracked):
        now = time.time()
        for p in tracked:
            pid = p['id']
            x1, y1, x2, y2 = map(int, p['box'])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            if pid in self.last_pos:
                lx, ly = self.last_pos[pid]
                dist = np.hypot(cx - lx, cy - ly)
            else:
                dist = 0

            self.last_pos[pid] = (cx, cy)

            # إذا ما تحرك كثير، نحسب مدة الوقوف
            if dist < 10:
                if pid not in self.stay_start:
                    self.stay_start[pid] = now
                dwell_time = now - self.stay_start[pid]
                if dwell_time > self.dwell_thresh:
                    cv2.circle(self.heatmap, (cx, cy), 20, 1.0, -1)
                    key = (cx // self.cell, cy // self.cell)
                    self.stats[key]["visits"] += 1
                    # متوسط مرجّح حسب عدد الزيارات
                    wprev = self.stats[key]["visit_weight"]
                    avg_prev = self.stats[key]["avg_dwell"]
                    wnew = wprev + 1
                    self.stats[key]["avg_dwell"] = (avg_prev * wprev + dwell_time) / max(wnew, 1)
                    self.stats[key]["visit_weight"] = wnew
                    if dwell_time > self.hot_thresh:
                        cv2.circle(self.heatmap, (cx, cy), 30, 2.0, -1)
                        self.stats[key]["hot"] += 1
            else:
                self.stay_start.pop(pid, None)

        # تبريد تدريجي (نخلي الألوان تخف ببطء)
        self.heatmap *= 0.98

    def render(self, frame):
        heat_norm = cv2.normalize(self.heatmap, None, 0, 255, cv2.NORM_MINMAX)
        heat_color = cv2.applyColorMap(heat_norm.astype(np.uint8), cv2.COLORMAP_JET)
        blended = cv2.addWeighted(frame, 0.7, heat_color, 0.3, 0)
        return blended

    def save_report(self):
        report_path = os.path.join(self.out_dir, f"{self.cam_id}_heatmap.json")
        summary = {
            "camera": self.cam_id,
            "generated_at": time.ctime(),
            "cell_size_px": self.cell,
            "zones": [
                {
                    "zone": f"({k[0]},{k[1]})",
                    "visits": v["visits"],
                    "hot_spots": v["hot"],
                    "avg_dwell": round(float(v["avg_dwell"]), 2)
                }
                for k, v in sorted(self.stats.items())
            ],
        }
        with open(report_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[HeatmapTracker] Saved heatmap report: {report_path}")
        return report_path
