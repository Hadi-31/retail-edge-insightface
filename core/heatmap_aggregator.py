import os, json, time
from glob import glob
from collections import defaultdict

def _load_json(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None

def combine_reports(out_dir="heatmap_reports", master_path=None):
    """
    يدمج جميع تقارير الكاميرات *_heatmap.json في ملف رئيسي واحد.
    البنية النهائية:
    {
      "generated_at": "...",
      "cameras": [ {...cam report...}, ... ],
      "aggregate": {
        "cell_size_px": min cell among cams,
        "zones": [{"zone":"(x,y)","visits":...,"hot_spots":...,"avg_dwell":...}, ...]
      }
    }
    """
    os.makedirs(out_dir, exist_ok=True)
    if master_path is None:
        master_path = os.path.join(out_dir, "master_heatmap.json")

    cam_files = sorted(glob(os.path.join(out_dir, "*_heatmap.json")))
    cams = []
    agg = defaultdict(lambda: {"visits": 0, "hot_spots": 0, "avg_dwell_sum": 0.0, "avg_dwell_w": 0})
    cell_min = None

    for fp in cam_files:
        data = _load_json(fp)
        if not data or "zones" not in data:
            continue
        cams.append(data)
        cell = data.get("cell_size_px")
        if isinstance(cell, int):
            cell_min = cell if cell_min is None else min(cell_min, cell)
        for z in data["zones"]:
            zone = z.get("zone")
            v = int(z.get("visits", 0))
            h = int(z.get("hot_spots", 0))
            d = float(z.get("avg_dwell", 0.0))
            agg[zone]["visits"] += v
            agg[zone]["hot_spots"] += h
            agg[zone]["avg_dwell_sum"] += d * max(v, 1)  # وزن بالزيارات
            agg[zone]["avg_dwell_w"] += max(v, 1)

    aggregate_zones = []
    for zone, met in sorted(agg.items()):
        w = max(met["avg_dwell_w"], 1)
        aggregate_zones.append({
            "zone": zone,
            "visits": met["visits"],
            "hot_spots": met["hot_spots"],
            "avg_dwell": round(met["avg_dwell_sum"] / w, 2)
        })

    master = {
        "generated_at": time.ctime(),
        "cameras": cams,
        "aggregate": {
            "cell_size_px": cell_min,
            "zones": aggregate_zones
        }
    }

    with open(master_path, "w") as f:
        json.dump(master, f, indent=2)

    print(f"[HeatmapAggregator] Master heatmap saved: {master_path}")
    return master_path
