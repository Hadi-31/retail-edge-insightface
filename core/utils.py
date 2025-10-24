import time
import cv2
import numpy as np

def now_s():
    return time.time()

def iou(boxA, boxB):
    # boxes: (x1,y1,x2,y2)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = areaA + areaB - inter + 1e-6
    return inter / union

def draw_box(img, box, color=(0,255,0), label=None):
    x1,y1,x2,y2 = [int(v) for v in box]
    cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
    if label:
        cv2.putText(img, label, (x1, max(0, y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

def blur_region(img, box, k=31):
    x1,y1,x2,y2 = [max(0,int(v)) for v in box]
    roi = img[y1:y2, x1:x2]
    if roi.size == 0: return
    roi = cv2.GaussianBlur(roi, (k,k), 0)
    img[y1:y2, x1:x2] = roi

def time_of_day_label():
    h = time.localtime().tm_hour
    if 6 <= h < 12: return "morning"
    if 12 <= h < 17: return "afternoon"
    if 17 <= h < 21: return "evening"
    return "night"

def clothing_style_from_colors(crop):
    # Lightweight heuristic: bright = sporty, darker + smooth = semi_formal/formal
    if crop is None or crop.size == 0:
        return "unknown"
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    v = hsv[:,:,2]
    mean_v = float(np.mean(v))
    sat = float(np.mean(hsv[:,:,1]))
    if mean_v > 170 and sat > 60:
        return "bright"
    if mean_v < 110:
        return "formal"
    return "casual"
