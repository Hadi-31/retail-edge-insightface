import os
import cv2
import numpy as np

# --- Optional: MemryX imports (only when USE_MEMRYX=1) -----------------------
MX_ENABLED = os.getenv("USE_MEMRYX", "0") == "1"
if MX_ENABLED:
    try:
        # MemryX SDK (Python) – matches the examples shipped with the SDK
        from memryx import AsyncAccl, MXImage
    except Exception as e:
        print("[PersonDetector] MemryX SDK not available in Python:", e)
        MX_ENABLED = False


class PersonDetector:
    """
    Person detector with two backends:

      1) MemryX (YOLO .dfp) via AsyncAccl  -> enabled when USE_MEMRYX=1 and DFP is present
      2) CPU fallback using OpenCV HOG     -> runs anywhere so you can still demo without hardware

    Env vars (optional):
      - USE_MEMRYX=1               : enable MemryX backend
      - MX_DFP_PERSON=<path.dfp>   : path to your YOLO .dfp (default: models/yolo_person.dfp)
      - MIN_SCORE=0.30             : confidence threshold (default 0.30)
      - NMS_IOU=0.50               : IOU for NMS on MemryX path (default 0.50)
    """

    def __init__(self):
        self.min_score = float(os.getenv("MIN_SCORE", "0.30"))
        self.nms_iou = float(os.getenv("NMS_IOU", "0.50"))

        self.use_memryx = MX_ENABLED
        self.mx = None
        self.hog = None

        if self.use_memryx:
            dfp_path = os.getenv("MX_DFP_PERSON", "models/yolo_person.dfp")
            if not os.path.exists(dfp_path):
                print(f"[PersonDetector] DFP not found at {dfp_path}. Falling back to CPU.")
                self.use_memryx = False
            else:
                print(f"[PersonDetector] Using MemryX backend with DFP: {dfp_path}")
                self.mx = AsyncAccl(dfp_path)  # loads the DFP on the accelerator

        if not self.use_memryx:
            # CPU fallback (HOG people detector)
            print("[PersonDetector] Using CPU fallback (OpenCV HOG).")
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    def infer(self, frame):
        """
        Returns: list of dicts
            [{'box': (x1,y1,x2,y2), 'conf': float}, ...] in frame coordinates
        """
        if self.use_memryx and self.mx is not None:
            return self._infer_memryx(frame)
        else:
            return self._infer_cpu(frame)

    # -------------------------------------------------------------------------
    # MemryX path
    # -------------------------------------------------------------------------
    def _infer_memryx(self, frame):
        """
        Runs the DFP on the accelerator and decodes outputs.
        We support several common output layouts to make it robust to different YOLO DFP builds:
          A) dict with 'boxes' (N,4) and 'scores' (N,) in xyxy
          B) single tensor (N,6) as [x1,y1,x2,y2,score,class]
          C) two unnamed tensors where one is (N,4) and the other is (N,) or (N,1)
        """
        # 1) Run inference
        mximg = MXImage(frame)               # MemryX helper wraps the numpy image
        outputs = self.mx.infer(mximg)       # returns dict[str, np.ndarray] or list-like

        # 2) Normalize to a dict for easier handling
        if isinstance(outputs, (list, tuple)):
            out_dict = {f"out{i}": arr for i, arr in enumerate(outputs)}
        elif isinstance(outputs, dict):
            out_dict = outputs
        else:
            # Unexpected; nothing we can do
            return []

        # 3) Try the easy case first: keys 'boxes' and 'scores'
        if "boxes" in out_dict and "scores" in out_dict:
            boxes = np.asarray(out_dict["boxes"])
            scores = np.asarray(out_dict["scores"]).reshape(-1)
            dets = self._pack_and_filter(boxes, scores, self.min_score)
            return self._nms(dets, iou_thresh=self.nms_iou)

        # 4) If there is a single tensor shaped (N,6 or 7) -> [x1,y1,x2,y2,score,(class)]
        if len(out_dict) == 1:
            arr = next(iter(out_dict.values()))
            arr = np.asarray(arr)
            if arr.ndim == 2 and arr.shape[1] in (6, 7):
                boxes = arr[:, 0:4]
                scores = arr[:, 4]
                dets = self._pack_and_filter(boxes, scores, self.min_score)
                return self._nms(dets, iou_thresh=self.nms_iou)

        # 5) If there are exactly 2 arrays: try to guess which is boxes and which is scores
        if len(out_dict) == 2:
            vals = list(out_dict.values())
            a, b = np.asarray(vals[0]), np.asarray(vals[1])
            # one of them (N,4), the other (N,) or (N,1)
            if a.ndim == 2 and a.shape[1] == 4 and b.ndim >= 1:
                boxes, scores = a, b.reshape(-1)
            elif b.ndim == 2 and b.shape[1] == 4 and a.ndim >= 1:
                boxes, scores = b, a.reshape(-1)
            else:
                boxes, scores = None, None
            if boxes is not None and scores is not None:
                dets = self._pack_and_filter(boxes, scores, self.min_score)
                return self._nms(dets, iou_thresh=self.nms_iou)

        # 6) Could not auto-decode: return empty (or add custom decoder here)
        #    If your DFP outputs a different layout, map it here and return the same format.
        #    Example: convert center-based boxes to xyxy, etc.
        return []

    # -------------------------------------------------------------------------
    # CPU fallback (OpenCV HOG)
    # -------------------------------------------------------------------------
    def _infer_cpu(self, frame):
        rects, weights = self.hog.detectMultiScale(
            frame, winStride=(8, 8), padding=(8, 8), scale=1.05
        )
        dets = []
        for (x, y, w, h), s in zip(rects, weights):
            x1, y1, x2, y2 = x, y, x + w, y + h
            if float(s) >= self.min_score:
                dets.append({"box": (x1, y1, x2, y2), "conf": float(s)})
        return dets

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def _pack_and_filter(boxes, scores, min_score):
        boxes = np.asarray(boxes).reshape(-1, 4)
        scores = np.asarray(scores).reshape(-1)
        dets = []
        for b, s in zip(boxes, scores):
            if float(s) < min_score:
                continue
            x1, y1, x2, y2 = [float(v) for v in b]
            dets.append({"box": (x1, y1, x2, y2), "conf": float(s)})
        return dets

    @staticmethod
    def _nms(dets, iou_thresh=0.5):
        """
        Plain NMS on xyxy boxes.
        dets: [{'box':(x1,y1,x2,y2), 'conf':score}, ...]
        """
        if not dets:
            return dets
        boxes = np.array([d["box"] for d in dets], dtype=np.float32)
        scores = np.array([d["conf"] for d in dets], dtype=np.float32)

        x1, y1, x2, y2 = boxes.T
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

            inds = np.where(iou <= iou_thresh)[0]
            order = order[inds + 1]

        return [dets[i] for i in keep]
