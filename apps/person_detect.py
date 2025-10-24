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
      - NMS_IOU=0.50               : IOU for NMS (default 0.50)
      - PERSON_IN_SZ=640           : max input size before detection (downscale) for speed
    """

    def __init__(self):
        self.min_score = float(os.getenv("MIN_SCORE", "0.30"))
        self.nms_iou = float(os.getenv("NMS_IOU", "0.50"))
        self.person_in_sz = int(os.getenv("PERSON_IN_SZ", "640"))

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
    # MemryX path (with internal downscale + rescale)
    # -------------------------------------------------------------------------
    def _infer_memryx(self, frame):
        """
        Runs the DFP on the accelerator and decodes outputs.
        Supports several common output layouts to be robust to different YOLO DFP builds:
          A) dict with 'boxes' (N,4) and 'scores' (N,) in xyxy
          B) single tensor (N,6 or 7) as [x1,y1,x2,y2,score,(class)]
          C) two unnamed tensors where one is (N,4) and the other is (N,) or (N,1)
        Also performs optional downscale before inference (PERSON_IN_SZ), then rescales boxes back.
        """
        h0, w0 = frame.shape[:2]
        # Downscale for speed if needed
        if self.person_in_sz > 0:
            scale = self.person_in_sz / max(h0, w0)
        else:
            scale = 1.0
        if scale < 1.0:
            inp = cv2.resize(frame, (int(w0 * scale), int(h0 * scale)))
        else:
            inp = frame

        # 1) Run inference
        mximg = MXImage(inp)                 # MemryX helper wraps the numpy image
        outputs = self.mx.infer(mximg)       # returns dict[str, np.ndarray] or list-like

        # 2) Normalize to a dict for easier handling
        if isinstance(outputs, (list, tuple)):
            out_dict = {f"out{i}": arr for i, arr in enumerate(outputs)}
        elif isinstance(outputs, dict):
            out_dict = outputs
        else:
            # Unexpected; nothing we can do
            return []

        dets = []

        # 3) Try the easy case first: keys 'boxes' and 'scores'
        if "boxes" in out_dict and "scores" in out_dict:
            boxes = np.asarray(out_dict["boxes"])
            scores = np.asarray(out_dict["scores"]).reshape(-1)
            dets = self._pack_and_filter(boxes, scores, self.min_score)

        # 4) If there is a single tensor shaped (N,6 or 7) -> [x1,y1,x2,y2,score,(class)]
        elif len(out_dict) == 1:
            arr = next(iter(out_dict.values()))
            arr = np.asarray(arr)
            if arr.ndim == 2 and arr.shape[1] in (6, 7):
                boxes = arr[:, 0:4]
                scores = arr[:, 4]
                dets = self._pack_and_filter(boxes, scores, self.min_score)

        # 5) If there are exactly 2 arrays: try to guess which is boxes and which is scores
        elif len(out_dict) == 2:
            vals = list(out_dict.values())
            a, b = np.asarray(vals[0]), np.asarray(vals[1])
            # one of them (N,4), the other (N,) or (N,1)
            if a.ndim == 2 and a.shape[1] == 4 and b.ndim >= 1:
                boxes, scores = a, b.reshape(-1)
                dets = self._pack_and_filter(boxes, scores, self.min_score)
            elif b.ndim == 2 and b.shape[1] == 4 and a.ndim >= 1:
                boxes, scores = b, a.reshape(-1)
                dets = self._pack_and_filter(boxes, scores, self.min_score)

        # If nothing matched, dets remains empty
        if not dets:
            return []

        # 6) NMS
        dets = self._nms(dets, iou_thresh=self.nms_iou)

        # 7) Rescale boxes back to original frame size if we downscaled
        if scale < 1.0:
            inv = 1.0 / scale
            for d in dets:
                x1, y1, x2, y2 = d["box"]
                d["box"] = (x1 * inv, y1 * inv, x2 * inv, y2 * inv)

        return dets

    # -------------------------------------------------------------------------
    # CPU fallback (OpenCV HOG) with the same downscale trick
    # -------------------------------------------------------------------------
    def _infer_cpu(self, frame):
        h0, w0 = frame.shape[:2]
        if self.person_in_sz > 0:
            scale = self.person_in_sz / max(h0, w0)
        else:
            scale = 1.0
        if scale < 1.0:
            inp = cv2.resize(frame, (int(w0 * scale), int(h0 * scale)))
        else:
            inp = frame

        rects, weights = self.hog.detectMultiScale(
            inp, winStride=(8, 8), padding=(8, 8), scale=1.05
        )
        dets = []
        for (x, y, w, h), s in zip(rects, weights):
            x1, y1, x2, y2 = x, y, x + w, y + h
            if float(s) >= self.min_score:
                # rescale to original frame coordinates if downscaled
                if scale < 1.0:
                    inv = 1.0 / scale
                    x1, y1, x2, y2 = x1 * inv, y1 * inv, x2 * inv, y2 * inv
                dets.append({"box": (x1, y1, x2, y2), "conf": float(s)})
        return self._nms(dets, iou_thresh=self.nms_iou)

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
