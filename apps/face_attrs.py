import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis

class FaceAttrs:
    def __init__(self, det_size=None):
        """
        Lite/Performance controls via env:
          - LITE_MODE=1          -> use 'buffalo_s' (أخف) بدل 'buffalo_l'
          - FACE_DET_SIZE=320    -> تصغير det_size (أصغر = أسرع)
          - ROI_HEAD_RATIO=0.6   -> نسبة ارتفاع منطقة الرأس داخل صندوق الشخص
          - MIN_FACE_IOU=0.05    -> حد أدنى لتقاطع الوجه مع صندوق الشخص (إذا استخدمنا fallback كامل الإطار)
        """
        lite = os.getenv("LITE_MODE", "0") == "1"
        model_name = "buffalo_s" if lite else "buffalo_l"
        dsz_env = int(os.getenv("FACE_DET_SIZE", "320" if lite else "640"))

        if det_size is None:
            det_size = (dsz_env, dsz_env)

        self.app = FaceAnalysis(name=model_name)
        # ctx_id=0: استخدم CPU/GPU الافتراضي المتوفر
        self.app.prepare(ctx_id=0, det_size=det_size)

        # ROI = الجزء العلوي من صندوق الشخص (رأس/كتفين)
        self.roi_head_ratio = float(os.getenv("ROI_HEAD_RATIO", "0.6"))
        self.min_face_iou = float(os.getenv("MIN_FACE_IOU", "0.05"))

    def analyze(self, frame, person_boxes):
        """
        لكل صندوق شخص، نحاول إيجاد وجه داخل منطقة الرأس فقط لرفع السرعة.
        نرجع قائمة بنفس طول person_boxes:
          {'has_face':bool, 'age':int|None, 'gender':str|None,
           'expression':str, 'face_box':(x1,y1,x2,y2)|None, 'is_child':bool}
        """
        results = []
        h, w = frame.shape[:2]

        # دالة IOU داخلية
        def iou(a, b):
            xA = max(a[0], b[0]); yA = max(a[1], b[1])
            xB = min(a[2], b[2]); yB = min(a[3], b[3])
            inter = max(0, xB - xA) * max(0, yB - yA)
            areaA = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
            areaB = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
            return inter / (areaA + areaB - inter + 1e-6)

        for pb in person_boxes:
            x1, y1, x2, y2 = map(int, pb)
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(w - 1, x2); y2 = min(h - 1, y2)
            if x2 <= x1 or y2 <= y1:
                results.append(self._empty_face())
                continue

            # قص منطقة الرأس/الكتفين فقط لتسريع الكشف
            y_mid = y1 + int(self.roi_head_ratio * (y2 - y1))
            y_mid = min(y_mid, y2)
            head_roi = frame[y1:y_mid, x1:x2]
            if head_roi.size == 0:
                results.append(self._empty_face())
                continue

            # كشف الوجوه داخل الـROI
            faces = self.app.get(head_roi)

            # لو ما وجد وجه داخل ROI، بإمكاننا تجربة الإطار الكامل كـ fallback (اختياري):
            # faces_full = self.app.get(frame)  # لو حبيت تفعل fallback كامل الإطار
            # ونختار أقرب وجه للصندوق. افتراضياً نكتفي بالـROI لتقليل الحمل.
            if len(faces) == 0:
                results.append(self._empty_face())
                continue

            # اختر أقوى وجه (أكبر مساحة داخل الـROI)
            def area_of_bbox(f):
                b = f.bbox.astype(int)
                return max(0, (b[2]-b[0])) * max(0, (b[3]-b[1]))
            fbest = max(faces, key=area_of_bbox)

            # صندوق الوجه ضمن ROI، نحوله لإحداثيات الإطار الأصلي
            fbox = fbest.bbox.astype(int)  # (fx1, fy1, fx2, fy2) بالنسبة للـROI
            fx1, fy1, fx2, fy2 = fbox
            fx1 += x1; fx2 += x1
            fy1 += y1; fy2 += y1
            face_box = (int(fx1), int(fy1), int(fx2), int(fy2))

            # (اختياري) تحقق من تقاطع الوجه مع صندوق الشخص (لو فعّلت fallback لاحقًا)
            if self.min_face_iou > 0:
                if iou((x1, y1, x2, y2), face_box) < self.min_face_iou:
                    results.append(self._empty_face())
                    continue

            # سمات الوجه (قد لا تكون كلها متوفرة حسب الموديل)
            age = getattr(fbest, "age", None)
            gender_idx = getattr(fbest, "gender", None)
            gender = "male" if gender_idx == 0 else ("female" if gender_idx == 1 else None)

            # InsightFace لا يُخرج تعبير بشكل مباشر هنا → نستخدم "neutral"
            expression = "neutral"
            is_child = (age is not None and age < 13)

            results.append({
                "has_face": True,
                "age": age,
                "gender": gender,
                "expression": expression,
                "face_box": face_box,
                "is_child": is_child
            })

        return results

    @staticmethod
    def _empty_face():
        return {
            "has_face": False,
            "age": None,
            "gender": None,
            "expression": None,
            "face_box": None,
            "is_child": False
        }
