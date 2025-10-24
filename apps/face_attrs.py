import os
import cv2
import numpy as np
from deepface import DeepFace

class FaceAttrs:
    def __init__(self, det_size=None):
        """
        Lite/Performance via env:
          - LITE_MODE=1             -> يخفّض الحمل (نستخدم head ROI فقط)
          - ROI_HEAD_RATIO=0.6      -> نسبة ارتفاع منطقة الرأس من صندوق الشخص
          - DF_DETECTOR=opencv      -> كاشف الوجوه (opencv | retinaface | mtcnn | yolov8 | ssd)
          - DF_ENFORCE=0/1          -> enforce_detection (0 أسرع)
        ملاحظة: det_size غير مستخدم هنا (DeepFace لا يحتاجه).
        """
        self.lite = os.getenv("LITE_MODE", "0") == "1"
        self.roi_head_ratio = float(os.getenv("ROI_HEAD_RATIO", "0.6"))
        self.detector_backend = os.getenv("DF_DETECTOR", "opencv").lower()
        self.enforce = bool(int(os.getenv("DF_ENFORCE", "0")))

        # حدد الأفعال المطلوبة: عمر + جنس + انفعالات
        self.actions = ["age", "gender", "emotion"]

    def analyze(self, frame, person_boxes):
        """
        لكل شخص: نحلل head-ROI بسرعة باستخدام DeepFace.analyze
        نُرجع قائمة بنفس طول person_boxes:
        {'has_face':bool, 'age':int|None, 'gender':str|None, 'expression':str,
         'face_box':(x1,y1,x2,y2)|None, 'is_child':bool}
        """
        results = []
        h, w = frame.shape[:2]

        for pb in person_boxes:
            x1, y1, x2, y2 = map(int, pb)
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(w - 1, x2); y2 = min(h - 1, y2)
            if x2 <= x1 or y2 <= y1:
                results.append(self._empty_face())
                continue

            # ROI: الجزء العلوي (رأس/كتفين) لتسريع الكشف
            y_mid = y1 + int(self.roi_head_ratio * (y2 - y1))
            y_mid = min(y_mid, y2)
            head = frame[y1:y_mid, x1:x2]
            if head.size == 0:
                results.append(self._empty_face())
                continue

            try:
                # DeepFace.analyze يقبل numpy array مباشرة
                out = DeepFace.analyze(
                    img_path = head,
                    actions = self.actions,
                    detector_backend = self.detector_backend,
                    enforce_detection = self.enforce,
                    prog_bar = False
                )

                # DeepFace قد يعيد dict أو list[dict] حسب النسخة
                if isinstance(out, list):
                    out = out[0]

                age = int(out.get("age")) if out.get("age") is not None else None
                gender = out.get("dominant_gender")
                emotion = out.get("dominant_emotion")

                # مواءمة التعبير لسياستك (happy/neutral/tired)
                expression = self._map_emotion(emotion)
                is_child = (age is not None and age < 13)

                # ملاحظة: DeepFace ما يعطينا bbox للوجه هنا بسهولة مع بعض الخلفيات،
                # لذلك نرجّع None لصندوق الوجه (الـUI تظل شغالة بدونها).
                face_box = None

                results.append({
                    "has_face": True,
                    "age": age,
                    "gender": gender,
                    "expression": expression,
                    "face_box": face_box,
                    "is_child": is_child
                })

            except Exception:
                # لو فشل الكشف على الـROI، نعتبر لا يوجد وجه (أسرع من عمل fallback للإطار الكامل)
                results.append(self._empty_face())

        return results

    @staticmethod
    def _map_emotion(e):
        """
        DeepFace يعطي: angry, disgust, fear, happy, sad, surprise, neutral
        نطابقها لثلاث فئات عندك: happy / neutral / tired
        """
        if not e:
            return "neutral"
        e = e.lower()
        if e == "happy":
            return "happy"
        if e in ("sad", "fear", "disgust"):
            return "tired"
        # 'angry' و 'surprise' نعتبرها أقرب للـ neutral للتبسيط
        return "neutral"

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
