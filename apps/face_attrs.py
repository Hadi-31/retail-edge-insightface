import cv2
import numpy as np
from insightface.app import FaceAnalysis

class FaceAttrs:
    def __init__(self, det_size=(640,640)):
        # buffalo_l includes detection + attributes; downloads on first run
        self.app = FaceAnalysis(name='buffalo_l')
        self.app.prepare(ctx_id=0, det_size=det_size)

    def analyze(self, frame, person_boxes):
        '''
        For each person box, try to find a face inside; return list of per-person dicts:
        {'has_face':bool, 'age':int|None, 'gender':str|None, 'expression':str, 'face_box':(x1,y1,x2,y2)|None, 'is_child':bool}
        '''
        faces = self.app.get(frame)  # detect faces in whole frame
        results = []

        # Convert faces to dicts
        face_items = []
        for f in faces:
            box = f.bbox.astype(int)  # (x1,y1,x2,y2)
            age = getattr(f, 'age', None)
            gender_idx = getattr(f, 'gender', None)
            gender = None
            if gender_idx is not None:
                gender = 'male' if gender_idx == 0 else 'female'
            expression = 'neutral'  # InsightFace doesn't expose expression directly here
            face_items.append({'box': tuple(int(x) for x in box), 'age': age, 'gender': gender, 'expression': expression})

        # assign faces to persons by IOU of boxes
        def iou(a,b):
            xA=max(a[0],b[0]); yA=max(a[1],b[1]); xB=min(a[2],b[2]); yB=min(a[3],b[3])
            inter=max(0,xB-xA)*max(0,yB-yA)
            areaA=(a[2]-a[0])*(a[3]-a[1]); areaB=(b[2]-b[0])*(b[3]-b[1])
            return inter/ (areaA+areaB-inter+1e-6)

        for pb in person_boxes:
            best, best_iou = None, 0.0
            for fi in face_items:
                iouv = iou(pb, fi['box'])
                if iouv > best_iou:
                    best, best_iou = fi, iouv
            if best and best_iou > 0.05:
                age = best['age']
                is_child = (age is not None and age < 13)
                results.append({'has_face': True, 'age': age, 'gender': best['gender'],
                                'expression': best['expression'], 'face_box': best['box'],
                                'is_child': is_child})
            else:
                results.append({'has_face': False, 'age': None, 'gender': None,
                                'expression': None, 'face_box': None, 'is_child': False})
        return results
