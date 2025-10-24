import os
import cv2
import numpy as np

class PersonDetector:
    '''
    Person detector with two backends:
      1) MemryX DFP adapter (if USE_MEMRYX=1 and SDK available)  -> TODO: plug your MemryX calls here
      2) CPU fallback using OpenCV HOG (works anywhere, zero download)
    '''
    def __init__(self):
        self.use_memryx = os.getenv("USE_MEMRYX", "0") == "1"
        self.hog = None
        if not self.use_memryx:
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        # else: initialize your MemryX accelerator here (load DFP, etc.)

    def infer(self, frame):
        '''
        Returns list of detections: [{'box':(x1,y1,x2,y2), 'conf':float}, ...] in frame coordinates
        '''
        if self.use_memryx:
            # TODO: Example pseudocode
            # boxes, scores = self.mx_infer_person(frame)
            # return [{'box': b, 'conf': s} for b, s in zip(boxes, scores)]
            pass

        # CPU fallback (HOG): returns rects in x,y,w,h
        rects, weights = self.hog.detectMultiScale(frame, winStride=(8,8), padding=(8,8), scale=1.05)
        detections = []
        for (x,y,w,h), s in zip(rects, weights):
            x1, y1, x2, y2 = x, y, x+w, y+h
            detections.append({'box': (x1,y1,x2,y2), 'conf': float(s)})
        return detections
