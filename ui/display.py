import cv2
import os
import numpy as np
from PIL import Image

class Display:
    def __init__(self, blur_faces=False):
        self.blur_faces = blur_faces
        self.current_ad_path = None
        self.current_reason = ""

    def set_ad(self, ad_path, reason):
        self.current_ad_path = ad_path
        self.current_reason = reason or ""

    def render(self, frame, persons, faces, draw_utils):
        # Draw person boxes
        for i, p in enumerate(persons):
            label = f"ID {p['id']}"
            draw_utils.draw_box(frame, p['box'], (0,255,0), label=label)

        # Draw faces and optionally blur
        for f in faces:
            if not f['face_box']: continue
            if self.blur_faces:
                draw_utils.blur_region(frame, f['face_box'])
            else:
                draw_utils.draw_box(frame, f['face_box'], (255,0,0), label="face")

        # Render ad thumbnail on the right-top corner
        if self.current_ad_path and os.path.exists(self.current_ad_path):
            try:
                ad = Image.open(self.current_ad_path).convert("RGB")
                ad = ad.resize((320, 180))
                ad_np = cv2.cvtColor(np.array(ad), cv2.COLOR_RGB2BGR)
                h, w = frame.shape[:2]
                x0, y0 = w-330, 10
                frame[y0:y0+180, x0:x0+320] = ad_np
                cv2.rectangle(frame, (x0, y0), (x0+320, y0+180), (0,0,0), 2)
                if self.current_reason:
                    cv2.putText(frame, self.current_reason, (x0, y0+200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2, cv2.LINE_AA)
                    cv2.putText(frame, self.current_reason, (x0, y0+200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            except Exception:
                pass
        return frame
