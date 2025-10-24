import cv2
import os
import numpy as np
from PIL import Image

class Display:
    """
    Improved display module:
      - Optional face blurring
      - Cached ad rendering
      - Lite mode (smaller overlay and text)
      - Robust to invalid frames / ads
    """

    def __init__(self, blur_faces=False):
        self.blur_faces = blur_faces
        self.current_ad_path = None
        self.current_reason = ""
        self.cached_ad = None
        self.cached_path = None
        self.lite_mode = os.getenv("LITE_MODE", "0") == "1"

    def set_ad(self, ad_path, reason):
        """Set ad path and reason. Reload only if path changed."""
        if ad_path != self.cached_path:
            self.cached_ad = None
        self.current_ad_path = ad_path
        self.current_reason = reason or ""

    def render(self, frame, persons, faces, draw_utils):
        if frame is None or frame.size == 0:
            return frame

        # --- Draw person boxes
        for p in persons:
            label = f"ID {p.get('id', '?')}"
            draw_utils.draw_box(frame, p.get('box'), (0, 255, 0), label=label)

        # --- Draw faces or blur
        for f in faces:
            fb = f.get('face_box')
            if not fb:
                continue
            if self.blur_faces:
                draw_utils.blur_region(frame, fb)
            else:
                draw_utils.draw_box(frame, fb, (255, 0, 0), label="face")

        # --- Draw current ad (if exists)
        ad_path = self.current_ad_path
        if ad_path and os.path.exists(ad_path):
            try:
                # cache the ad image so we don't reload every frame
                if self.cached_ad is None or self.cached_path != ad_path:
                    ad = Image.open(ad_path).convert("RGB")
                    size = (240, 135) if self.lite_mode else (320, 180)
                    ad = ad.resize(size)
                    self.cached_ad = cv2.cvtColor(np.array(ad), cv2.COLOR_RGB2BGR)
                    self.cached_path = ad_path

                ad_np = self.cached_ad
                h, w = frame.shape[:2]
                ad_h, ad_w = ad_np.shape[:2]
                x0, y0 = w - ad_w - 10, 10

                # prevent clipping
                if y0 + ad_h <= h and x0 + ad_w <= w:
                    frame[y0:y0 + ad_h, x0:x0 + ad_w] = ad_np
                    cv2.rectangle(frame, (x0, y0), (x0 + ad_w, y0 + ad_h), (0, 0, 0), 2)

                    # reason text background (semi-transparent)
                    if self.current_reason:
                        overlay = frame.copy()
                        text_y = y0 + ad_h + 22
                        cv2.rectangle(
                            overlay,
                            (x0, y0 + ad_h),
                            (x0 + ad_w, y0 + ad_h + 30),
                            (0, 0, 0),
                            -1,
                        )
                        alpha = 0.5
                        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
                        cv2.putText(
                            frame,
                            self.current_reason,
                            (x0 + 5, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            1,
                            cv2.LINE_AA,
                        )
            except Exception as e:
                print(f"[Display] failed to render ad: {e}")
                pass

        return frame
