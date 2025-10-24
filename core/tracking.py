from .utils import iou

class IOUTracker:
    def __init__(self, iou_thresh=0.4, max_age=30):
        self.next_id = 1
        self.tracks = {}  # id -> {'box':..., 'age':0}
        self.iou_thresh = iou_thresh
        self.max_age = max_age

    def update(self, detections):
        assigned = {}
        # Try to match existing tracks
        for tid, t in list(self.tracks.items()):
            best_iou, best_det = 0, None
            for i, det in enumerate(detections):
                if i in assigned: continue
                iouv = iou(t['box'], det['box'])
                if iouv > best_iou:
                    best_iou, best_det = iouv, i
            if best_det is not None and best_iou >= self.iou_thresh:
                self.tracks[tid]['box'] = detections[best_det]['box']
                self.tracks[tid]['age'] = 0
                assigned[best_det] = tid
            else:
                self.tracks[tid]['age'] += 1
                if self.tracks[tid]['age'] > self.max_age:
                    del self.tracks[tid]

        # Create new tracks
        for i, det in enumerate(detections):
            if i in assigned: continue
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = {'box': det['box'], 'age': 0}

        # Output list with ids
        output = []
        for tid, t in self.tracks.items():
            output.append({'id': tid, 'box': t['box']})
        return output
