"""
YOLODetector: small wrapper around ultralytics.YOLO model for inference.
"""
from ultralytics import YOLO


class YOLODetector:
    def __init__(self, weights='yolo11n.pt', device='cpu'):
        self.model = YOLO(weights)
        try:
            self.model.to(device)
        except Exception:
            # some ultralytics versions ignore .to() or require different call
            pass

    def predict(self, frame, conf=0.35):
        # returns Results or list[Results]
        results = self.model(frame, conf=conf, verbose=False)
        return results