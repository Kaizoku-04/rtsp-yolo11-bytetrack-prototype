"""
ByteTrackerWrapper: wraps ultralytics.byte_tracker if available, otherwise
provides a graceful detection-only fallback.
"""
import numpy as np


try:
    from ultralytics.trackers.byte_tracker import BYTETracker
except Exception:
    BYTETracker = None


class ByteTrackerWrapper:
    def __init__(self, frame_rate=20, track_buffer=30):
        self.frame_rate = frame_rate
        self.track_buffer = track_buffer
        self.tracker = None
        if BYTETracker is not None:
            # attempt to initialize
            try:
                from types import SimpleNamespace
                args = SimpleNamespace(track_buffer=self.track_buffer)
                self.tracker = BYTETracker(args, frame_rate=self.frame_rate)
            except Exception:
                self.tracker = None

    def update(self, dets, img=None):
        """
        dets: np.ndarray of shape (N,6) -> [tl_x, tl_y, w, h, score, cls]
        returns tracker output (list/ndarray) or empty list if no tracker.
        """
        if self.tracker is None:
            # return raw detections as fallback: convert tlwh->xyxy + fake id
            out = []
            for i, d in enumerate(dets):
                x, y, w, h, s, cls = d
                out.append([x, y, x+w, y+h, s, cls, -1])
            return np.array(out) if len(out) else []

        # Some BYTETracker variants expect different input shapes. Try common call
        try:
            tracks = self.tracker.update(dets)
            return tracks
        except TypeError:
            # try including image
            try:
                tracks = self.tracker.update(dets, img)
                return tracks
            except Exception:
                return []
        except Exception:
            return []