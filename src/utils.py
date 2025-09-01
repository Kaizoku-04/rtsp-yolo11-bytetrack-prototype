"""
Helper utilities for conversions and drawing.
"""
import numpy as np
import cv2


def xyxy_to_tlwh(xyxy):
    x1, y1, x2, y2 = xyxy
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    return [x1, y1, w, h]


def detections_from_result(result):
    """
    Convert Ultralytics result.boxes to np.ndarray of shape (N,6):
    [tl_x, tl_y, w, h, score, cls]
    """
    boxes = getattr(result, 'boxes', None)
    if boxes is None:
        return np.zeros((0,6), dtype=float)

    try:
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy()
    except Exception:
        # fallback if attributes are plain lists
        arr = np.array(boxes)
        if arr.size == 0:
            return np.zeros((0,6), dtype=float)
        # best-effort parsing
        xyxy = arr[:, :4]
        conf = arr[:, 4]
        cls = arr[:, 5] if arr.shape[1] > 5 else np.zeros(len(arr))

    dets = []
    for b, s, c in zip(xyxy, conf, cls):
        tlwh = xyxy_to_tlwh(b)
        dets.append([tlwh[0], tlwh[1], tlwh[2], tlwh[3], float(s), float(c)])
    if len(dets) == 0:
        return np.zeros((0,6), dtype=float)
    return np.array(dets, dtype=float)


def draw_tracks(frame, tracks):
    """
    Draw bounding boxes and track ids. Accepts tracker outputs or raw detections.
    """
    if tracks is None:
        return frame

    for t in tracks:
        try:
            if isinstance(t, (list, tuple, np.ndarray)) and len(t) >= 7:
                x1, y1, x2, y2, score, cls, tid = t[:7]
            elif hasattr(t, 'tlwh') and hasattr(t, 'track_id'):
                tlwh = t.tlwh
                x1, y1, w, h = tlwh
                x2 = x1 + w
                y2 = y1 + h
                tid = getattr(t, 'track_id', -1)
                score = getattr(t, 'score', 0)
                cls = getattr(t, 'cls', 0)
            else:
                continue

            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,220,0), 2)
            label = f"ID:{int(tid)} cls:{int(cls)}"
            cv2.putText(frame, label, (x1, max(15, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,220,0), 2)
        except Exception:
            continue
    return frame
