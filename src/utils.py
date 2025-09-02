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


def draw_tracks_as_list(frame, tracks):

    '''
    Draws a compact list of tracks at the bottom-left of the frame.
    Expects tracks like rows [x1,y1,x2,y2,score,cls,track_id] or STrack-like objects.
    '''
    h, w = frame.shape[:2]
    lines = []
    # Normalize tracks into lines of text
    if tracks is None:
        pass
    else:
        #ndarray or list of rows
        try:
            arr = np.array(tracks)
            if arr.ndim == 2 and arr.shape[1] >= 7:
                for row in arr:
                    x1,y1,x2,y2,score,cls,tid = row[:7]
                    lines.append(f"ID{int(tid)} cls{int(cls)} [{int(x1)},{int(y1)}-{int(x2)},{int(y2)}]")
            else:
                # fallback: try iterate
                for t in tracks:
                    if hasattr(t, 'tlwh'):
                        x,y,w_,h_ = t.tlwh
                        x1,y1,x2,y2 = int(x), int(y), int(x+w_), int(y+h_)
                        tid = getattr(t, 'track_id', -1)
                        cls = getattr(t, 'cls',0)
                        lines.append(f"ID{int(tid)} cls{int(cls)} [{int(x1)},{int(y1)}-{int(x2)},{int(y2)}]")
        except Exception:
            pass
    # Render lines bottom_left upward
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    line_height = 18
    padding = 8

    # background box
    box_h = line_height * max(1, len(lines)) + padding*2
    box_w = int(w * 0.45) # make it up to 45% of width
    x0, y0 = 10, h - box_h - 10
    cv2.rectangle(frame, (x0, y0), (x0 + box_w, y0 + box_h), (0, 0, 0), -1)
    cv2.rectangle(frame, (x0,y0), (x0 + box_w, y0 + box_h), (50, 205, 50), 1)

    #draw each line
    for i, line in enumerate(lines[::-1]): #reverse so first item at bottom
        y = y0 + box_h - padding - i * line_height - 4
        cv2.putText(frame, line, (x0 + padding,y), font, font_scale, (200, 255, 200), thickness, cv2.LINE_AA)

    return frame 