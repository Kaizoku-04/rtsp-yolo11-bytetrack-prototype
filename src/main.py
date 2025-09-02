"""
Entry point: set up capture, detector, tracker, and the main loop.
"""
import argparse
import time
from src.rtsp_capture import RTSPCapture
from src.detector import YOLODetector
from src.tracker import ByteTrackerWrapper
from src.utils import draw_tracks_as_list, detections_from_result
import cv2
import numpy as np

try:
    # tkinter is in stdlib; used only to query screen size
    from tkinter import Tk
except Exception:
    Tk = None

import csv
from datetime import datetime, timezone


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--rtsp', required=True, help='RTSP URL for phone stream')
    p.add_argument('--model', default='yolo11n.pt', help='YOLO model (weights)')
    p.add_argument('--conf', type=float, default=0.35, help='detection confidence')
    p.add_argument('--device', default='cpu', help='device, e.g. cpu or cuda:0')
    p.add_argument('--fps', type=int, default=20, help='expected FPS for tracker')
    p.add_argument('--resize', type=int, default=720, help='max dimension to resize for processing (0 to disable)')
    return p.parse_args()

WINDOW_NAME = 'rtsp->yolo11->bytetrack'
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)  # allows manual resize

#get screen size
screen_w, screen_h = None, None
if Tk is not None:
    try:
        root = Tk()
        root.withdraw()
        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
    except Exception:
        screen_w, screen_h = None, None

#fallback: if we couldn't get screen size, use some safe defaults
if screen_w is None or screen_h is None:
    screen_w, screen_h = 1366, 768

#optionally move window to top-left
try:
    cv2.moveWindow(WINDOW_NAME, 0, 0)
except Exception:
    pass

is_fullscreen = False

def read_latest_frame(cap, max_reads = 10):
    """
    Read repeatedly from cap and return the latest decoded frame.
    max_reads prevents an infinite loop on some buggy backends.
    """
    frame = None
    reads = 0
    while reads < max_reads:
        ret, tmp = cap.read()
        if not ret:
            break
        frame = tmp
        reads += 1
    return frame

def main():

    LOG_DETECTIONS = True
    LOG_FILE = 'detections.csv'
    csv_file = None
    csv_writer = None
    if LOG_DETECTIONS:
        csv_file = open(LOG_FILE, 'a', newline='')
        csv_writer = csv.writer(csv_file)
        #write header only if file new/empty (simple approach)
        try:
            csv_file.seek(0)
            if not csv_file.read(1):
                csv_writer.writerow(['timestamp','frame_idx', 'track_id', 'cls', 'x1', 'y1', 'x2', 'y2', 'score'])
                csv_file.flush()
        except Exception:
            pass
    frame_idx = 0

    args = parse_args()

    cap = RTSPCapture(args.rtsp)
    if not cap.opened():
        print('ERROR: cannot open RTSP stream:', args.rtsp)
        return

    detector = YOLODetector(args.model, device=args.device)
    tracker = ByteTrackerWrapper(frame_rate=args.fps)

    fps_time = time.time()
    fps_counter = 0
    fps = 0.0

    while True:
        frame = read_latest_frame(cap)
        if frame is None:
            time.sleep(0.01)
            continue


        # Store original dimensions
        orig_h, orig_w = frame.shape[:2]
        
        # Resize for processing if needed
        if args.resize and args.resize > 0:
            h, w = frame.shape[:2]
            if max(h, w) > args.resize:
                scale = args.resize / max(h, w)
                frame_proc = cv2.resize(frame, (int(w*scale), int(h*scale)))
            else:
                frame_proc = frame
        else:
            frame_proc = frame

        # Get processed dimensions
        proc_h, proc_w = frame_proc.shape[:2]

        # Detection on processed frame
        results = detector.predict(frame_proc, conf=args.conf)
        result = results[0] if isinstance(results, (list, tuple)) else results
        dets = detections_from_result(result)  # These are in frame_proc coordinates

        # Tracking (using processed frame coordinates)
        tracks = tracker.update(dets, frame_proc)  # Pass frame_proc for consistency

        # Transform tracks back to original coordinates if we resized
        if proc_w != orig_w or proc_h != orig_h:
            scale_x = orig_w / proc_w
            scale_y = orig_h / proc_h
            if tracks is not None and len(tracks) > 0:
                # Convert tracks to numpy array for easy processing
                tracks_arr = np.array(tracks)
                # Scale bounding box coordinates
                tracks_arr[:, [0, 2]] *= scale_x  # x1, x2
                tracks_arr[:, [1, 3]] *= scale_y  # y1, y2
                tracks = tracks_arr.tolist()

        # tracking
        tracks = tracker.update(dets, frame_proc)

        frame_idx += 1
        if LOG_DETECTIONS and csv_writer is not None:
            ts = datetime.now(timezone.utc)
            if isinstance(tracks, (list, tuple, np.ndarray)) and len(tracks):
                for row in tracks:
                    try:
                        # handle both numpy rows and track objects
                        if isinstance(row, (list, tuple, np.ndarray)):
                            x1, y1, x2, y2, score, cls, tid = row[:7]
                        else:
                            # assume it's a ByteTrack object with .tlbr and .track_id
                            x1, y1, x2, y2 = row.tlbr
                            tid = row.track_id
                            score = getattr(row, "score", 1.0)
                            cls = getattr(row, "cls", -1)

                        csv_writer.writerow([ts, frame_idx, int(tid), int(cls),
                                            int(x1), int(y1), int(x2), int(y2), float(score)])
                    except Exception as e:
                        print("[CSV LOGGING ERROR]", e, "row:", row)
            csv_file.flush()


        # draw
        out = frame.copy()
        out = draw_tracks_as_list(out, tracks)

        # fps calc
        fps_counter += 1
        if time.time() - fps_time >= 1.0:
            fps = fps_counter / (time.time() - fps_time)
            fps_time = time.time()
            fps_counter = 0
        cv2.putText(out, f'FPS: {fps:.1f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50,205,50), 2)

        h, w = out.shape[:2]
        #reserve some pixels for window decorations / taskbar
        usable_h = int(screen_h * 0.90) # use 90% of screen height to be safe
        usable_w = int(screen_w * 0.98) # use 89% of screen width
        scale = min(1.0, usable_w / w, usable_h / h)

        if scale < 1.0:
            disp_w, disp_h = int(w * scale), int(h * scale)
            disp_frame = cv2.resize(out, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
        else:
            disp_frame = out 
        
        cv2.imshow(WINDOW_NAME, disp_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('f'):
            # toggle fullscreen
            is_fullscreen = not is_fullscreen
            if is_fullscreen:
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        elif key == ord('r'):
            # Reset to window-normal and move to top-left
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            try:
                cv2.moveWindow(WINDOW_NAME, 0, 0)
            except Exception:
                pass

    if csv_file is not None:
        csv_file.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()