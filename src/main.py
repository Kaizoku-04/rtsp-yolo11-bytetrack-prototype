"""
Entry point: set up capture, detector, tracker, and the main loop.
"""
import argparse
import time
from src.rtsp_capture import RTSPCapture
from src.detector import YOLODetector
from src.tracker import ByteTrackerWrapper
from src.utils import draw_tracks, detections_from_result
import cv2


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--rtsp', required=True, help='RTSP URL for phone stream')
    p.add_argument('--model', default='yolo11n.pt', help='YOLO model (weights)')
    p.add_argument('--conf', type=float, default=0.35, help='detection confidence')
    p.add_argument('--device', default='cpu', help='device, e.g. cpu or cuda:0')
    p.add_argument('--fps', type=int, default=20, help='expected FPS for tracker')
    p.add_argument('--resize', type=int, default=720, help='max dimension to resize for processing (0 to disable)')
    return p.parse_args()


def main():
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
        ret, frame = cap.read()
        if not ret:
            # reconnect attempts inside RTSPCapture; just wait briefly
            time.sleep(0.2)
            continue

        # optional resize for speed
        if args.resize and args.resize > 0:
            h, w = frame.shape[:2]
            if max(h, w) > args.resize:
                scale = args.resize / max(h, w)
                frame_proc = cv2.resize(frame, (int(w*scale), int(h*scale)))
            else:
                frame_proc = frame
        else:
            frame_proc = frame

        # detection
        results = detector.predict(frame_proc, conf=args.conf)
        result = results[0] if isinstance(results, (list, tuple)) else results

        # detections -> tlwh,score,cls
        dets = detections_from_result(result)

        # tracking
        tracks = tracker.update(dets, frame_proc)

        # draw
        out = frame.copy()
        out = draw_tracks(out, tracks)

        # fps calc
        fps_counter += 1
        if time.time() - fps_time >= 1.0:
            fps = fps_counter / (time.time() - fps_time)
            fps_time = time.time()
            fps_counter = 0
        cv2.putText(out, f'FPS: {fps:.1f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50,205,50), 2)

        cv2.imshow('rtsp->yolo11->bytetrack', out)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()