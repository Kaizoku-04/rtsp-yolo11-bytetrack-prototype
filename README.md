# RTSP -> OpenCV -> YOLO11n -> ByteTrack (prototype)


This prototype repo captures an RTSP stream (from a phone), runs YOLO11n detections
(via Ultralytics), and performs online tracking with ByteTrack (if available).


Quick start:


1. Install dependencies (see `requirements.txt`). If you have a CUDA-enabled GPU,
install an appropriate `torch` build first (see https://pytorch.org).


```bash
pip install -r requirements.txt
# If you have GPU, install torch first. Example (Linux, CUDA 12):
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```


2. Run the demo (replace RTSP URL):


```bash
python -m src.main --rtsp "rtsp://192.168.1.100:8554/live" --model yolo11n.pt --conf 0.35 --device cpu --resize 720
```


Notes & tips:
- If `ultralytics.trackers.byte_tracker` is not present in your ultralytics package,
the code will still run in detection-only mode. You can either `pip install` an
external ByteTrack implementation or use the `model.track(...)` high-level API.
- To use two streams later, you can run two instances or modify `src/main.py`
to spawn two capture threads and separate trackers. I can create that for you
next if you want.
