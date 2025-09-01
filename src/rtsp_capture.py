"""
Simple RTSP capture wrapper with auto-reconnect.
"""
import cv2
import time


class RTSPCapture:
    def __init__(self, rtsp_url, reconnect_wait=1.0):
        self.rtsp_url = rtsp_url
        self.reconnect_wait = reconnect_wait
        self.cap = None
        self._open()

    def _open(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cap = cv2.VideoCapture(self.rtsp_url)
        # small sleep to let stream initialize
        time.sleep(0.3)

    def opened(self):
        return (self.cap is not None) and self.cap.isOpened()

    def read(self):
        if not self.opened():
            # try to reconnect
            try:
                self._open()
            except Exception:
                time.sleep(self.reconnect_wait)
                return False, None

        ret, frame = self.cap.read()
        if not ret:
            # try a quick reconnect
            try:
                self._open()
            except Exception:
                pass
            time.sleep(self.reconnect_wait)
            return False, None
        return True, frame

    def release(self):
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass