import cv2
import time
import threading


class Camera:

    def __init__(self, camera_id=0, width=640, height=480, fps=15):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps

        self.cap = None
        self.running = False
        self.thread = None
        self.frame_callback = None

    def start(self, frame_callback):

        self.frame_callback = frame_callback

        self.cap = cv2.VideoCapture(self.camera_id)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera.")

        self.running = True

        self.thread = threading.Thread(
            target=self._capture_loop,
            daemon=True
        )

        self.thread.start()

    def _capture_loop(self):

        frame_interval = 1.0 / self.fps

        while self.running:

            start_time = time.perf_counter()

            ret, frame = self.cap.read()

            if not ret:
                print("Failed to read frame.")
                continue

            if self.frame_callback:
                self.frame_callback(frame)

            elapsed = time.perf_counter() - start_time

            sleep_time = max(
                0,
                frame_interval - elapsed
            )

            time.sleep(sleep_time)

    def stop(self):

        self.running = False

        if self.thread:
            self.thread.join(timeout=2)

        if self.cap:
            self.cap.release()