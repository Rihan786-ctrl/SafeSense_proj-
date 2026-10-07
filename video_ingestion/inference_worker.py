import threading
import queue


class InferenceWorker:

    def __init__(self, inference_function):

        self.inference_function = inference_function

        # Keep only the latest pending window
        self.window_queue = queue.Queue(maxsize=1)

        self.running = False
        self.thread = None

    def start(self):

        self.running = True

        self.thread = threading.Thread(
            target=self._worker_loop,
            daemon=True
        )

        self.thread.start()

    def submit(self, window):

        # If an old window is waiting, remove it.
        if self.window_queue.full():

            try:
                self.window_queue.get_nowait()
            except queue.Empty:
                pass

        # Add the latest window
        try:
            self.window_queue.put_nowait(window)
        except queue.Full:
            pass

    def _worker_loop(self):

        while self.running:

            try:
                window = self.window_queue.get(
                    timeout=0.1
                )
            except queue.Empty:
                continue

            result = self.inference_function(window)

            print(
                f"Inference completed: {result}"
            )

    def stop(self):

        self.running = False

        if self.thread:
            self.thread.join(timeout=2)