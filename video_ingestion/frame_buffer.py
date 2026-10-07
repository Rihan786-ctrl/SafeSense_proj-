from collections import deque


class SlidingFrameBuffer:

    def __init__(self, window_size=30, stride=5):
        self.window_size = window_size
        self.stride = stride

        self.buffer = deque(maxlen=window_size)
        self.new_frames = 0

    def add_frame(self, frame):
        """
        Add one frame to the buffer.

        Returns:
            A 30-frame snapshot when a prediction window is ready.
            None otherwise.
        """

        self.buffer.append(frame)
        self.new_frames += 1

        # Wait until the first 30 frames are collected
        if len(self.buffer) < self.window_size:
            return None

        # After the first window, create a new window
        # every time 5 new frames arrive.
        if self.new_frames >= self.stride:
            self.new_frames = 0

            # Return a copy of the current window.
            return list(self.buffer)

        return None