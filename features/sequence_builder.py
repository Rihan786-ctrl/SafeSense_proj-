from collections import deque

import numpy as np


class SkeletonSequenceBuilder:
    """
    Builds fixed-length temporal skeleton sequences.

    Each skeleton has shape:
        (17, 3)

    A complete sequence has shape:
        (window_size, 17, 3)
    """

    def __init__(self, window_size=30):
        self.window_size = window_size

        self.sequence = deque(
            maxlen=window_size
        )

    def add_skeleton(self, skeleton):
        """
        Add one skeleton to the temporal buffer.

        Args:
            skeleton: numpy array of shape (17, 3)

        Returns:
            A complete sequence of shape
            (30, 17, 3) when ready.
            Otherwise None.
        """

        skeleton = np.asarray(
            skeleton,
            dtype=np.float32
        )

        expected_shape = (17, 3)

        if skeleton.shape != expected_shape:
            raise ValueError(
                f"Expected skeleton shape "
                f"{expected_shape}, "
                f"got {skeleton.shape}"
            )

        self.sequence.append(skeleton)

        if len(self.sequence) < self.window_size:
            return None

        return np.stack(
            self.sequence,
            axis=0
        )