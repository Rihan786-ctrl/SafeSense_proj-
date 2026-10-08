import numpy as np


class SkeletonProcessor:
    """
    Converts pose-estimation output into a normalized
    skeleton representation suitable for temporal
    activity recognition.
    """

    def __init__(self, normalize=True):
        self.normalize = normalize

    def process(self, person):
        """
        Process one detected person's pose.

        Args:
            person: Dictionary containing:
                keypoints: (17, 2)
                confidence: (17,)

        Returns:
            Skeleton array of shape (17, 3):
                [x, y, confidence]
        """

        keypoints = np.asarray(
            person["keypoints"],
            dtype=np.float32
        )

        confidence = person.get("confidence")

        if confidence is None:
            confidence = np.ones(
                keypoints.shape[0],
                dtype=np.float32
            )
        else:
            confidence = np.asarray(
                confidence,
                dtype=np.float32
            )

        if self.normalize:
            keypoints = self._normalize_keypoints(
                keypoints
            )

        skeleton = np.concatenate(
            [
                keypoints,
                confidence[:, None]
            ],
            axis=1
        )

        return skeleton

    def _normalize_keypoints(self, keypoints):
        """
        Normalize keypoints relative to the person's
        bounding extent.

        Output coordinates are approximately in [0, 1].
        """

        min_xy = np.min(
            keypoints,
            axis=0
        )

        max_xy = np.max(
            keypoints,
            axis=0
        )

        scale = max_xy - min_xy

        scale = np.where(
            scale < 1e-6,
            1.0,
            scale
        )

        normalized = (
            keypoints - min_xy
        ) / scale

        return normalized