
import threading

from video_ingestion.camera import Camera
from video_ingestion.pose_worker import PoseProcessingWorker
from video_ingestion.inference_worker import InferenceWorker

from pose.pose_estimator import PoseEstimator
from features.skeleton import SkeletonProcessor
from features.sequence_builder import SkeletonSequenceBuilder


def inference_function(sequence):
    """
    Temporary validator until the activity model is trained.

    Expected sequence shape: (30, 17, 3).
    """
    return {
        "status": "validation_only",
        "sequence_shape": sequence.shape,
        "sequence_dtype": str(sequence.dtype),
    }


def main():
    pose_estimator = PoseEstimator()
    skeleton_processor = SkeletonProcessor(normalize=True)
    sequence_builder = SkeletonSequenceBuilder(
        window_size=30,
        stride=5,
    )

    inference_worker = InferenceWorker(
        inference_function=inference_function,
    )

    # Protect the sequence builder if callbacks are ever invoked
    # concurrently.
    sequence_lock = threading.Lock()

    counters = {
        "frames_received": 0,
        "poses_detected": 0,
        "sequences_submitted": 0,
    }

    def process_frame(frame):
        counters["frames_received"] += 1

        result = pose_estimator.process(frame)
        persons = pose_estimator.extract_keypoints(result)

        # Reset the sequence if no person is detected.
        if not persons:
            with sequence_lock:
                sequence_builder.reset()
            return

        counters["poses_detected"] += 1
        skeleton = skeleton_processor.process(persons[0])

        with sequence_lock:
            sequence = sequence_builder.add_skeleton(skeleton)

        if sequence is not None:
            counters["sequences_submitted"] += 1
            inference_worker.submit(sequence)

    pose_worker = PoseProcessingWorker(process_frame)

    camera = Camera(
        camera_id=0,
        width=640,
        height=480,
        fps=15,
    )

    try:
        inference_worker.start()
        pose_worker.start()
        camera.start(pose_worker.submit)

        print("SafeSense skeleton pipeline started.")
        print("Press Ctrl+C to stop.")

        stop_event = threading.Event()

        while not stop_event.wait(timeout=1):
            pass

    except KeyboardInterrupt:
        print("\nStopping SafeSense...")

    finally:
        camera.stop()
        pose_worker.stop()
        inference_worker.stop()

        print("SafeSense stopped.")
        print("Frames received:", counters["frames_received"])
        print("Poses detected:", counters["poses_detected"])
        print("Sequences submitted:", counters["sequences_submitted"])


if __name__ == "__main__":
    main()
