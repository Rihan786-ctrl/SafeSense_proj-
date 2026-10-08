from video_ingestion.camera import Camera
from video_ingestion.frame_buffer import SlidingFrameBuffer
from video_ingestion.inference_worker import InferenceWorker


def inference_function(window):
    """
    Temporary inference function used to validate
    the real-time video ingestion pipeline.
    """

    return {
        "status": "success",
        "frames_received": len(window),
        "frame_shape": window[0].shape,
    }


def main():
    buffer = SlidingFrameBuffer(
        window_size=30,
        stride=5
    )

    worker = InferenceWorker(
        inference_function=inference_function
    )

    def process_frame(frame):
        window = buffer.add_frame(frame)

        if window is not None:
            worker.submit(window)

    camera = Camera(
        camera_id=0,
        width=640,
        height=480,
        fps=15
    )

    worker.start()
    camera.start(process_frame)

    print("SafeSense video pipeline started.")
    print("Press Ctrl+C to stop.")

    try:
        while True:
            pass

    except KeyboardInterrupt:
        print("\nStopping SafeSense...")

    finally:
        camera.stop()
        worker.stop()
        print("SafeSense stopped.")


if __name__ == "__main__":
    main()