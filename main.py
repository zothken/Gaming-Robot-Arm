import cv2
from recorder import init_camera, init_recorder, get_frame, release_resources
from figure_detector import detect_figures
from tracker import CentroidTracker
from utils.logger import logger
from utils.visualization import show_frames
# from utils.timing import FPSTracker


def main():
    logger.info("Starting Gaming Robot Arm System...")

    cam = init_camera(0)
    out = init_recorder(cam)

    tracker = CentroidTracker()
    # fps_tracker = FPSTracker()

    while True:
        frame = get_frame(cam)
        if frame is None:
            logger.warning("Empty frame received — stopping.")
            break

        processed, gray, blurred, thresh = detect_figures(frame, tracker=tracker)

        # fps = fps_tracker.update()
        # logger.debug(f"FPS: {fps:.2f}")

        show_frames({
            "Grayscale": gray,
            "Blurred": blurred,
            "Thresholded": thresh,
            "Processed": processed
        })

        out.write(processed)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            logger.info("Exit requested by user.")
            break

    release_resources(cam, out)
    logger.info("Recording finished. System stopped cleanly.")

if __name__ == "__main__":
    main()
