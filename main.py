import cv2
from vision.recorder import init_camera, init_recorder, get_frame, release_resources
from vision.figure_detector import detect_figures
from vision.tracker import CentroidTracker
from utils.logger import logger
from utils.visualization import show_frames

# Download .zip file of this project:
# zip -r gaming-robot-arm-light.zip . -x ".venv/*" "uArm-Python-SDK-2.0/*" "__pycache__/*" "Aufnahmen/*"

def main():
    logger.info("Starting Gaming Robot Arm System...")

    cam = init_camera(0)
    out = init_recorder(cam)

    tracker=CentroidTracker()

    while True:
        frame = get_frame(cam)
        if frame is None:
            logger.warning("Empty frame received — stopping.")
            break

        processed, gray, blurred, thresh = detect_figures(
            frame, tracker=tracker)

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
