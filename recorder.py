import cv2
import os
from datetime import datetime
from utils.logger import logger


def init_camera(camera_index=0):
    cam = cv2.VideoCapture(camera_index)
    if not cam.isOpened():
        raise RuntimeError(f"Failed to open camera with index {camera_index}.")
    logger.info(f"Camera {camera_index} initialized.")
    return cam

# Creates an output directory with the given folder name if it doesn't exist.
def create_output_dir(folder_name="Aufnahmen"):
    
    output_dir = os.path.join(os.getcwd(), folder_name)
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Output directory ready: {output_dir}")

    return output_dir

# Initializes the camera and video writer for recording.
def init_recorder(cam):
    ret, frame = cam.read()
    if not ret:
        raise RuntimeError("Failed to grab initial frame for recorder initialization.")

    frame_height, frame_width = frame.shape[:2]
    fps = cam.get(cv2.CAP_PROP_FPS) or 30.0

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    output_dir = create_output_dir()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = os.path.join(output_dir, f"video_{timestamp}.mp4")

    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    logger.info("Recording started.")

    return out

#Reads a frame from the camera. Raises an error if reading fails.
def get_frame(cam):
    ret, frame = cam.read()
    if not ret:
        raise RuntimeError("Failed to read frame from camera.")
    return frame

# Releases camera and video writer resources and closes all OpenCV windows.
def release_resources(cam, out):
    cam.release()
    out.release()
    logger.info("Recorder and camera released.")
    cv2.destroyAllWindows()