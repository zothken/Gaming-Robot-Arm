import cv2
import numpy as np
from tracker import CentroidTracker
from utils.logger import logger
from utils.visualization import draw_detections

# Detects circular figures in the frame, classifies them by color, and annotates the frame.
def detect_figures(frame, tracker=None):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Parameters for filtering contours (disk size)
    MIN_RADIUS = 20
    MAX_RADIUS = 50

    # Ensure block_size is odd and >=3
    THRESH_BLOCK = 49
    if THRESH_BLOCK % 2 == 0:
        THRESH_BLOCK += 1
    THRESH_C = 10

    # Adaptive Threshold
    thresh = cv2.adaptiveThreshold(
        blurred, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 
        THRESH_BLOCK, 
        THRESH_C
        )


    centroids = []
    colors = []
    black_count, white_count = 0, 0

    # Hough Circles
    circles = cv2.HoughCircles(blurred, 
        cv2.HOUGH_GRADIENT, 
        dp=1.2, minDist=20, 
        param1=50, 
        param2=30, 
        minRadius=MIN_RADIUS, 
        maxRadius=MAX_RADIUS
        )
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")

        for (x, y, r) in circles:

            center = (x, y)
            centroids.append(center)

            # Create mask for local brightness measurement
            mask = np.zeros(blurred.shape, dtype="uint8")
            cv2.circle(mask, center, r, 255, -1)
            mean_val = cv2.mean(blurred, mask=mask)[0]

            if mean_val < 128:
                color = "black"
                black_count += 1
            else:
                color = "white"
                white_count += 1
            colors.append(color)

    if tracker is not None:
        tracker.update(centroids, colors)

    if circles is None or len(centroids) == 0:
        return frame, gray, blurred, thresh

    draw_detections(frame, circles, colors, black_count, white_count)
    
    if tracker:
        objects = tracker.update(centroids, colors)
        for (objectID, (centroid, color)) in objects.items():
            smoothed = np.mean(tracker.positions[objectID], axis=0).astype(int)
            cv2.putText(frame, f"ID {objectID}", (smoothed[0] - 10, smoothed[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            cv2.circle(frame, tuple(smoothed), 4, (255, 255, 0), -1)

   
    return frame, gray, blurred, thresh