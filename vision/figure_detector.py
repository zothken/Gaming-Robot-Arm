import cv2
import numpy as np
from utils.visualization import draw_detections, draw_ids


# Detects circular figures in the frame, classifies them by color, and annotates the frame.
def detect_figures(frame, tracker=None):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Parameters for filtering contours (disk size)
    MIN_RADIUS = 20
    MAX_RADIUS = 30

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
        param1=80, 
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

    if len(centroids) > 0:
        sorted_data = sorted(zip(centroids, colors), key=lambda c: (c[0][0], c[0][1]))
        centroids, colors = zip(*sorted_data)
        centroids, colors = list(centroids), list(colors)

    if tracker is not None:
        tracker.update(centroids, colors)

    if circles is None or len(centroids) == 0:
        return frame, gray, blurred, thresh

    draw_detections(frame, circles, colors, black_count, white_count)
    draw_ids(frame, tracker)
   
    return frame, gray, blurred, thresh