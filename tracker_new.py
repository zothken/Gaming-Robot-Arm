import cv2
import numpy as np
import os
from datetime import datetime

# Camera setup
cam = cv2.VideoCapture(1)
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Output folder relative to Script
output_dir = "Aufnahmen"
# Create folder if not found
os.makedirs(output_dir, exist_ok=True)  

# folder name with timestamp
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_path = os.path.join(output_dir, f"video_{timestamp}.mp4")

# Initialize VideoWriter (path, codec, FPS, size)
out = cv2.VideoWriter(output_path, fourcc, 20.0, (frame_width, frame_height))


# Parameters for filtering contours (disk size)
MIN_RADIUS = 10
MAX_RADIUS = 60

# Ensure block_size is odd and >=3
THRESH_BLOCK = 49
THRESH_C = 10

#tracker = CentroidTracker(maxDisappeared=30)

def nothing(x): pass

while True:
    ret, frame = cam.read()
    if not ret: break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)


    # Adaptive Threshold
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, THRESH_BLOCK, THRESH_C)


    centroids = []
    black_count, white_count = 0, 0

    # Hough Circles
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20, param1=50, param2=30, minRadius=MIN_RADIUS, maxRadius=MAX_RADIUS)
    if circles is  None: continue

    circles = np.round(circles[0, :]).astype("int")

    for (x, y, r) in circles:

        center = (int(x), int(y))
        centroids.append(center)

        # Create mask for local brightness measurement
        mask = np.zeros(blurred.shape, dtype="uint8")
        cv2.circle(mask, center, r, 255, -1)
        mean_val = cv2.mean(blurred, mask=mask)[0]

        if mean_val < 128:
            color = "black"
            black_count += 1
            circle_color = (255, 0, 0)
        else:
            color = "white"
            white_count += 1
            circle_color = (255, 0, 0)

        # Draw circles around detected figures
        cv2.circle(frame, (x, y), r, (255, 0, 0), 2)
        cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)

        # Label circles with color
        cv2.putText(frame, color, (x - 20, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Show counts
        cv2.putText(frame, f"Black: {black_count}  White: {white_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    #objects = tracker.update(centroids)

    # for (objectID, centroid) in objects.items():
    #     # Position glätten
    #     smoothed = np.mean(tracker.positions[objectID], axis=0).astype(int)
    #     text = f"ID {objectID}"
    #     cv2.putText(frame, text, (smoothed[0] - 10, smoothed[1] - 10),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    #     cv2.circle(frame, tuple(smoothed), 4, (255, 255, 0), -1)

    cv2.imshow("Original Frame", frame)
    cv2.imshow("Grayscale", gray)
    cv2.imshow("Blurred", blurred)
    cv2.imshow("Thresholded", thresh)

    # Write the frame to the output file
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

cam.release()
out.release()
cv2.destroyAllWindows()