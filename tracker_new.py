import cv2
import numpy as np
import os
from datetime import datetime

# Camera setup
cam = cv2.VideoCapture(1)
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

# # Define the codec and create VideoWriter object
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# # Output folder relative to Script
# output_dir = "Aufnahmen"
# # Create folder if not found
# os.makedirs(output_dir, exist_ok=True)  

# # folder name with timestamp
# timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# output_path = os.path.join(output_dir, f"video_{timestamp}.mp4")

# # Initialize VideoWriter (path, codec, FPS, size)
# #out = cv2.VideoWriter(output_path, fourcc, 20.0, (frame_width, frame_height))

# Parameters for filtering contours (disk size)
MIN_AREA = 250
MAX_AREA = 750

#tracker = CentroidTracker(maxDisappeared=30)

cv2.namedWindow("Thresholded")

def nothing(x): pass

# Slider for adaptive Threshold parameters
cv2.createTrackbar("Block Size", "Thresholded", 40, 100, nothing)
cv2.createTrackbar("C", "Thresholded", 10, 20, nothing)

# Save previous values
last_block_size, last_C = -1, -1


while True:
    ret, frame = cam.read()
    if not ret: break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #gray = cv2.equalizeHist(gray)  # Histogram equalization
    #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    #gray = clahe.apply(gray) # Histogram equalization via CLAHE

    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Trackbar values
    block_size = cv2.getTrackbarPos("Block Size", "Thresholded")
    C = cv2.getTrackbarPos("C", "Thresholded")

    if (block_size != last_block_size or C != last_C ):
        print(f"Block Size: {block_size}, C: {C}")
        last_block_size, last_C = block_size, C

    # Ensure block_size is odd and >=3
    if block_size % 2 == 0: block_size += 1
    if block_size < 3: block_size = 3

    # Adaptive Threshold
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, block_size, C)

    # Find contours
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centroids = []
    black_count, white_count = 0, 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if MIN_AREA < area < MAX_AREA:

            # check how circular the contour is
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0: continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if not (0.85 < circularity < 1.1): continue

            (x, y), radius = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y))
            centroids.append(center)

            # Create mask for this contour
            mask = np.zeros(gray.shape, dtype="uint8")
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            mean_val = cv2.mean(gray, mask=mask)[0]

            if mean_val < 128:
                color = "black"
                black_count += 1
                circle_color = (0, 0, 0)
            else:
                color = "white"
                white_count += 1
                circle_color = (255, 255, 255)

            # Draw circle and label
            cv2.circle(frame, center, int(radius), (0, 255, 0), 2)
            cv2.putText(frame, color, (center[0] - 20, center[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Hough Circles
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20, param1=50, param2=30, minRadius=10, maxRadius=60)

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            # Draw Hough circles
            cv2.circle(frame, (x, y), r, (255, 0, 0), 2)
            cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)

    #objects = tracker.update(centroids)

    # for (objectID, centroid) in objects.items():
    #     # Position glätten
    #     smoothed = np.mean(tracker.positions[objectID], axis=0).astype(int)
    #     text = f"ID {objectID}"
    #     cv2.putText(frame, text, (smoothed[0] - 10, smoothed[1] - 10),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    #     cv2.circle(frame, tuple(smoothed), 4, (255, 255, 0), -1)

    # Show counts
    cv2.putText(frame, f"Black: {black_count}  White: {white_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imshow("Original Frame", frame)
    cv2.imshow("Grayscale", gray)
    cv2.imshow("Blurred", blurred)
    cv2.imshow("Thresholded", thresh)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

cam.release()
#out.release()
cv2.destroyAllWindows()