import cv2
import numpy as np
import os
from datetime import datetime

# Initialize webcam
cam = cv2.VideoCapture(1)

# Get the default frame width and height
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Output folder relative to Script
output_dir = "Aufnahmen"
# Ordner anlegen, falls nicht vorhanden
os.makedirs(output_dir, exist_ok=True)  

# folder name with timestamp
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_path = os.path.join(output_dir, f"video_{timestamp}.mp4")

# VideoWriter initialisieren (Pfad, Codec, FPS, Größe)
out = cv2.VideoWriter(output_path, fourcc, 20.0, (frame_width, frame_height))

# Parameters for filtering contours (disk size)
MIN_AREA = 100
MAX_AREA = 2000

while True:
    ret, frame = cam.read()
    if not ret:
        break

    # Convert to grayscale and blur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Thresholding (invert if black disks on light background)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
        )

    # Find contours
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    black_count = 0
    white_count = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if MIN_AREA < area < MAX_AREA:
            # Minimum enclosing circle
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y))
            radius = int(radius)

            # Create mask for this contour
            mask = np.zeros(gray.shape, dtype="uint8")
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            mean_val = cv2.mean(gray, mask=mask)[0]

            # Classify color
            if mean_val < 128:
                color = "black"
                black_count += 1
                circle_color = (0, 0, 0)  # BGR
            else:
                color = "white"
                white_count += 1
                circle_color = (255, 255, 255)

            # Draw circle and label
            cv2.circle(frame, center, radius, (0, 255, 0), 2)
            cv2.putText(frame, color, (center[0]-20, center[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1
                        )

            # Print coordinates
            print(f"{color} disk at {center}")

    # Show counts
    cv2.putText(frame, f"Black: {black_count}  White: {white_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2
                )

   # Show all steps in separate windows
    cv2.imshow("Original Frame", frame)
    cv2.imshow("Grayscale", gray)
    cv2.imshow("Blurred", blurred)
    cv2.imshow("Thresholded", thresh)

    # Write the frame to the output file
    #out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
out.release()
cv2.destroyAllWindows()
