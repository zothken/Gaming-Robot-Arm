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
#out = cv2.VideoWriter(output_path, fourcc, 20.0, (frame_width, frame_height))

# Parameters for filtering contours (disk size)
MIN_AREA = 50
MAX_AREA = 250

# Fenster für Threshold + Trackbars
cv2.namedWindow("Thresholded")

def nothing(x):
    pass

# Trackbars erstellen
cv2.createTrackbar("Block Size", "Thresholded", 11, 100, nothing)  # Start=11, max=50
cv2.createTrackbar("C", "Thresholded", 2, 20, nothing)            # Start=2, max=20
cv2.createTrackbar("Mode", "Thresholded", 1, 1, nothing)          # 0=Mean, 1=Gaussian
cv2.createTrackbar("Inv", "Thresholded", 0, 1, nothing)           # 0=Normal, 1=Inverse
#last values to track changes: 50, 3, 1, 0

# Variablen zum Merken der letzten Werte
last_block_size, last_C, last_mode, last_inv = -1, -1, -1, -1

while True:
    ret, frame = cam.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Trackbar-Werte lesen
    block_size = cv2.getTrackbarPos("Block Size", "Thresholded")
    C = cv2.getTrackbarPos("C", "Thresholded")
    mode = cv2.getTrackbarPos("Mode", "Thresholded")
    inv = cv2.getTrackbarPos("Inv", "Thresholded")

    # Werte nur ausgeben, wenn sie sich geändert haben
    if (block_size != last_block_size or C != last_C or 
        mode != last_mode or inv != last_inv):
        print(f"Block Size: {block_size}, C: {C}, Mode: {mode}, Inv: {inv}")
        last_block_size, last_C, last_mode, last_inv = block_size, C, mode, inv

    # Blockgröße korrigieren (muss ungerade und >=3 sein)
    if block_size % 2 == 0:
        block_size += 1
    if block_size < 3:
        block_size = 3

    # Adaptive Methode wählen
    adaptive_method = cv2.ADAPTIVE_THRESH_MEAN_C if mode == 0 else cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    thresh_type = cv2.THRESH_BINARY_INV if inv == 1 else cv2.THRESH_BINARY

    # Adaptive Threshold
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresh_type, block_size, C)

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
            cv2.putText(frame, color, (center[0]-20, center[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # Print coordinates
            #print(f"{color} disk at {center}")

    # Show counts
    cv2.putText(frame, f"Black: {black_count}  White: {white_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

   # Show all steps in separate windows
    cv2.imshow("Original Frame", frame)
    cv2.imshow("Grayscale", gray)
    cv2.imshow("Blurred", blurred)
    cv2.imshow("Thresholded", thresh)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
#out.release()
cv2.destroyAllWindows()
