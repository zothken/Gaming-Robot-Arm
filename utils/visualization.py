import cv2
import numpy as np
from tracker import CentroidTracker


# Displays multiple frames in separate windows.
def show_frames(frames_dict):

    for name, frame in frames_dict.items():
        cv2.imshow(name, frame)


# Draw circles and their center points around detected figures and display the total counts.
def draw_detections(frame, circles, colors, black_count, white_count):
    if circles is None or len(colors) == 0:
        return frame
     
    for ((x, y, r), color) in zip(circles, colors):
        cv2.circle(frame, (x, y), r, (255, 0, 0), 2)
        cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)

        # Label circles with color
        cv2.putText(frame, color, (x - 20, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
       
    # Show counts
    cv2.putText(frame, f"Black: {black_count}  White: {white_count}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    

def draw_ids(frame, tracker: CentroidTracker):
    for (objectID, (centroid, color)) in tracker.objects.items():
        smoothed = np.mean(tracker.positions[objectID], axis=0).astype(int)

        cv2.putText(frame, f"ID {objectID}", (smoothed[0] - 10, smoothed[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        cv2.circle(frame, tuple(smoothed), 4, (255, 255, 0), -1)
