import cv2


# Displays multiple frames in separate windows.
def show_frames(frames_dict, delay=1):

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
