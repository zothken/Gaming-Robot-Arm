import cv2
import numpy as np

def detect_board_positions(frame):
    """
    Detects all 24 positions on a Nine Men's Morris board, robust to rotation/shift.

    Args:
        frame: BGR image of the board

    Returns:
        positions: List of (x, y) coordinates in the original image
        annotated_frame: Frame with detected board and positions drawn
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

    # Detect lines
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)
    annotated_frame = frame.copy()
    if lines is not None:
        for l in lines:
            x1, y1, x2, y2 = l[0]
            cv2.line(annotated_frame, (x1, y1), (x2, y2), (0,255,0), 1)

    # Use contours to find the outer board boundary
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return [], annotated_frame

    largest = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest)
    box = cv2.boxPoints(rect)
    box = box.astype(np.int32)

    # Draw the boundary for debugging
    cv2.polylines(annotated_frame, [box], isClosed=True, color=(255,0,0), thickness=2)

    # Sort box points: top-left, top-right, bottom-right, bottom-left
    def sort_box(pts):
        pts = pts[np.argsort(pts[:,1])]  # sort by y
        top = pts[:2]
        bottom = pts[2:]
        top = top[np.argsort(top[:,0])]
        bottom = bottom[np.argsort(bottom[:,0])]
        return np.array([top[0], top[1], bottom[1], bottom[0]])

    box = sort_box(box)

    # Normalize to a standard 500x500 board
    dst = np.array([[0,0],[500,0],[500,500],[0,500]], dtype="float32")
    M = cv2.getPerspectiveTransform(box.astype("float32"), dst)
    inv_M = cv2.getPerspectiveTransform(dst, box.astype("float32"))

    # Define 24 positions in normalized coordinates
    # Outer, middle, inner squares
    positions_norm = []
    squares = [0, 250, 500]  # normalized coordinates
    for y in squares:
        for x in squares:
            positions_norm.append([x, y])

    # Midpoints of edges
    mid_squares = [125, 375]
    for y in squares:
        for x in mid_squares:
            positions_norm.append([x, y])
    for y in mid_squares:
        for x in squares:
            positions_norm.append([x, y])

    positions_norm = np.array(positions_norm, dtype="float32")

    # Transform back to original image coordinates
    positions_orig = cv2.perspectiveTransform(positions_norm.reshape(-1,1,2), inv_M)
    positions_orig = positions_orig.reshape(-1,2).astype(int)

    # Draw positions for debugging
    for x, y in positions_orig:
        cv2.circle(annotated_frame, (x, y), 5, (0,0,255), -1)

    return positions_orig.tolist(), annotated_frame


# -------------------------------
# Standalone debug / test mode
# -------------------------------
if __name__ == "__main__":
    import sys
    import cv2

    test_image_path = r"C:\Users\nando\OneDrive\Documents\Uni\Bachelor\PP-BA\Graphics\Annotated_Board.png"

    frame = cv2.imread(test_image_path)
    if frame is None:
        print("No input image specified, using camera...")
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError("Could not read frame from camera.")

    positions, annotated = detect_board_positions(frame)
    print("Detected positions:", positions)

    cv2.imshow("Annotated Board", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()