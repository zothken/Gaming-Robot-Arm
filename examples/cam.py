import cv2

def print_camera_info(cap: cv2.VideoCapture) -> None:
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps and fps > 0:
        print(f"Resolution: {width}x{height}, FPS: {fps:.2f}")
    else:
        print(f"Resolution: {width}x{height}, FPS: (unknown)")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Could not open camera 0")

print_camera_info(cap)

while True:
    ok, frame = cap.read()
    if not ok:
        break
    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
