import cv2

def print_camera_info(cap: cv2.VideoCapture) -> None:
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps and fps > 0:
        print(f"Aufloesung: {width}x{height}, FPS: {fps:.2f}")
    else:
        print(f"Aufloesung: {width}x{height}, FPS: (unbekannt)")

cap = cv2.VideoCapture(1)
if not cap.isOpened():
    raise SystemExit("Konnte Kamera 0 nicht oeffnen.")

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, 30)

print_camera_info(cap)

while True:
    ok, frame = cap.read()
    if not ok:
        break
    cv2.imshow("Kamera", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
