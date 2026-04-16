import cv2

# Zeigt mehrere Bilder gleichzeitig in separaten Fenstern an.
def show_frames(frames_dict):
    for name, frame in frames_dict.items():
        cv2.imshow(name, frame)


# Zeichnet Kreise und Schwerpunkte erkannter Figuren und zeigt die Gesamtanzahl an.
def draw_detections(frame, circles, colors, black_count, white_count):
    if circles is None or len(colors) == 0:
        return frame

    for ((x, y, r), color) in zip(circles, colors):
        circle_color = (255, 0, 0) if color in {"weiss", "white"} else (144, 238, 144)
        cv2.circle(frame, (x, y), r, circle_color, 2)
        cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)

        # # Kreise mit Farbtext beschriften
        # cv2.putText(
        #     frame,
        #     color,
        #     (x - 20, y - 10),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.5,
        #     (0, 0, 255),
        #     1,
        # )

    # Anzahl der schwarzen und weissen Steine anzeigen
    cv2.putText(
        frame,
        f"Schwarz: {black_count}  Weiss: {white_count}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 0),
        2,
    )


def draw_assignment_labels(frame, assignments, *, font_scale: float = 0.6) -> None:
    """Zeichnet Brett-Labels fuer Zuordnungen mit Schwerpunkt- und Farbfeld."""
    for a in assignments:
        x, y = a["centroid"]
        lbl = a["label"]
        color_name = str(a.get("color", "")).lower()
        color = (0, 0, 0) if color_name in {"weiss", "white"} else (255, 255, 255)
        cv2.putText(
            frame,
            lbl,
            (int(x) - 15, int(y) - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            2,
            cv2.LINE_AA,
        )
