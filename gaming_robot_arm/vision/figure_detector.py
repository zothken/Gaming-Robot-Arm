from collections import Counter, deque
from contextlib import nullcontext
from typing import Any, Dict, List, Literal, Sequence, Tuple, overload

import cv2
import numpy as np

from gaming_robot_arm.config import CAMERA_INDEX, FRAME_RATE
from gaming_robot_arm.utils.logger import logger
from gaming_robot_arm.vision.recording import RecordingSession, get_effective_camera_fps, open_camera
from gaming_robot_arm.vision.visualization import draw_assignment_labels, draw_detections, draw_ids

# Temporärer Korrekturwert für die Brett-Zuordnung (Pixelraum).
ASSIGNMENT_Y_OFFSET_PX = 0.0

Assignment = Dict[str, object]
DetectionResult = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
DetectionResultWithAssignments = Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    List[Assignment],
]


def estimate_assign_distance(
    board_coords: Dict[str, Tuple[int, int]],
    *,
    factor: float = 0.55,
    min_px: float = 5.0,
    max_px: float = 250.0,
) -> float:
    """Leitet einen plausiblen Zuordnungs-Radius aus den Brettabstaenden ab.

    Nutzt die mediane *nearest-neighbour* Distanz der Brettpunkte als typische
    Feld-Abstands-Skala und nimmt davon einen Bruchteil als Zuordnungsschwelle.
    """
    if not board_coords:
        return max_px

    pts = np.array(list(board_coords.values()), dtype=np.float32)
    if len(pts) < 2:
        return max_px

    # Nearest-neighbour-Abstand pro Punkt.
    dists = np.linalg.norm(pts[None, :, :] - pts[:, None, :], axis=2)
    np.fill_diagonal(dists, np.inf)
    nn = np.min(dists, axis=1)
    nn = nn[np.isfinite(nn) & (nn > 1e-6)]
    if nn.size == 0:
        return max_px

    typical_spacing = float(np.median(nn))
    suggested = factor * typical_spacing
    # Maximal nicht groesser als die typische Nachbarschaftsdistanz (sonst wird "alles" gematcht).
    max_allowed = max(min(float(max_px), 0.9 * typical_spacing), float(min_px))
    return float(min(max(suggested, min_px), max_allowed))


def assign_figures_to_board(
    centroids: Sequence[Tuple[int, int]],
    colors: Sequence[str],
    board_coords: Dict[str, Tuple[int, int]],
    max_dist_px: float | None = None,
    labels_order: Sequence[str] | None = None,
) -> List[Assignment]:
    """
    Ordnet erkannte Figuren (Pixel-Koordinaten) robust den 24 Brettlabels zu.

    - Gated nearest-neighbour: Zuordnung nur, wenn Distanz < max_dist_px.
    - Ein Label wird hoechstens einmal vergeben (kleinster Fehler gewinnt).
    - Gibt eine Liste mit Label, Farbe, Pixel-Position und Distanz zurueck.
    """
    if len(centroids) != len(colors):
        raise ValueError("Centroids und Farben muessen die gleiche Laenge haben.")

    if not centroids or not board_coords:
        return []

    labels = list(labels_order) if labels_order else list(board_coords.keys())
    board_pts = np.array([board_coords[lbl] for lbl in labels], dtype=np.float32)
    dets = np.array(centroids, dtype=np.float32)

    # Falls kein Schwellwert uebergeben wurde, skaliere ihn mit dem Brettabstand
    if max_dist_px is None:
        max_dist_px = estimate_assign_distance(board_coords)

    # Distanzmatrix: [n_dets x 24]
    D = np.linalg.norm(dets[:, None, :] - board_pts[None, :, :], axis=2)

    assignments: List[Assignment] = []
    used_dets, used_labels = set(), set()

    while np.isfinite(D).any():
        i, j = np.unravel_index(np.argmin(D), D.shape)
        dist = D[i, j]
        if dist > max_dist_px:
            break  # keine plausiblen Matches mehr

        if i in used_dets or j in used_labels:
            D[i, j] = np.inf
            continue

        assignments.append(
            {
                "label": labels[j],
                "centroid": (int(dets[i, 0]), int(dets[i, 1])),
                "color": colors[i],
                "dist_px": float(dist),
            }
        )

        used_dets.add(i)
        used_labels.add(j)
        D[i, :] = np.inf
        D[:, j] = np.inf

    # Fuer reproduzierbare Weiterverarbeitung nach Label sortieren
    assignments.sort(key=lambda a: a["label"])
    return assignments


class AssignmentStabilizer:
    """Zaehlt, wie haeufig Labels erkannt werden, und filtert nur stabile Zuordnungen."""

    def __init__(self, labels: Sequence[str], window: int = 60) -> None:
        self.labels = list(labels)
        self.window = max(1, window)
        self.present = {lbl: deque(maxlen=self.window) for lbl in self.labels}
        self.centroids = {lbl: deque(maxlen=self.window) for lbl in self.labels}
        self.colors = {lbl: deque(maxlen=self.window) for lbl in self.labels}

    def update(self, assignments: Sequence[Assignment]) -> None:
        detected = {a["label"]: a for a in assignments}
        for lbl in self.labels:
            if lbl in detected:
                a = detected[lbl]
                self.present[lbl].append(1)
                self.centroids[lbl].append(np.array(a["centroid"], dtype=np.float32))
                self.colors[lbl].append(a["color"])
            else:
                self.present[lbl].append(0)

    def stable_assignments(
        self,
        *,
        min_ratio: float = 0.5,
        min_samples: int = 10,
    ) -> List[Assignment]:
        stable: List[Assignment] = []
        for lbl in self.labels:
            hist = self.present[lbl]
            if len(hist) < max(1, min_samples):
                continue
            if hist[-1] != 1:
                # Nur Labels werten, die im aktuellen Frame sichtbar sind.
                continue
            ratio = float(sum(hist)) / len(hist)
            if ratio < min_ratio or len(self.centroids[lbl]) == 0:
                continue
            # Nutze den aktuellsten Schwerpunkt (keine Mittelung ueber vergangene Positionen),
            # damit ein Verschieben sofort auf das neue Feld springt.
            centroid = tuple(np.round(self.centroids[lbl][-1]).astype(int))
            recent_colors = self.colors[lbl]
            color = recent_colors[-1] if recent_colors else Counter(self.colors[lbl]).most_common(1)[0][0]
            stable.append(
                {"label": lbl, "centroid": centroid, "color": color, "presence_ratio": ratio}
            )
        stable.sort(key=lambda a: a["label"])
        return stable


@overload
def detect_figures(
    frame: np.ndarray,
    tracker: Any = None,
    board_coords: Dict[str, Tuple[int, int]] | None = None,
    max_assign_dist: float | None = None,
    labels_order: Sequence[str] | None = None,
    return_assignments: Literal[False] = False,
    draw_assignments: bool = False,
    debug_assignments: bool = False,
) -> DetectionResult: ...


@overload
def detect_figures(
    frame: np.ndarray,
    tracker: Any = None,
    board_coords: Dict[str, Tuple[int, int]] | None = None,
    max_assign_dist: float | None = None,
    labels_order: Sequence[str] | None = None,
    return_assignments: Literal[True] = ...,
    draw_assignments: bool = False,
    debug_assignments: bool = False,
) -> DetectionResultWithAssignments: ...


def detect_figures(
    frame: np.ndarray,
    tracker: Any = None,
    board_coords: Dict[str, Tuple[int, int]] | None = None,
    max_assign_dist: float | None = None,
    labels_order: Sequence[str] | None = None,
    return_assignments: bool = False,
    draw_assignments: bool = False,
    debug_assignments: bool = False,
) -> DetectionResult | DetectionResultWithAssignments:
    """
    Erkennt runde Figuren im Bild, klassifiziert sie nach Farbe und annotiert den Frame.

    Optional:
    - board_coords: dict A1-C8 -> (u,v), um die erkannten Kreise Feldern zuzuordnen.
    - return_assignments: liefert zusaetzlich eine Liste von Zuordnungen zurueck.
    - draw_assignments: schreibt die Feld-Labels in das Frame (ohne Tracker-ID).
    - max_assign_dist: Abstandsschwelle in Pixeln; None nutzt einen automatisch
      aus den Brettabstaenden abgeleiteten Wert.
    - labels_order: optionale feste Label-Reihenfolge (z.B. sortiert), falls board_coords nicht geordnet ist.
    - debug_assignments: Loggt erkannte Kreise, naechste Labels und Zuordnungen.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Parameter zur Filterung von Konturen (Scheibengroesse)
    MIN_RADIUS = 3
    MAX_RADIUS = 20

    # Sicherstellen, dass block_size ungerade und mindestens 3 ist
    THRESH_BLOCK = 49
    if THRESH_BLOCK % 2 == 0:
        THRESH_BLOCK += 1
    THRESH_C = 10

    # Adaptives Schwellwertverfahren
    thresh = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        THRESH_BLOCK,
        THRESH_C,
    )

    centroids = []
    colors = []
    black_count, white_count = 0, 0

    # Kreisdetektion mit Hough-Transformation
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=20,
        param1=80,
        param2=30,
        minRadius=MIN_RADIUS,
        maxRadius=MAX_RADIUS,
    )

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")

        for (x, y, r) in circles:
            center = (x, y)
            centroids.append(center)

            # Maske fuer lokale Helligkeitsmessung erzeugen
            mask = np.zeros(blurred.shape, dtype="uint8")
            cv2.circle(mask, center, r, 255, -1)
            mean_val = cv2.mean(blurred, mask=mask)[0]

            if mean_val < 128:
                color = "schwarz"
                black_count += 1
            else:
                color = "weiss"
                white_count += 1
            colors.append(color)

    if len(centroids) > 0:
        sorted_data = sorted(zip(centroids, colors), key=lambda c: (c[0][0], c[0][1]))
        centroids, colors = zip(*sorted_data)
        centroids, colors = list(centroids), list(colors)

    if tracker is not None:
        tracker.update(centroids, colors)

    assignments: List[Assignment] = []
    if board_coords:
        assignments = assign_figures_to_board(
            centroids,
            colors,
            board_coords,
            max_dist_px=max_assign_dist,
            labels_order=labels_order,
        )

        if debug_assignments:
            labels = list(board_coords.keys())
            coords_arr = np.array(list(board_coords.values()), dtype=np.float32)
            if centroids:
                nearest = []
                for c in centroids:
                    dists = np.linalg.norm(coords_arr - np.array(c, dtype=np.float32), axis=1)
                    idx = int(np.argmin(dists))
                    nearest.append((labels[idx], float(dists[idx])))
                logger.info(
                    "Detektions-Debug: %s Kreise, Farben=%s, naechste Labels=%s",
                    len(centroids),
                    colors,
                    nearest,
                )
            else:
                logger.info("Detektions-Debug: keine Kreise erkannt.")

            if assignments:
                logger.info("Detektions-Debug: Zuordnungen=%s", assignments)
            else:
                logger.info(
                    "Detektions-Debug: keine Zuordnungen bei max_dist=%.1fpx",
                    max_assign_dist if max_assign_dist is not None else float("nan"),
                )

        if draw_assignments and assignments:
            draw_assignment_labels(frame, assignments, font_scale=0.6)

    if circles is None or len(centroids) == 0:
        if return_assignments:
            return frame, gray, blurred, thresh, assignments
        return frame, gray, blurred, thresh

    draw_detections(frame, circles, colors, black_count, white_count)
    # if tracker is not None:
    #     draw_ids(frame, tracker)

    if return_assignments:
        return frame, gray, blurred, thresh, assignments
    return frame, gray, blurred, thresh


def run_live_assignment_test(
    *,
    camera_index: int = CAMERA_INDEX,
    max_assign_dist: float | None = None,
    window_name: str = "Figuren-Detektor (A1-C8)",
    use_stabilizer: bool = True,
    min_ratio: float = 0.5,
    min_samples: int | None = None,
    debug_assignments: bool = False,
) -> None:
    """
    Live-Test: markiert erkannte Figuren und versieht sie mit Feld-Labels (A1-C8).

    Erfordert Brett-Pixel in gaming_robot_arm/calibration/cam_to_robot_homography.json (Kalibrierung Option 1).
    Druecke 'q', um die Anzeige zu beenden. Gewertet werden nur Figuren, die in
    mindestens min_ratio der letzten Sekunde erkannt wurden (nutzt FRAME_RATE).
    Wenn use_stabilizer=False, werden die Roh-Zuordnungen je Frame gezeichnet.
    """
    try:
        from gaming_robot_arm.calibration.calibration import load_board_pixels
    except Exception:
        logger.exception("Kalibrations-Modul konnte nicht importiert werden.")
        return

    try:
        with open_camera(camera_index=camera_index) as cam:
            fps = get_effective_camera_fps(cam, configured_fps=FRAME_RATE)
            window_frames = max(1, int(round(fps)))
            min_samples = window_frames if min_samples is None else max(1, int(min_samples))

            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            # Ersten Frame lesen, um die Aufloesung zu kennen (Kalibrierung kann auf anderer Aufloesung basiert haben).
            ret0, frame0 = cam.read()
            if not ret0:
                logger.error("Kameraframe konnte nicht gelesen werden, breche ab.")
                return

            try:
                board_coords = load_board_pixels(frame_size=(frame0.shape[1], frame0.shape[0]))
            except FileNotFoundError:
                logger.error(
                    "Keine Brett-Pixel gefunden. Bitte `python -m gaming_robot_arm.calibration.calibration` "
                    "ausfuehren (Option 1)."
                )
                return
            except Exception:
                logger.exception("Fehler beim Laden der Brett-Pixel.")
                return

            labels_order = sorted(board_coords.keys())
            stabilizer = AssignmentStabilizer(labels_order, window=window_frames) if use_stabilizer else None
            assign_dist = max_assign_dist or estimate_assign_distance(board_coords)

            # Wenn die Kalibrierung klar nicht zur Aufloesung passt, ist Zuordnung sehr wahrscheinlich 0.
            xs = [p[0] for p in board_coords.values()]
            ys = [p[1] for p in board_coords.values()]
            inside = sum(
                1
                for x, y in board_coords.values()
                if 0 <= x < frame0.shape[1] and 0 <= y < frame0.shape[0]
            )
            inside_ratio = inside / max(1, len(board_coords))
            if inside_ratio < 0.8:
                logger.warning(
                    "Brett-Pixel passen evtl. nicht zur Kamera-Aufloesung (%sx%s): inside_ratio=%.2f (bbox=%sx%s..%sx%s).",
                    frame0.shape[1],
                    frame0.shape[0],
                    inside_ratio,
                    min(xs),
                    min(ys),
                    max(xs),
                    max(ys),
                )

            logger.info(
                "Starte Live-Test fuer Figuren-Zuordnung (max_dist=%.1fpx). 'q' beendet.",
                assign_dist,
            )

            # Ersten Frame nicht verwerfen.
            pending_frame = frame0

            while True:
                if pending_frame is not None:
                    frame = pending_frame
                    pending_frame = None
                else:
                    ret, frame = cam.read()
                    if not ret:
                        logger.warning("Kameraframe konnte nicht gelesen werden, breche ab.")
                        break

                processed, _, _, _, assignments = detect_figures(
                    frame,
                    board_coords=board_coords,
                    max_assign_dist=assign_dist,
                    labels_order=labels_order,
                    draw_assignments=False,
                    return_assignments=True,
                    debug_assignments=debug_assignments,
                )

                if use_stabilizer and stabilizer is not None:
                    stabilizer.update(assignments)
                    stable = stabilizer.stable_assignments(
                        min_ratio=min_ratio,
                        min_samples=min_samples,
                    )
                    draw_assignment_labels(processed, stable, font_scale=0.7)
                    overlay = (
                        f"{len(stable)} stabile Zuordnungen  "
                        f"| Roh: {len(assignments)}  |  q=beenden"
                    )
                else:
                    draw_assignment_labels(processed, assignments, font_scale=0.6)
                    overlay = f"{len(assignments)} Zuordnungen  |  q=beenden"
                cv2.putText(
                    processed,
                    overlay,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow(window_name, processed)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    logger.info("Beendet durch Tastendruck.")
                    break
    finally:
        cv2.destroyAllWindows()


def detect_board_assignments(
    board_pixels: Dict[str, Tuple[float, float]],
    attempts: int = 6,
    session: RecordingSession | None = None,
    labels_order: Sequence[str] | None = None,
    camera_index: int = CAMERA_INDEX,
    *,
    debug_assignments: bool = False,
) -> List[Dict[str, object]]:
    """
    Nimmt mehrere Bilder auf und versucht, Figuren robust den Brettpositionen zuzuordnen.

    - Nutzt einen adaptiven Zuordnungsradius basierend auf den Brettabstaenden.
    - Erhoeht schrittweise die Distanzschwelle, falls keine Zuordnung gelingt.
    - Kann optional in eine laufende RecordingSession mitschreiben.
    - Ein Label zaehlt nur, wenn es in mindestens der Haelfte der betrachteten Frames
      innerhalb der letzten Sekunde erkannt wurde (FRAME_RATE bestimmt die Fensterlaenge).
    """
    assignment_board_pixels = (
        {lbl: (float(u), float(v) + ASSIGNMENT_Y_OFFSET_PX) for lbl, (u, v) in board_pixels.items()}
        if ASSIGNMENT_Y_OFFSET_PX
        else board_pixels
    )
    labels_order = labels_order or sorted(board_pixels.keys())
    frames_seen = 0

    base_dist = estimate_assign_distance(board_pixels)
    dist_steps = [
        base_dist,
        base_dist * 1.1,
        base_dist * 1.2,
        base_dist * 1.3,
        base_dist * 1.5,
        base_dist * 1.8,
    ]

    try:
        with (
            open_camera(camera_index=camera_index) if session is None else nullcontext(session.camera)
        ) as cam:
            fps = get_effective_camera_fps(cam, configured_fps=FRAME_RATE)
            window_frames = max(1, int(round(fps)))
            stabilizer = AssignmentStabilizer(labels_order, window=window_frames)
            frames_to_capture = max(window_frames, attempts)
            frame_check_done = False

            for idx in range(frames_to_capture):
                ret, frame = cam.read()
                if not ret:
                    logger.warning("Kameraframe konnte nicht gelesen werden (Versuch %s).", idx + 1)
                    continue
                if session is not None:
                    session.write(frame)

                if not frame_check_done:
                    h, w = frame.shape[:2]
                    xs = [p[0] for p in board_pixels.values()]
                    ys = [p[1] for p in board_pixels.values()]
                    min_x, max_x = min(xs), max(xs)
                    min_y, max_y = min(ys), max(ys)

                    inside = sum(1 for x, y in board_pixels.values() if 0 <= x < w and 0 <= y < h)
                    inside_ratio = inside / max(1, len(board_pixels))
                    if inside_ratio < 0.8:
                        logger.warning(
                            "Kalibrierung passt evtl. nicht zur Kamera-Aufloesung (%sx%s): inside_ratio=%.2f (bbox=%sx%s..%sx%s). "
                            "Kalibrierung ggf. bei gleicher Aufloesung wiederholen oder Kamera-Aufloesung angleichen.",
                            w,
                            h,
                            inside_ratio,
                            min_x,
                            min_y,
                            max_x,
                            max_y,
                        )
                    else:
                        bbox_w = max_x - min_x
                        bbox_h = max_y - min_y
                        logger.debug(
                            "Board-Pixel im Frame: inside_ratio=%.2f, bbox=%.0fx%.0f (%.0f%% x %.0f%% vom Frame).",
                            inside_ratio,
                            bbox_w,
                            bbox_h,
                            100.0 * bbox_w / max(1.0, float(w)),
                            100.0 * bbox_h / max(1.0, float(h)),
                        )
                    frame_check_done = True

                frames_seen += 1
                try:
                    dist = dist_steps[idx] if idx < len(dist_steps) else dist_steps[-1]
                    _, _, _, _, assignments = detect_figures(
                        frame,
                        board_coords=assignment_board_pixels,
                        max_assign_dist=dist,
                        return_assignments=True,
                        debug_assignments=debug_assignments,
                        labels_order=labels_order,
                        draw_assignments=False,
                    )
                except Exception:
                    logger.exception("Fehler bei der Zuordnung der Figuren zum Brett.")
                    continue

                stabilizer.update(assignments)

    except RuntimeError as exc:
        logger.warning("Kamera konnte nicht geoeffnet werden: %s", exc)
        return []

    if frames_seen < window_frames:
        return []

    stable = stabilizer.stable_assignments(min_ratio=0.5, min_samples=window_frames)
    return stable


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Live-Vorschau fuer Figuren-Detektion mit Brett-Labels.",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=CAMERA_INDEX,
        help="Kamera-Index (Standard: 0).",
    )
    parser.add_argument(
        "--max-assign-dist",
        type=float,
        default=None,
        help="Maximaler Pixel-Abstand fuer Zuordnungen (Standard: automatisch).",
    )
    parser.add_argument(
        "--window-name",
        type=str,
        default="Figuren-Detektor (A1-C8)",
        help="Fenstertitel fuer die Live-Ansicht.",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Roh-Zuordnungen pro Frame statt stabiler Labels anzeigen.",
    )
    parser.add_argument(
        "--min-ratio",
        type=float,
        default=0.5,
        help="Mindest-Anteil fuer stabile Zuordnung (Standard: 0.5).",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=None,
        help="Minimale Anzahl Frames fuer stabile Zuordnung (Standard: ~1s Kameraframe-Rate).",
    )
    parser.add_argument(
        "--debug-assignments",
        action="store_true",
        help="Loggt Naehe/Zuordnungen fuer Debugging (sehr gespraechig).",
    )
    args = parser.parse_args()

    run_live_assignment_test(
        camera_index=args.camera_index,
        max_assign_dist=args.max_assign_dist,
        window_name=args.window_name,
        use_stabilizer=not args.raw,
        min_ratio=args.min_ratio,
        min_samples=args.min_samples,
        debug_assignments=args.debug_assignments,
    )


if __name__ == "__main__":
    main()
