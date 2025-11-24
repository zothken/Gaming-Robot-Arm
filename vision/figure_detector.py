from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

from vision.visualization import draw_detections, draw_ids


def estimate_assign_distance(
    board_coords: Dict[str, Tuple[int, int]],
    *,
    factor: float = 0.35,
    min_px: float = 30.0,
    max_px: float = 140.0,
) -> float:
    """Leitet einen plausiblen Zuordnungs-Radius aus den Brettabstaenden ab."""
    if not board_coords:
        return max_px

    pts = np.array(list(board_coords.values()), dtype=np.float32)
    if len(pts) < 2:
        return max_px

    dists = np.linalg.norm(pts[None, :, :] - pts[:, None, :], axis=2)
    finite = dists[np.isfinite(dists) & (dists > 1e-6)]
    if finite.size == 0:
        return max_px

    median_spacing = float(np.median(finite))
    suggested = factor * median_spacing
    return float(min(max(suggested, min_px), max_px))


def assign_figures_to_board(
    centroids: Sequence[Tuple[int, int]],
    colors: Sequence[str],
    board_coords: Dict[str, Tuple[int, int]],
    max_dist_px: float | None = None,
) -> List[Dict[str, object]]:
    """
    Ordnet erkannte Figuren (Pixel-Koordinaten) robust den 24 Brettlabels zu.

    - Gated nearest-neighbour: Zuordnung nur, wenn Distanz < max_dist_px.
    - Ein Label wird hoechstens einmal vergeben (kleinster Fehler gewinnt).
    - Gibt eine Liste mit Label, Farbe, Pixel-Position und Distanz zurueck.
    """
    if len(centroids) != len(colors):
        raise ValueError("Centroids and colors must have the same length")

    if not centroids or not board_coords:
        return []

    labels = list(board_coords.keys())
    board_pts = np.array([board_coords[lbl] for lbl in labels], dtype=np.float32)
    dets = np.array(centroids, dtype=np.float32)

    # Falls kein Schwellwert uebergeben wurde, skaliere ihn mit dem Brettabstand
    if max_dist_px is None:
        max_dist_px = estimate_assign_distance(board_coords)

    # Distanzmatrix: [n_dets x 24]
    D = np.linalg.norm(dets[:, None, :] - board_pts[None, :, :], axis=2)

    assignments: List[Dict[str, object]] = []
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


def detect_figures(
    frame,
    tracker=None,
    board_coords: Dict[str, Tuple[int, int]] | None = None,
    max_assign_dist: float | None = None,
    return_assignments: bool = False,
    draw_assignments: bool = False,
):
    """
    Erkennt runde Figuren im Bild, klassifiziert sie nach Farbe und annotiert den Frame.

    Optional:
    - board_coords: dict A1-C8 -> (u,v), um die erkannten Kreise Feldern zuzuordnen.
    - return_assignments: liefert zusaetzlich eine Liste von Zuordnungen zurueck.
    - draw_assignments: schreibt die Feld-Labels in das Frame (ohne Tracker-ID).
    - max_assign_dist: Abstandsschwelle in Pixeln; None nutzt einen automatisch
      aus den Brettabstaenden abgeleiteten Wert.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Parameter zur Filterung von Konturen (Scheibengroesse)
    MIN_RADIUS = 20
    MAX_RADIUS = 30

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
                color = "black"
                black_count += 1
            else:
                color = "white"
                white_count += 1
            colors.append(color)

    if len(centroids) > 0:
        sorted_data = sorted(zip(centroids, colors), key=lambda c: (c[0][0], c[0][1]))
        centroids, colors = zip(*sorted_data)
        centroids, colors = list(centroids), list(colors)

    if tracker is not None:
        tracker.update(centroids, colors)

    assignments = []
    if board_coords:
        assignments = assign_figures_to_board(
            centroids,
            colors,
            board_coords,
            max_dist_px=max_assign_dist,
        )

        if draw_assignments and assignments:
            for a in assignments:
                x, y = a["centroid"]
                lbl = a["label"]
                color = (0, 0, 0) if a["color"] == "white" else (255, 255, 255)
                cv2.putText(
                    frame,
                    lbl,
                    (x - 15, y - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                    cv2.LINE_AA,
                )

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
