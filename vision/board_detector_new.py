# Minimaler Board-Detector: nutzt nur das binäre Bild (bw_block=15, bw_C=10) als Basis.
# Optional: Live-Slider für die wenigen genutzten Hough/Orientierungs-Parameter.

from dataclasses import dataclass
import cv2
import numpy as np

# Feste Standardparameter (nur diese werden via Slider angepasst)
DEFAULT_PARAMS = {
    "hough_thresh": 60,
    "min_len_pct": 10,  # Prozent von max(H, W)
    "max_gap": 12,
    "angle_tol": 10,
}


@dataclass(frozen=True)
class Line:
    orientation: str  # 'H' oder 'V'
    slope: float
    intercept: float

    @property
    def anchor(self) -> float:
        return self.intercept


def angle_deg(x1, y1, x2, y2) -> float:
    return float(np.degrees(np.arctan2((y2 - y1), (x2 - x1))))


def rotate_points(pts, center, angle_deg):
    """Rotiert Punkte um center um angle_deg."""
    if len(pts) == 0:
        return np.empty((0, 2), dtype=np.float32)
    pts_arr = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    cx, cy = center
    ang = np.deg2rad(angle_deg)
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]], dtype=np.float32)
    shifted = pts_arr - np.array([cx, cy], dtype=np.float32)
    rotated = shifted @ R.T + np.array([cx, cy], dtype=np.float32)
    return rotated


def rotate_segments(segments, center, angle_deg):
    """Rotiert Segmentendpunkte um center um angle_deg."""
    if segments is None or len(segments) == 0:
        return np.empty((0, 4), dtype=np.float32)
    pts = np.asarray(segments, dtype=np.float32).reshape(-1, 2)
    rotated = rotate_points(pts, center, angle_deg)
    return rotated.reshape(-1, 4)


def estimate_board_rotation(segments):
    """Schaetzt dominante Brettrotation via Double-Angle Mittelwert, gewichtet nach Segmentlaenge."""
    if segments is None or len(segments) == 0:
        return 0.0
    angles = []
    weights = []
    for (x1, y1, x2, y2) in segments:
        ang = angle_deg(x1, y1, x2, y2)
        ang = ((ang + 90.0) % 180.0) - 90.0  # [-90, 90)
        w = np.hypot(x2 - x1, y2 - y1)
        if w <= 1e-3:
            continue
        angles.append(ang)
        weights.append(w)
    if not weights:
        return 0.0
    ang_rad = np.deg2rad(np.array(angles, dtype=np.float32))
    weights = np.array(weights, dtype=np.float32)
    c = np.sum(weights * np.cos(2.0 * ang_rad))
    s = np.sum(weights * np.sin(2.0 * ang_rad))
    if abs(c) < 1e-6 and abs(s) < 1e-6:
        return 0.0
    dominant = 0.5 * np.arctan2(s, c)
    return float(np.rad2deg(dominant))


def separate_segments_by_orientation(segments, tolerance_deg=10.0):
    horizontals, verticals = [], []
    for (x1, y1, x2, y2) in segments:
        a = (angle_deg(x1, y1, x2, y2) + 180.0) % 180.0
        if min(abs(a - 0), abs(a - 180)) <= tolerance_deg:
            horizontals.append((x1, y1, x2, y2))
        elif min(abs(a - 90), abs(a - 270)) <= tolerance_deg:
            verticals.append((x1, y1, x2, y2))
    return horizontals, verticals


def fit_line_from_segments(segments, vertical=False):
    pts = []
    for (x1, y1, x2, y2) in segments:
        pts.append([x1, y1])
        pts.append([x2, y2])
    pts = np.array(pts, dtype=np.float32)
    if not vertical:
        X = np.vstack([pts[:, 0], np.ones(len(pts))]).T
        m, b = np.linalg.lstsq(X, pts[:, 1], rcond=None)[0]
        return Line('H', float(m), float(b))
    Y = np.vstack([pts[:, 1], np.ones(len(pts))]).T
    m, b = np.linalg.lstsq(Y, pts[:, 0], rcond=None)[0]
    return Line('V', float(m), float(b))


def draw_infinite_line(img, line, color, thickness=2, rotation_deg=0.0, center=None):
    h, w = img.shape[:2]
    m, b = line.slope, line.intercept
    if line.orientation == 'H':
        x1, x2 = 0, w - 1
        y1 = float(m * x1 + b)
        y2 = float(m * x2 + b)
        p1, p2 = (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2)))
    else:
        y1, y2 = 0, h - 1
        x1 = float(m * y1 + b)
        x2 = float(m * y2 + b)
        p1, p2 = (x1, y1), (x2, y2)
    pts = np.array([p1, p2], dtype=np.float32)
    if center is not None and abs(rotation_deg) > 1e-3:
        pts = rotate_points(pts, center, rotation_deg)
    p1 = tuple(np.round(pts[0]).astype(int))
    p2 = tuple(np.round(pts[1]).astype(int))
    cv2.line(img, p1, p2, color, thickness)


def line_anchor_value(line):
    return line.intercept


def ensure_lines(lines, expected=7, axis='H'):
    orientation = 'H' if axis == 'H' else 'V'
    sorted_lines = sorted(lines, key=line_anchor_value)
    if len(sorted_lines) == expected:
        return sorted_lines
    if not sorted_lines:
        return [Line(orientation, 0.0, float(i * 50)) for i in range(expected)]

    # Wenn genügend Linien vorhanden sind, wähle gleichmäßig verteilte Anker
    if len(sorted_lines) > expected:
        indices = [int(group[len(group) // 2]) for group in np.array_split(np.arange(len(sorted_lines)), expected)]
        return [sorted_lines[idx] for idx in indices]

    # Zu wenige Linien: fehlende Linien nach außen extrapolieren (stabiler als Einfügen in der Mitte)
    anchors = np.array([ln.anchor for ln in sorted_lines], dtype=float)
    if len(anchors) >= 2:
        spacing = np.median(np.diff(anchors))
        if spacing <= 1e-3:
            spacing = 40.0
    else:
        spacing = 40.0

    lines_adjusted = sorted_lines.copy()
    # extrapoliere abwechselnd nach außen (oben/unten bzw. links/rechts)
    while len(lines_adjusted) < expected:
        anchors = np.array([ln.anchor for ln in lines_adjusted], dtype=float)
        anchors_sorted = np.sort(anchors)
        lowest, highest = anchors_sorted[0], anchors_sorted[-1]
        slope_ref = float(lines_adjusted[0].slope if abs(lines_adjusted[0].slope) < 1e3 else 0.0)
        # füge unten/links hinzu
        if len(lines_adjusted) < expected:
            new_anchor_low = lowest - spacing
            lines_adjusted.append(Line(orientation, slope_ref, float(new_anchor_low)))
        # füge oben/rechts hinzu
        if len(lines_adjusted) < expected:
            new_anchor_high = highest + spacing
            lines_adjusted.append(Line(orientation, slope_ref, float(new_anchor_high)))
        lines_adjusted = sorted(lines_adjusted, key=line_anchor_value)

    return lines_adjusted


def intersect(hline, vline):
    mh, bh = hline.slope, hline.intercept
    mv, bv = vline.slope, vline.intercept
    A = 1.0 - mh * mv
    B = mh * bv + bh
    if abs(A) < 1e-9:
        y = bh
        x = mv * y + bv
    else:
        y = B / A
        x = mv * y + bv
    return int(round(x)), int(round(y))


def detect_board_positions(frame_bgr, debug=True, return_bw=False, params=None):
    H_img, W_img = frame_bgr.shape[:2]
    annotated = frame_bgr.copy()
    p = DEFAULT_PARAMS.copy()
    if params:
        p.update(params)

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Fest: bw_block=15, bw_C=10
    bw = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 10
    )
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    edges = cv2.Canny(bw, 50, 150, apertureSize=3)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    min_len = max(10, int(max(W_img, H_img) * p["min_len_pct"] * 0.01))
    max_gap = max(0, int(p["max_gap"]))
    hough_thresh = max(1, int(p["hough_thresh"]))

    segs = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=hough_thresh, minLineLength=min_len, maxLineGap=max_gap
    )
    if segs is None:
        if debug:
            print("⚠️ Keine Linien erkannt.")
        return ([], annotated, bw) if return_bw else ([], annotated)

    segments = segs[:, 0].astype(np.float32)
    for (x1, y1, x2, y2) in segments:
        cv2.line(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)

    image_center = (0.5 * W_img, 0.5 * H_img)
    board_rotation = estimate_board_rotation(segments)
    if board_rotation > 45.0:
        board_rotation -= 90.0
    elif board_rotation < -45.0:
        board_rotation += 90.0
    if debug:
        cv2.putText(annotated, f"rot {board_rotation:+.1f} deg", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    segments_aligned = rotate_segments(segments, image_center, -board_rotation)

    horiz_segs, vert_segs = separate_segments_by_orientation(segments_aligned, tolerance_deg=float(p["angle_tol"]))

    def anchors_midpoints(segs, vertical=False):
        return [0.5 * (s[1] + s[3]) for s in segs] if not vertical else [0.5 * (s[0] + s[2]) for s in segs]

    y_mids = anchors_midpoints(horiz_segs, vertical=False)
    x_mids = anchors_midpoints(vert_segs, vertical=True)

    def peak_centers(vals, bins, want_k=7, nms=2):
        if len(vals) == 0:
            return []
        hist, edges_hist = np.histogram(vals, bins=bins)
        sorted_idx = np.argsort(hist)[::-1]
        suppressed = np.zeros_like(hist, dtype=bool)
        peaks = []
        for idx in sorted_idx:
            if hist[idx] <= 0 or suppressed[idx]:
                continue
            peaks.append(idx)
            if len(peaks) == want_k:
                break
            lo, hi = max(0, idx - nms), min(len(hist), idx + nms + 1)
            suppressed[lo:hi] = True
        if not peaks:
            return []
        peaks = np.array(peaks, dtype=int)
        centers = 0.5 * (edges_hist[peaks] + edges_hist[peaks + 1])
        centers.sort()
        return centers.tolist()

    H_centers = peak_centers(y_mids, bins=max(14, H_img // 20), want_k=7, nms=2)
    V_centers = peak_centers(x_mids, bins=max(14, W_img // 20), want_k=7, nms=2)

    def group_by_peak(segments, centers, vertical=False):
        groups = [[] for _ in centers]
        for seg in segments:
            if not centers:
                continue
            val = 0.5 * (seg[1] + seg[3]) if not vertical else 0.5 * (seg[0] + seg[2])
            i = int(np.argmin([abs(val - c) for c in centers]))
            groups[i].append(seg)
        return groups

    H_groups = group_by_peak(horiz_segs, H_centers, vertical=False)
    V_groups = group_by_peak(vert_segs, V_centers, vertical=True)

    H_lines = []
    for g in H_groups:
        if not g:
            continue
        ln = fit_line_from_segments(g, vertical=False)
        H_lines.append(ln)
        draw_infinite_line(annotated, ln, (255, 255, 0), 1, rotation_deg=board_rotation, center=image_center)

    V_lines = []
    for g in V_groups:
        if not g:
            continue
        ln = fit_line_from_segments(g, vertical=True)
        V_lines.append(ln)
        draw_infinite_line(annotated, ln, (255, 255, 0), 1, rotation_deg=board_rotation, center=image_center)

    H_lines = ensure_lines(sorted(H_lines, key=line_anchor_value), expected=7, axis='H')
    V_lines = ensure_lines(sorted(V_lines, key=line_anchor_value), expected=7, axis='V')

    for ln in H_lines:
        draw_infinite_line(annotated, ln, (255, 0, 0), 2, rotation_deg=board_rotation, center=image_center)
    for ln in V_lines:
        draw_infinite_line(annotated, ln, (255, 0, 0), 2, rotation_deg=board_rotation, center=image_center)

    outer = (0, 6)
    middle = (1, 5)
    inner = (2, 4)
    center_idx = 3
    iC = center_idx

    def quad_positions(iT, iB, iL, iR):
        return [
            intersect(H_lines[iT], V_lines[iL]),
            intersect(H_lines[iT], V_lines[iC]),
            intersect(H_lines[iT], V_lines[iR]),
            intersect(H_lines[iC], V_lines[iR]),
            intersect(H_lines[iB], V_lines[iR]),
            intersect(H_lines[iB], V_lines[iC]),
            intersect(H_lines[iB], V_lines[iL]),
            intersect(H_lines[iC], V_lines[iL]),
        ]

    coords = {}
    for label, (iT, iB, iL, iR) in zip(
        ["A", "B", "C"],
        [outer + (outer[0], outer[1]), middle + (middle[0], middle[1]), inner + (inner[0], inner[1])],
    ):
        pts = quad_positions(iT, iB, iL, iR)
        for idx, (x, y) in enumerate(pts, start=1):
            coords[f"{label}{idx}"] = (int(x), int(y))

    # Punkte in Original-Koordinaten zurückrotieren
    def rotate_coords_dict(coords_dict, angle_deg, center):
        if not coords_dict:
            return {}
        pts = np.array(list(coords_dict.values()), dtype=np.float32)
        rotated = rotate_points(pts, center, angle_deg)
        keys = list(coords_dict.keys())
        return {k: (int(round(p[0])), int(round(p[1]))) for k, p in zip(keys, rotated)}

    coords_rot = rotate_coords_dict(coords, board_rotation, image_center)

    def draw_labels(frame, coords_dict):
        out = frame.copy()
        for key, (x, y) in coords_dict.items():
            cv2.circle(out, (x, y), 6, (0, 0, 255), -1)
            cv2.putText(out, key, (x - 20, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 164, 0), 2, cv2.LINE_AA)
        return out

    annotated = draw_labels(annotated, coords_rot)
    if debug:
        print(f"{len(coords)} Brettpositionen (sollte 24 sein)")

    positions_rot = rotate_points(list(coords.values()), image_center, board_rotation)
    positions = [(int(round(x)), int(round(y))) for x, y in positions_rot]
    if return_bw:
        return positions, annotated, bw
    return positions, annotated


if __name__ == "__main__":
    try:
        from vision.recording import open_camera
    except ModuleNotFoundError:
        import sys
        from pathlib import Path

        project_root = Path(__file__).resolve().parents[1]
        if str(project_root) not in sys.path:
            sys.path.append(str(project_root))
        from vision.recording import open_camera

    import argparse

    parser = argparse.ArgumentParser(description="Board detection (simple, BW-based).")
    parser.add_argument("--image", type=str, help="Optionaler Pfad zu einem Testbild.")
    parser.add_argument("--single-frame", action="store_true", help="Nur ersten Kameraframe auswerten.")
    args = parser.parse_args()

    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            raise RuntimeError(f"Bild nicht gefunden: {args.image}")
        pts, vis, bw = detect_board_positions(img, debug=True, return_bw=True)
        print(f"Gefundene Brett-Positionen: {len(pts)}")
        cv2.imshow("Board Detection Debug (new)", vis)
        cv2.imshow("BW Threshold", bw)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        win = "Board Detection Debug (new)"
        bw_win = "BW Threshold"
        ui = "Board Params"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.namedWindow(bw_win, cv2.WINDOW_NORMAL)
        cv2.namedWindow(ui, cv2.WINDOW_NORMAL)

        def add_slider(name, init, maxval):
            cv2.createTrackbar(name, ui, init, maxval, lambda v: None)
            cv2.setTrackbarPos(name, ui, init)

        add_slider("hough_thresh", DEFAULT_PARAMS["hough_thresh"], 120)
        add_slider("min_len_pct", DEFAULT_PARAMS["min_len_pct"], 40)
        add_slider("max_gap", DEFAULT_PARAMS["max_gap"], 50)
        add_slider("angle_tol", int(DEFAULT_PARAMS["angle_tol"]), 30)

        with open_camera() as cam:
            while True:
                ok, frame = cam.read()
                if not ok:
                    raise RuntimeError("Kameraframe konnte nicht gelesen werden.")

                params = {
                    "hough_thresh": cv2.getTrackbarPos("hough_thresh", ui),
                    "min_len_pct": cv2.getTrackbarPos("min_len_pct", ui),
                    "max_gap": cv2.getTrackbarPos("max_gap", ui),
                    "angle_tol": max(1, cv2.getTrackbarPos("angle_tol", ui)),
                }

                pts, vis, bw = detect_board_positions(frame, debug=False, return_bw=True, params=params)
                cv2.imshow(win, vis)
                cv2.imshow(bw_win, bw)
                key = cv2.waitKey(1) & 0xFF
                if args.single_frame or key in (ord("q"), 27):
                    break
        cv2.destroyAllWindows()
