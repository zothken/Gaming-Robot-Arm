# Minimaler Board-Detector: nutzt adaptives BW + Kanten + Hough als Basis.
# Optional: Live-Slider für BW/Canny/Hough/Orientierung.

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Literal, overload
import cv2
import numpy as np

from gaming_robot_arm.config import BOARD_LINE_PARAMS
from gaming_robot_arm.games.mill.board import BOARD_LABELS, RINGS

# Feste Standardparameter (nur diese werden via Slider angepasst)
DEFAULT_PARAMS = BOARD_LINE_PARAMS
CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.py"
ENABLE_ROTATION_ESTIMATION = False  # TODO: wieder aktivieren, sobald stabil
BOARD_PARAM_ORDER = [
    ("hough_thresh", None),
    ("min_len_pct", "Prozent von max(H, W)"),
    ("max_gap", None),
    ("angle_tol", None),
    ("bw_block", None),
    ("bw_C", None),
    ("bw_open", None),
    ("edge_close", None),
    ("blur_ksize", None),
]


@dataclass(frozen=True)
class Line:
    orientation: str  # 'H' oder 'V'
    slope: float
    intercept: float

    @property
    def anchor(self) -> float:
        return self.intercept


@dataclass(frozen=True)
class LineGroup:
    line: Line
    span_min: float
    span_max: float

    @property
    def anchor(self) -> float:
        return self.line.anchor


def angle_deg(x1, y1, x2, y2) -> float:
    return float(np.degrees(np.arctan2((y2 - y1), (x2 - x1))))


def rotate_points(pts, center, angle_deg):
    """Rotiert Punkte um den Mittelpunkt `center` um `angle_deg`."""
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
    """Rotiert Segmentendpunkte um den Mittelpunkt `center` um `angle_deg`."""
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
        if min(abs(a - 0.0), abs(180.0 - a)) <= tolerance_deg:
            horizontals.append((x1, y1, x2, y2))
        elif abs(a - 90.0) <= tolerance_deg:
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


def segment_span(segments, vertical=False):
    if len(segments) == 0:
        return 0.0, 0.0
    vals = []
    for (x1, y1, x2, y2) in segments:
        if vertical:
            vals.extend([y1, y2])
        else:
            vals.extend([x1, x2])
    return float(np.min(vals)), float(np.max(vals))


def intersect_within_spans(h_group, v_group, img_w, img_h, span_pad):
    x, y = intersect(h_group.line, v_group.line)
    if x < -span_pad or x > img_w - 1 + span_pad:
        return False
    if y < -span_pad or y > img_h - 1 + span_pad:
        return False
    if x < h_group.span_min - span_pad or x > h_group.span_max + span_pad:
        return False
    if y < v_group.span_min - span_pad or y > v_group.span_max + span_pad:
        return False
    return True


def intersection_counts(h_groups, v_groups, img_w, img_h, span_pad):
    h_counts = [0] * len(h_groups)
    v_counts = [0] * len(v_groups)
    for i, h in enumerate(h_groups):
        for j, v in enumerate(v_groups):
            if intersect_within_spans(h, v, img_w, img_h, span_pad):
                h_counts[i] += 1
                v_counts[j] += 1
    return h_counts, v_counts


def select_lines_by_intersections(groups, scores, expected, center_anchor):
    if len(groups) <= expected:
        return [g.line for g in groups]
    pairs = sorted(zip(groups, scores), key=lambda p: p[0].anchor)
    groups_sorted = [p[0] for p in pairs]
    scores_sorted = np.array([p[1] for p in pairs], dtype=np.float32)

    best_slice = groups_sorted[:expected]
    best_score = float("inf")
    for i in range(len(groups_sorted) - expected + 1):
        subset = groups_sorted[i:i + expected]
        sub_scores = scores_sorted[i:i + expected]
        anchors = np.array([g.anchor for g in subset], dtype=np.float32)
        if len(anchors) >= 2:
            diffs = np.diff(anchors)
            mean_diff = float(np.mean(diffs))
            spacing_var = float(np.std(diffs) / (abs(mean_diff) + 1e-3))
        else:
            spacing_var = 0.0
        center_pen = abs(float(np.mean(anchors)) - center_anchor) / (abs(center_anchor) + 1e-3)
        score = -float(np.sum(sub_scores)) + 2.0 * spacing_var + 0.3 * center_pen
        if score < best_score:
            best_score = score
            best_slice = subset

    return [g.line for g in best_slice]


def normalize_board_params(params):
    normalized = dict(params)
    if "bw_block" in normalized:
        normalized["bw_block"] = max(3, int(normalized["bw_block"]) | 1)
    if "blur_ksize" in normalized:
        normalized["blur_ksize"] = max(1, int(normalized["blur_ksize"]) | 1)
    if "bw_open" in normalized:
        normalized["bw_open"] = max(1, int(normalized["bw_open"]))
    if "edge_close" in normalized:
        normalized["edge_close"] = max(1, int(normalized["edge_close"]))
    return normalized


def format_board_params(params):
    lines = ["BOARD_LINE_PARAMS = {"]
    for key, comment in BOARD_PARAM_ORDER:
        if key not in params:
            continue
        val = int(params[key])
        if comment:
            lines.append(f'    "{key}": {val},  # {comment}')
        else:
            lines.append(f'    "{key}": {val},')
    lines.append("}")
    return "\n".join(lines) + "\n"


def write_board_params_to_config(params, path=CONFIG_PATH):
    if not path.exists():
        print(f"WARN: config.py nicht gefunden: {path}")
        return False
    text = path.read_text(encoding="utf-8")
    pattern = r"BOARD_LINE_PARAMS\s*=\s*\{.*?\}\n?"
    match = re.search(pattern, text, flags=re.S)
    if not match:
        print("WARN: BOARD_LINE_PARAMS Block nicht gefunden.")
        return False
    new_block = format_board_params(params)
    updated = text[:match.start()] + new_block + text[match.end():]
    if updated != text:
        path.write_text(updated, encoding="utf-8")
    return True


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


@overload
def detect_board_positions(
    frame_bgr,
    debug: bool = True,
    return_bw: Literal[False] = False,
    params=None,
) -> tuple[list[tuple[int, int]], np.ndarray]: ...


@overload
def detect_board_positions(
    frame_bgr,
    debug: bool = True,
    return_bw: Literal[True] = ...,
    params=None,
) -> tuple[list[tuple[int, int]], np.ndarray, np.ndarray]: ...


def detect_board_positions(frame_bgr, debug=True, return_bw=False, params=None):
    H_img, W_img = frame_bgr.shape[:2]
    annotated = frame_bgr.copy()
    p = DEFAULT_PARAMS.copy()
    if params:
        p.update(params)

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blur_ksize = int(p.get("blur_ksize", 5))
    if blur_ksize % 2 == 0:
        blur_ksize += 1
    blur_ksize = max(1, blur_ksize)
    blur = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    bw_block = int(p.get("bw_block", 15))
    if bw_block % 2 == 0:
        bw_block += 1
    bw_block = max(3, bw_block)
    bw_C = int(p.get("bw_C", 10))
    bw = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, bw_block, bw_C
    )
    bw_open = max(1, int(p.get("bw_open", 3)))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((bw_open, bw_open), np.uint8))

    edges = cv2.Canny(bw, 50, 150, apertureSize=3)
    edge_close = max(1, int(p.get("edge_close", 3)))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((edge_close, edge_close), np.uint8))

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

    # Brettrotation (vorerst deaktiviert): nutze nur horizontale/vertikale Segmente mit angle_tol.
    image_center = (0.5 * W_img, 0.5 * H_img)
    board_rotation = 0.0
    segments_aligned = segments
    if ENABLE_ROTATION_ESTIMATION:
        board_rotation = estimate_board_rotation(segments)
        if board_rotation > 45.0:
            board_rotation -= 90.0
        elif board_rotation < -45.0:
            board_rotation += 90.0
        if debug:
            cv2.putText(annotated, f"rot {board_rotation:+.1f} deg", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        segments_aligned = rotate_segments(segments, image_center, -board_rotation)

    # Segmentanker clustern → Linien fitten im ausgerichteten System
    horiz_segs, vert_segs = separate_segments_by_orientation(segments_aligned, tolerance_deg=float(p["angle_tol"]))

    def cluster_and_fit(seg_list, vertical=False):
        if len(seg_list) == 0:
            return []
        anchors = [0.5 * (s[1] + s[3]) if not vertical else 0.5 * (s[0] + s[2]) for s in seg_list]
        order = np.argsort(anchors)
        anchors_sorted = [anchors[i] for i in order]
        segs_sorted = [seg_list[i] for i in order]
        if len(anchors_sorted) >= 2:
            diffs = np.diff(anchors_sorted)
            median_spacing = float(np.median(diffs)) if len(diffs) else 40.0
        else:
            median_spacing = 40.0
        thresh = max(15.0, 0.5 * median_spacing)

        clusters = [[segs_sorted[0]]]
        last_anchor = anchors_sorted[0]
        for seg, a in zip(segs_sorted[1:], anchors_sorted[1:]):
            if abs(a - last_anchor) > thresh:
                clusters.append([seg])
            else:
                clusters[-1].append(seg)
            last_anchor = a

        groups = []
        for c in clusters:
            ln = fit_line_from_segments(c, vertical=vertical)
            span_min, span_max = segment_span(c, vertical=vertical)
            groups.append(LineGroup(ln, span_min, span_max))

        groups = sorted(groups, key=lambda g: g.anchor)
        return groups

    H_groups = cluster_and_fit(horiz_segs, vertical=False)
    V_groups = cluster_and_fit(vert_segs, vertical=True)

    if H_groups and V_groups:
        # Nur Schnittpunkte bei Segment-Ueberlappung helfen, Feldlinien von Brettkanten zu trennen.
        span_pad = max(8.0, 0.02 * min(W_img, H_img))
        h_counts, v_counts = intersection_counts(H_groups, V_groups, W_img, H_img, span_pad)
        H_lines = select_lines_by_intersections(H_groups, h_counts, expected=7, center_anchor=H_img * 0.5)
        V_lines = select_lines_by_intersections(V_groups, v_counts, expected=7, center_anchor=W_img * 0.5)
    else:
        H_lines = [g.line for g in H_groups]
        V_lines = [g.line for g in V_groups]

    # Robustheit: stelle sicher, dass immer 7 Linien pro Achse vorhanden sind (im ausgerichteten System)
    H_lines = ensure_lines(H_lines, expected=7, axis='H')
    V_lines = ensure_lines(V_lines, expected=7, axis='V')

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

    coords_aligned = {}
    for label, (iT, iB, iL, iR) in zip(
        list(RINGS.keys()),
        [outer + (outer[0], outer[1]), middle + (middle[0], middle[1]), inner + (inner[0], inner[1])],
    ):
        pts = quad_positions(iT, iB, iL, iR)
        for idx, (x, y) in enumerate(pts, start=1):
            coords_aligned[f"{label}{idx}"] = (int(x), int(y))

    # Koordinaten zurück ins Originalbild rotieren
    def rotate_coords_dict(coords_dict, angle_deg, center):
        if not coords_dict:
            return {}
        pts = np.array(list(coords_dict.values()), dtype=np.float32)
        rotated = rotate_points(pts, center, angle_deg)
        keys = list(coords_dict.keys())
        return {k: (int(round(p[0])), int(round(p[1]))) for k, p in zip(keys, rotated)}

    coords = rotate_coords_dict(coords_aligned, board_rotation, image_center)

    def draw_labels(frame, coords_dict):
        out = frame.copy()
        for key, (x, y) in coords_dict.items():
            cv2.circle(out, (x, y), 6, (0, 0, 255), -1)
            cv2.putText(out, key, (x - 20, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 164, 0), 2, cv2.LINE_AA)
        return out

    annotated = draw_labels(annotated, coords)
    if debug:
        print(f"{len(coords)} Brettpositionen (sollte {len(BOARD_LABELS)} sein)")

    positions: list[tuple[int, int]] = list(coords.values())
    if return_bw:
        return positions, annotated, bw
    return positions, annotated


if __name__ == "__main__":
    from gaming_robot_arm.vision.recording import open_camera

    import argparse

    parser = argparse.ArgumentParser(description="Brettdetektion (einfach, BW-basiert).")
    parser.add_argument("--image", type=str, help="Optionaler Pfad zu einem Testbild.")
    parser.add_argument("--single-frame", action="store_true", help="Nur ersten Kameraframe auswerten.")
    args = parser.parse_args()

    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            raise RuntimeError(f"Bild nicht gefunden: {args.image}")
        pts, vis, bw = detect_board_positions(img, debug=True, return_bw=True)
        print(f"Gefundene Brett-Positionen: {len(pts)}")
        cv2.imshow("Brettdetektion Debug (neu)", vis)
        cv2.imshow("BW-Schwellwert", bw)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        win = "Brettdetektion Debug (neu)"
        bw_win = "BW-Schwellwert"
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
        add_slider("bw_block", DEFAULT_PARAMS["bw_block"], 99)
        add_slider("bw_C", DEFAULT_PARAMS["bw_C"], 30)
        add_slider("bw_open", DEFAULT_PARAMS["bw_open"], 9)
        add_slider("edge_close", DEFAULT_PARAMS["edge_close"], 9)
        add_slider("blur_ksize", DEFAULT_PARAMS["blur_ksize"], 21)
        add_slider("save_config", 0, 1)

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
                    "bw_block": cv2.getTrackbarPos("bw_block", ui),
                    "bw_C": cv2.getTrackbarPos("bw_C", ui),
                    "bw_open": max(1, cv2.getTrackbarPos("bw_open", ui)),
                    "edge_close": max(1, cv2.getTrackbarPos("edge_close", ui)),
                    "blur_ksize": max(1, cv2.getTrackbarPos("blur_ksize", ui)),
                }

                save_params = normalize_board_params(params)
                if cv2.getTrackbarPos("save_config", ui) == 1:
                    if write_board_params_to_config(save_params):
                        DEFAULT_PARAMS.update(save_params)
                        print("Board-Parameter in config.py gespeichert.")
                    cv2.setTrackbarPos("save_config", ui, 0)

                pts, vis, bw = detect_board_positions(frame, debug=False, return_bw=True, params=params)
                cv2.imshow(win, vis)
                cv2.imshow(bw_win, bw)
                key = cv2.waitKey(1) & 0xFF
                if args.single_frame or key in (ord("q"), 27):
                    break
        cv2.destroyAllWindows()
