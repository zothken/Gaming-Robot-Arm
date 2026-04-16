# Board-Detector: Konturbasierte Erkennung der 3 verschachtelten Quadrate.
# Erkennt die Muehle-Brettpositionen stabil ueber Konturen statt Hough-Linien.

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Literal, overload
import numpy as np

from gaming_robot_arm.games.mill.core.board import BOARD_LABELS, RINGS

try:
    import cv2
except ModuleNotFoundError as exc:  # pragma: no cover - optionaler Laufzeitpfad
    _cv2_import_error = exc

    class _Cv2Proxy:
        def __getattr__(self, name: str):
            raise ModuleNotFoundError(
                "Mill-Board-Detektion benoetigt OpenCV (`opencv-python`)."
            ) from _cv2_import_error

    cv2 = _Cv2Proxy()  # type: ignore[assignment]

# Konturen-basierte Parameter
BOARD_LINE_PARAMS = {
    "blur_ksize": 21,
    "bw_block": 19,
    "bw_C": 10,
    "bw_open": 2,
    "morph_close": 15,
    "approx_eps_pct": 3,
    "min_area_pct": 1,
    "ema_alpha": 40,
}
DEFAULT_PARAMS = BOARD_LINE_PARAMS.copy()
DETECTOR_PATH = Path(__file__).resolve()

BOARD_PARAM_ORDER = [
    ("blur_ksize", "Gauss-Blur (ungerade)"),
    ("bw_block", "Adaptiver Schwellwert Blockgroesse"),
    ("bw_C", "Adaptiver Schwellwert C"),
    ("bw_open", "Morph. Opening Kernel"),
    ("morph_close", "Morph. Closing Kernel (Linien verbinden)"),
    ("approx_eps_pct", "approxPolyDP Epsilon (% Umfang)"),
    ("min_area_pct", "Min. Konturflaeche (% Bildflaeche)"),
    ("ema_alpha", "EMA Glaettung (alpha*100)"),
]


# ---------------------------------------------------------------------------
# EMA state (module-level, persists across calls within one session)
# ---------------------------------------------------------------------------
_ema_positions: dict[str, tuple[float, float]] = {}
_ema_initialized: bool = False


def reset_ema() -> None:
    """Reset the EMA smoothing state (e.g. when the board is repositioned)."""
    global _ema_positions, _ema_initialized
    _ema_positions = {}
    _ema_initialized = False


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 corners as TL, TR, BR, BL.

    Uses sum (x+y) for TL/BR and difference (y-x) for TR/BL.
    """
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    d = pts[:, 1] - pts[:, 0]
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]   # TL: smallest x+y
    ordered[2] = pts[np.argmax(s)]   # BR: largest x+y
    ordered[1] = pts[np.argmin(d)]   # TR: smallest y-x
    ordered[3] = pts[np.argmax(d)]   # BL: largest y-x
    return ordered


def _midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return 0.5 * (a + b)


def _centroid(pts: np.ndarray) -> np.ndarray:
    return pts.reshape(-1, 2).mean(axis=0)


def _quad_area(corners: np.ndarray) -> float:
    """Shoelace area for 4 ordered corners."""
    c = corners.reshape(4, 2)
    x, y = c[:, 0], c[:, 1]
    return 0.5 * abs(float(
        np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))
    ))


def _is_roughly_convex(corners: np.ndarray, min_angle: float = 30.0) -> bool:
    """Check that all interior angles are within [min_angle, 180 - min_angle]."""
    c = corners.reshape(4, 2)
    for i in range(4):
        p0 = c[(i - 1) % 4]
        p1 = c[i]
        p2 = c[(i + 1) % 4]
        v1 = p0 - p1
        v2 = p2 - p1
        cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
        angle = np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))
        if angle < min_angle or angle > (180.0 - min_angle):
            return False
    return True



def _ring_positions_from_corners(corners: np.ndarray) -> list[tuple[int, int]]:
    """Compute 8 positions (clockwise from TL) from 4 ordered corners TL,TR,BR,BL."""
    tl, tr, br, bl = corners[0], corners[1], corners[2], corners[3]
    positions = [
        tl,                    # pos 1: top-left
        _midpoint(tl, tr),     # pos 2: top-center
        tr,                    # pos 3: top-right
        _midpoint(tr, br),     # pos 4: right-center
        br,                    # pos 5: bottom-right
        _midpoint(br, bl),     # pos 6: bottom-center
        bl,                    # pos 7: bottom-left
        _midpoint(bl, tl),     # pos 8: left-center
    ]
    return [(int(round(p[0])), int(round(p[1]))) for p in positions]


def _estimate_inner_corners(outer: np.ndarray, fraction: float) -> np.ndarray:
    """Estimate an inner square's corners by interpolating toward the centroid."""
    center = _centroid(outer)
    return outer + fraction * (center - outer)


# ---------------------------------------------------------------------------
# Contour detection
# ---------------------------------------------------------------------------

def _find_board_quadrilaterals(
    bw: np.ndarray,
    approx_eps_pct: float,
    min_area: float,
) -> list[np.ndarray]:
    """Find quadrilateral contours in a binary image.

    Returns list of 4-corner arrays sorted by area (descending).
    """
    contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    quads: list[tuple[float, np.ndarray]] = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        peri = cv2.arcLength(cnt, True)
        eps = approx_eps_pct * 0.01 * peri

        # Try to approximate to 4 vertices
        approx = cv2.approxPolyDP(cnt, eps, True)

        if len(approx) == 4:
            corners = _order_corners(approx.reshape(4, 2))
            if _is_roughly_convex(corners):
                quads.append((area, corners))
        elif 5 <= len(approx) <= 8:
            # Try more aggressive simplification
            for mult in [1.5, 2.0, 2.5, 3.0]:
                approx2 = cv2.approxPolyDP(cnt, eps * mult, True)
                if len(approx2) == 4:
                    corners = _order_corners(approx2.reshape(4, 2))
                    if _is_roughly_convex(corners):
                        quads.append((area, corners))
                    break
        # Also try minAreaRect as fallback for large contours
        if area > min_area * 3 and not any(
            abs(a - area) / max(a, 1) < 0.1 for a, _ in quads
        ):
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            corners = _order_corners(box)
            rect_area = _quad_area(corners)
            # Only use minAreaRect if it reasonably matches the contour area
            if rect_area > 0 and 0.6 < area / rect_area < 1.1:
                quads.append((area, corners))

    # Sort by area descending, remove near-duplicates
    quads.sort(key=lambda x: x[0], reverse=True)
    filtered: list[np.ndarray] = []
    for area, corners in quads:
        is_dup = False
        for existing in filtered:
            # Check if centroids are close and areas are similar
            c1 = _centroid(corners)
            c2 = _centroid(existing)
            dist = float(np.linalg.norm(c1 - c2))
            a1 = _quad_area(corners)
            a2 = _quad_area(existing)
            if dist < 30 and abs(a1 - a2) / max(a1, a2, 1) < 0.15:
                is_dup = True
                break
        if not is_dup:
            filtered.append(corners)

    return filtered


def _find_outer_square(
    quads: list[np.ndarray],
    img_area: float,
) -> np.ndarray | None:
    """Find the outer board square from candidate quadrilaterals.

    Returns the largest valid quadrilateral, or None.
    """
    for q in quads:
        area = _quad_area(q)
        if area < img_area * 0.01:
            continue
        rect = cv2.minAreaRect(q.reshape(4, 1, 2).astype(np.float32))
        w, h = rect[1]
        if w < 1 or h < 1:
            continue
        aspect = max(w, h) / min(w, h)
        if aspect > 2.5:
            continue
        return q
    return None


# ---------------------------------------------------------------------------
# Parameter persistence (kept for compatibility)
# ---------------------------------------------------------------------------

def normalize_board_params(params):
    normalized = dict(params)
    if "bw_block" in normalized:
        normalized["bw_block"] = max(3, int(normalized["bw_block"]) | 1)
    if "blur_ksize" in normalized:
        normalized["blur_ksize"] = max(1, int(normalized["blur_ksize"]) | 1)
    if "bw_open" in normalized:
        normalized["bw_open"] = max(1, int(normalized["bw_open"]))
    if "morph_close" in normalized:
        normalized["morph_close"] = max(1, int(normalized["morph_close"]))
    if "approx_eps_pct" in normalized:
        normalized["approx_eps_pct"] = max(1, int(normalized["approx_eps_pct"]))
    if "min_area_pct" in normalized:
        normalized["min_area_pct"] = max(1, int(normalized["min_area_pct"]))
    if "ema_alpha" in normalized:
        normalized["ema_alpha"] = max(1, min(100, int(normalized["ema_alpha"])))
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


def write_board_params_to_detector(params, path=DETECTOR_PATH):
    if not path.exists():
        print(f"WARN: Detector-Datei nicht gefunden: {path}")
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


# ---------------------------------------------------------------------------
# Main detection function
# ---------------------------------------------------------------------------

@overload
def detect_board_positions(
    frame_bgr,
    debug: bool = True,
    *,
    return_bw: Literal[False] = False,
    params=None,
) -> tuple[list[tuple[int, int]], np.ndarray]: ...


@overload
def detect_board_positions(
    frame_bgr,
    debug: bool = True,
    *,
    return_bw: Literal[True],
    params=None,
) -> tuple[list[tuple[int, int]], np.ndarray, np.ndarray]: ...


def detect_board_positions(frame_bgr, debug=True, return_bw=False, params=None):
    global _ema_positions, _ema_initialized

    H_img, W_img = frame_bgr.shape[:2]
    img_area = float(H_img * W_img)
    annotated = frame_bgr.copy()
    p = DEFAULT_PARAMS.copy()
    if params:
        p.update(params)

    # --- Step 1: Preprocessing ---
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    blur_ksize = int(p.get("blur_ksize", 21))
    if blur_ksize % 2 == 0:
        blur_ksize += 1
    blur_ksize = max(1, blur_ksize)
    blur = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    bw_block = int(p.get("bw_block", 19))
    if bw_block % 2 == 0:
        bw_block += 1
    bw_block = max(3, bw_block)
    bw_C = int(p.get("bw_C", 10))
    bw = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, bw_block, bw_C
    )

    bw_open = max(1, int(p.get("bw_open", 2)))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((bw_open, bw_open), np.uint8))

    # Morphological closing to connect broken board lines
    morph_close = max(1, int(p.get("morph_close", 15)))
    bw_closed = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((morph_close, morph_close), np.uint8))

    # --- Step 2: Find quadrilateral contours ---
    approx_eps_pct = max(1, int(p.get("approx_eps_pct", 3)))
    min_area_pct = max(1, int(p.get("min_area_pct", 1)))
    min_area = img_area * min_area_pct * 0.01

    quads = _find_board_quadrilaterals(bw_closed, approx_eps_pct, min_area)

    # --- Step 3: Find outer square, compute middle/inner geometrically ---
    outer = _find_outer_square(quads, img_area)

    # Mill boards have 3 equally-spaced concentric squares.
    # Middle corners are 1/3 of the way from outer to center,
    # inner corners are 2/3 of the way.
    middle = _estimate_inner_corners(outer, 1.0 / 3.0) if outer is not None else None
    inner = _estimate_inner_corners(outer, 2.0 / 3.0) if outer is not None else None

    # --- Step 4: Compute 24 positions ---
    coords: dict[str, tuple[int, int]] = {}
    squares = {"A": outer, "B": middle, "C": inner}

    all_found = all(s is not None for s in squares.values())

    if all_found:
        for ring_name, ring_corners in squares.items():
            assert ring_corners is not None  # guaranteed by all_found
            positions = _ring_positions_from_corners(ring_corners)
            for idx, pos in enumerate(positions, start=1):
                coords[f"{ring_name}{idx}"] = pos

    # --- Step 5: EMA smoothing ---
    ema_alpha = max(1, min(100, int(p.get("ema_alpha", 40)))) / 100.0

    if coords and len(coords) == len(BOARD_LABELS):
        if not _ema_initialized:
            _ema_positions = {k: (float(v[0]), float(v[1])) for k, v in coords.items()}
            _ema_initialized = True
        else:
            # Check for large displacement (board moved)
            displacements = []
            for k, (nx, ny) in coords.items():
                if k in _ema_positions:
                    ox, oy = _ema_positions[k]
                    displacements.append(((nx - ox) ** 2 + (ny - oy) ** 2) ** 0.5)
            avg_disp = float(np.mean(displacements)) if displacements else 0.0

            if avg_disp > 50.0:
                # Board was moved significantly, reset EMA
                _ema_positions = {k: (float(v[0]), float(v[1])) for k, v in coords.items()}
            else:
                # Smooth
                for k, (nx, ny) in coords.items():
                    if k in _ema_positions:
                        ox, oy = _ema_positions[k]
                        _ema_positions[k] = (
                            ema_alpha * nx + (1 - ema_alpha) * ox,
                            ema_alpha * ny + (1 - ema_alpha) * oy,
                        )
                    else:
                        _ema_positions[k] = (float(nx), float(ny))

        # Use smoothed positions
        coords = {k: (int(round(v[0])), int(round(v[1]))) for k, v in _ema_positions.items()}
    elif _ema_initialized and _ema_positions:
        # Detection failed this frame, use previous smoothed positions
        coords = {k: (int(round(v[0])), int(round(v[1]))) for k, v in _ema_positions.items()}

    # --- Visualization ---
    # Draw all candidate contours lightly
    for q in quads:
        pts = q.reshape(4, 1, 2).astype(np.int32)
        cv2.polylines(annotated, [pts], True, (0, 255, 0), 1)

    # Draw identified squares
    colors = {"A": (255, 100, 0), "B": (255, 255, 0), "C": (255, 0, 255)}
    for ring_name, ring_corners in squares.items():
        if ring_corners is not None:
            pts = ring_corners.reshape(4, 1, 2).astype(np.int32)
            cv2.polylines(annotated, [pts], True, colors[ring_name], 3)

    # Draw position labels
    for key, (x, y) in coords.items():
        cv2.circle(annotated, (x, y), 6, (0, 0, 255), -1)
        cv2.putText(annotated, key, (x - 20, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 164, 0), 2, cv2.LINE_AA)

    if debug:
        n_detected = sum(1 for s in [outer, middle, inner] if s is not None)
        print(f"{len(coords)} Brettpositionen (sollte {len(BOARD_LABELS)} sein), "
              f"{n_detected}/3 Quadrate erkannt, {len(quads)} Kandidaten")

    positions: list[tuple[int, int]] = [coords[lbl] for lbl in BOARD_LABELS] if len(coords) == len(BOARD_LABELS) else []
    if return_bw:
        return positions, annotated, bw
    return positions, annotated


if __name__ == "__main__":
    from gaming_robot_arm.vision.recording import open_camera

    import argparse

    parser = argparse.ArgumentParser(description="Brettdetektion (konturbasiert).")
    parser.add_argument("--image", type=str, help="Optionaler Pfad zu einem Testbild.")
    parser.add_argument("--single-frame", action="store_true", help="Nur ersten Kameraframe auswerten.")
    args = parser.parse_args()

    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            raise RuntimeError(f"Bild nicht gefunden: {args.image}")
        pts, vis, bw_out = detect_board_positions(img, debug=True, return_bw=True)
        print(f"Gefundene Brett-Positionen: {len(pts)}")
        cv2.imshow("Brettdetektion Debug (Konturen)", vis)
        cv2.imshow("BW-Schwellwert", bw_out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        win = "Brettdetektion Debug (Konturen)"
        bw_win = "BW-Schwellwert"
        ui = "Board Params"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.namedWindow(bw_win, cv2.WINDOW_NORMAL)
        cv2.namedWindow(ui, cv2.WINDOW_NORMAL)

        def add_slider(name, init, maxval):
            cv2.createTrackbar(name, ui, init, maxval, lambda v: None)
            cv2.setTrackbarPos(name, ui, init)

        add_slider("blur_ksize", DEFAULT_PARAMS["blur_ksize"], 41)
        add_slider("bw_block", DEFAULT_PARAMS["bw_block"], 99)
        add_slider("bw_C", DEFAULT_PARAMS["bw_C"], 30)
        add_slider("bw_open", DEFAULT_PARAMS["bw_open"], 15)
        add_slider("morph_close", DEFAULT_PARAMS["morph_close"], 30)
        add_slider("approx_eps_pct", DEFAULT_PARAMS["approx_eps_pct"], 10)
        add_slider("min_area_pct", DEFAULT_PARAMS["min_area_pct"], 10)
        add_slider("ema_alpha", DEFAULT_PARAMS["ema_alpha"], 100)
        add_slider("save_config", 0, 1)

        with open_camera() as cam:
            while True:
                ok, frame = cam.read()
                if not ok:
                    raise RuntimeError("Kameraframe konnte nicht gelesen werden.")

                slider_params = {
                    "blur_ksize": max(1, cv2.getTrackbarPos("blur_ksize", ui)),
                    "bw_block": cv2.getTrackbarPos("bw_block", ui),
                    "bw_C": cv2.getTrackbarPos("bw_C", ui),
                    "bw_open": max(1, cv2.getTrackbarPos("bw_open", ui)),
                    "morph_close": max(1, cv2.getTrackbarPos("morph_close", ui)),
                    "approx_eps_pct": max(1, cv2.getTrackbarPos("approx_eps_pct", ui)),
                    "min_area_pct": max(1, cv2.getTrackbarPos("min_area_pct", ui)),
                    "ema_alpha": max(1, cv2.getTrackbarPos("ema_alpha", ui)),
                }

                save_params = normalize_board_params(slider_params)
                if cv2.getTrackbarPos("save_config", ui) == 1:
                    if write_board_params_to_detector(save_params):
                        DEFAULT_PARAMS.update(save_params)
                        print("Board-Parameter in mill_board_detector.py gespeichert.")
                    cv2.setTrackbarPos("save_config", ui, 0)

                pts, vis, bw_out = detect_board_positions(frame, debug=False, return_bw=True, params=slider_params)
                cv2.imshow(win, vis)
                cv2.imshow(bw_win, bw_out)
                key = cv2.waitKey(1) & 0xFF
                if args.single_frame or key in (ord("q"), 27):
                    break
        cv2.destroyAllWindows()
