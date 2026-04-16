from collections import Counter, deque
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, List, Literal, Mapping, Protocol, Sequence, Tuple, TypedDict, overload

import cv2
import numpy as np

from gaming_robot_arm.config import CAMERA_INDEX, FRAME_RATE
from gaming_robot_arm.games.common.interfaces import Player
from gaming_robot_arm.games.mill.core.board import BOARD_LABELS
from gaming_robot_arm.utils.logger import logger
from gaming_robot_arm.vision.detector_config import (
    load_figure_params,
    save_figure_params,
)
from gaming_robot_arm.vision.recording import get_effective_camera_fps, open_camera
from gaming_robot_arm.vision.visualization import draw_assignment_labels, draw_detections

# Temporärer Korrekturwert für die Brett-Zuordnung (Pixelraum).
ASSIGNMENT_Y_OFFSET_PX = 0.0
DEFAULT_FIGURE_PARAMS: dict[str, int | float] = load_figure_params()
FIGURE_PARAM_ORDER = [
    ("blur_ksize", "Gauß-Blur-Kernelgroesse (ungerade; groesser = staerkeres Glaetten)"),
    ("thresh_block", "Blockgroesse fuer adaptive Schwellwertbildung (ungerade, >=3)"),
    ("thresh_c", "Konstante C fuer adaptive Schwellwertbildung"),
    ("min_radius", "Kleinster erlaubter Kreisradius fuer HoughCircles"),
    ("max_radius", "Groesster erlaubter Kreisradius fuer HoughCircles"),
    ("hough_dp", "Inverses Aufloesungsverhaeltnis des Hough-Akkumulators"),
    ("hough_min_dist", "Mindestabstand zwischen erkannten Kreiszentren (px)"),
    ("hough_param1", "Oberer Canny-Schwellwert in HoughCircles"),
    ("hough_param2", "Akkumulator-Schwelle (hoeher = strengere Kreisakzeptanz)"),
    ("brightness_split", "Helligkeitsgrenze fuer Farbklassifikation (schwarz/weiss)"),
]


def normalize_figure_params(params: Mapping[str, int | float] | None = None) -> dict[str, int | float]:
    normalized = dict(DEFAULT_FIGURE_PARAMS)
    if params:
        normalized.update(params)

    blur_ksize = max(1, int(normalized["blur_ksize"]))
    if blur_ksize % 2 == 0:
        blur_ksize += 1
    normalized["blur_ksize"] = blur_ksize

    thresh_block = max(3, int(normalized["thresh_block"]))
    if thresh_block % 2 == 0:
        thresh_block += 1
    normalized["thresh_block"] = thresh_block

    normalized["thresh_c"] = int(normalized["thresh_c"])
    normalized["min_radius"] = max(1, int(normalized["min_radius"]))
    normalized["max_radius"] = max(int(normalized["min_radius"]), int(normalized["max_radius"]))
    normalized["hough_dp"] = max(0.1, float(normalized["hough_dp"]))
    normalized["hough_min_dist"] = max(1, int(normalized["hough_min_dist"]))
    normalized["hough_param1"] = max(1, int(normalized["hough_param1"]))
    normalized["hough_param2"] = max(1, int(normalized["hough_param2"]))
    normalized["brightness_split"] = min(255, max(0, int(normalized["brightness_split"])))

    return normalized

def write_figure_params_to_detector(params: Mapping[str, int | float]) -> bool:
    save_figure_params(normalize_figure_params(params))
    return True


class AssignmentBase(TypedDict):
    label: str
    centroid: tuple[int, int]
    color: str


class Assignment(AssignmentBase, total=False):
    dist_px: float
    presence_ratio: float


DetectionResult = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
DetectionResultWithAssignments = Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    List[Assignment],
]
BoardCoordSource = Literal["calibration", "live", "auto"]


class FigureDetectionSession(Protocol):
    """Minimales Session-Interface fuer Live-Kamera plus optionales Mitschreiben."""

    camera: cv2.VideoCapture

    def write(self, frame: np.ndarray) -> None: ...


@dataclass(slots=True)
class BoardAssignmentObservation:
    """Zwischenergebnis einer fortlaufenden Brettbeobachtung."""

    stable_board: dict[str, Player | None]
    stable_assignments: list["Assignment"]
    raw_assignments: list["Assignment"]
    frames_seen: int
    board_source: str
    ready: bool


def estimate_assign_distance(
    board_coords: Mapping[str, tuple[float, float]],
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


def _with_assignment_offset(
    board_coords: Mapping[str, tuple[float, float]] | None,
) -> dict[str, tuple[float, float]] | None:
    if board_coords is None:
        return None
    if not ASSIGNMENT_Y_OFFSET_PX:
        return {lbl: (float(u), float(v)) for lbl, (u, v) in board_coords.items()}
    return {lbl: (float(u), float(v) + ASSIGNMENT_Y_OFFSET_PX) for lbl, (u, v) in board_coords.items()}


def _inside_ratio(
    board_coords: Mapping[str, tuple[float, float]],
    *,
    width: int,
    height: int,
) -> float:
    inside = sum(1 for x, y in board_coords.values() if 0 <= x < width and 0 <= y < height)
    return inside / max(1, len(board_coords))


def _coord_shift_median_px(
    first: Mapping[str, tuple[float, float]],
    second: Mapping[str, tuple[float, float]],
) -> float:
    common = sorted(set(first.keys()) & set(second.keys()))
    if not common:
        return 0.0
    diffs = [
        float(np.linalg.norm(np.array(first[lbl], dtype=np.float32) - np.array(second[lbl], dtype=np.float32)))
        for lbl in common
    ]
    return float(np.median(np.array(diffs, dtype=np.float32)))


def _distance_steps(base_dist: float) -> list[float]:
    return [base_dist]


def _detect_live_board_coords(frame: np.ndarray) -> dict[str, tuple[float, float]] | None:
    try:
        from gaming_robot_arm.vision.mill_board_detector import detect_board_positions
    except Exception:
        logger.exception("Board-Detector konnte fuer Live-Brettkoordinaten nicht importiert werden.")
        return None

    try:
        result = detect_board_positions(frame, debug=False, return_bw=False)
    except Exception:
        logger.exception("Live-Brettdetektion fuer Zuordnung fehlgeschlagen.")
        return None

    positions = result[0] if isinstance(result, tuple) and len(result) > 0 else []
    if positions is None or len(positions) != len(BOARD_LABELS):
        return None
    return {
        label: (float(x), float(y))
        for label, (x, y) in zip(BOARD_LABELS, positions)
    }


def _assignment_count_for_coords(
    frame: np.ndarray,
    board_coords: Mapping[str, tuple[float, float]],
    labels_order: Sequence[str],
    *,
    max_assign_dist: float,
) -> int:
    _, _, _, _, assignments = detect_figures(
        frame.copy(),
        board_coords=board_coords,
        max_assign_dist=max_assign_dist,
        labels_order=labels_order,
        draw_assignments=False,
        return_assignments=True,
        debug_assignments=False,
    )
    return len(assignments)


def _select_assignment_board_coords(
    frame: np.ndarray,
    *,
    calibration_coords: Mapping[str, tuple[float, float]] | None,
    labels_order: Sequence[str] | None,
    board_source: BoardCoordSource,
) -> tuple[dict[str, tuple[float, float]] | None, list[str], str]:
    cal_coords = dict(calibration_coords) if calibration_coords else None
    cal_labels = list(labels_order) if labels_order else (sorted(cal_coords.keys()) if cal_coords else [])

    if board_source == "calibration":
        return cal_coords, cal_labels, "calibration"

    live_coords = _detect_live_board_coords(frame)
    live_labels = (
        cal_labels
        if cal_labels and live_coords and set(cal_labels) == set(live_coords.keys())
        else (sorted(live_coords.keys()) if live_coords else [])
    )

    if board_source == "live":
        if live_coords is not None:
            return live_coords, live_labels, "live"
        return None, [], "live_unavailable"

    # auto
    if cal_coords is None and live_coords is None:
        return None, [], "none"
    if cal_coords is None and live_coords is not None:
        logger.warning("Keine Kalibrier-Brettpunkte verfuegbar; nutze Live-Brettdetektion.")
        return live_coords, live_labels, "live"
    if cal_coords is not None and live_coords is None:
        return cal_coords, cal_labels, "calibration"

    assert cal_coords is not None and live_coords is not None
    h, w = frame.shape[:2]
    cal_dist = estimate_assign_distance(cal_coords)
    live_dist = estimate_assign_distance(live_coords)
    cal_count = _assignment_count_for_coords(frame, cal_coords, cal_labels, max_assign_dist=cal_dist)
    live_count = _assignment_count_for_coords(frame, live_coords, live_labels, max_assign_dist=live_dist)
    shift_median = _coord_shift_median_px(cal_coords, live_coords)
    cal_inside = _inside_ratio(cal_coords, width=w, height=h)

    choose_live = False
    if live_count > cal_count:
        choose_live = True
    elif cal_inside < 0.8 and live_count >= cal_count:
        choose_live = True
    elif shift_median > max(15.0, 1.2 * cal_dist) and live_count >= cal_count:
        choose_live = True

    logger.info(
        "Brettquellen-Vergleich: cal_count=%s live_count=%s cal_dist=%.1f live_dist=%.1f shift_med=%.1fpx inside_cal=%.2f",
        cal_count,
        live_count,
        cal_dist,
        live_dist,
        shift_median,
        cal_inside,
    )

    if choose_live:
        logger.warning("Nutze Live-Brettpunkte fuer Figuren-Zuordnung (auto).")
        return live_coords, live_labels, "live"
    return cal_coords, cal_labels, "calibration"


def _empty_player_board() -> dict[str, Player | None]:
    return {label: None for label in BOARD_LABELS}


def _assignments_to_player_board(assignments: Sequence["Assignment"]) -> dict[str, Player | None]:
    board = _empty_player_board()
    for item in assignments:
        label = str(item.get("label", "")).upper()
        if label not in board:
            continue
        color = str(item.get("color", "")).lower()
        if color in {"weiss", "white"}:
            board[label] = "W"
        elif color in {"schwarz", "black"}:
            board[label] = "B"
    return board



def hough_detect_assignments(
    frame: np.ndarray,
    board_coords: Mapping[str, tuple[float, float]],
    max_assign_dist: float,
    labels_order: Sequence[str],
    debug_assignments: bool = False,
) -> list[Assignment]:
    """Wrapper um *detect_figures* mit dem gleichen Callable-Interface wie alternative Detektoren."""
    _, _, _, _, assignments = detect_figures(
        frame,
        board_coords=board_coords,
        max_assign_dist=max_assign_dist,
        return_assignments=True,
        labels_order=labels_order,
        debug_assignments=debug_assignments,
        draw_assignments=False,
    )
    return assignments


class BoardAssignmentStream:
    """Verarbeitet Videoframes fortlaufend zu stabilen Brettbeobachtungen."""

    def __init__(
        self,
        board_pixels: Mapping[str, tuple[float, float]],
        *,
        labels_order: Sequence[str] | None = None,
        board_source: BoardCoordSource = "live",
        window_frames: int = 30,
        min_ratio: float = 0.5,
        min_samples: int | None = None,
        debug_assignments: bool = False,
        max_assign_dist: float | None = None,
    ) -> None:
        self._calibration_board_pixels = _with_assignment_offset(board_pixels) or {}
        self._labels_hint = (
            list(labels_order) if labels_order else sorted(self._calibration_board_pixels.keys())
        )
        self._board_source: BoardCoordSource = board_source
        self._window_frames = max(1, int(window_frames))
        self._min_ratio = float(min_ratio)
        self._min_samples = self._window_frames if min_samples is None else max(1, int(min_samples))
        self._debug_assignments = bool(debug_assignments)
        self._max_assign_dist = (
            float(max_assign_dist) if max_assign_dist is not None and float(max_assign_dist) > 0 else None
        )

        self._frames_seen = 0
        self._frame_check_done = False
        self._warned_missing_board = False
        self._assignment_board_pixels: dict[str, tuple[float, float]] | None = None
        self._labels_order: list[str] = list(self._labels_hint)
        self._active_source = "calibration"
        self._assign_dist = 0.0
        self._dist_steps: list[float] = []
        self._stabilizer: AssignmentStabilizer | None = None

    def _ensure_board_setup(self, frame: np.ndarray) -> bool:
        if self._frame_check_done:
            return self._assignment_board_pixels is not None

        board_coords, labels_order, active_source = _select_assignment_board_coords(
            frame,
            calibration_coords=self._calibration_board_pixels,
            labels_order=self._labels_hint,
            board_source=self._board_source,
        )
        if board_coords is None:
            if not self._warned_missing_board:
                logger.warning("Keine Brettkoordinaten fuer Figuren-Zuordnung verfuegbar.")
                self._warned_missing_board = True
            return False

        self._assignment_board_pixels = board_coords
        self._labels_order = labels_order
        self._active_source = active_source
        self._assign_dist = self._max_assign_dist or estimate_assign_distance(self._assignment_board_pixels)
        self._dist_steps = [self._assign_dist] if self._max_assign_dist is not None else _distance_steps(self._assign_dist)
        self._stabilizer = AssignmentStabilizer(self._labels_order, window=self._window_frames)

        logger.info(
            "Figuren-Zuordnung nutzt Brettquelle '%s' (basis_dist=%.1fpx).",
            self._active_source,
            self._assign_dist,
        )

        h, w = frame.shape[:2]
        xs = [p[0] for p in self._assignment_board_pixels.values()]
        ys = [p[1] for p in self._assignment_board_pixels.values()]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        inside_ratio = _inside_ratio(self._assignment_board_pixels, width=w, height=h)
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

        self._frame_check_done = True
        return True

    def process_frame(self, frame: np.ndarray) -> BoardAssignmentObservation:
        if not self._ensure_board_setup(frame):
            return BoardAssignmentObservation(
                stable_board=_empty_player_board(),
                stable_assignments=[],
                raw_assignments=[],
                frames_seen=self._frames_seen,
                board_source="none",
                ready=False,
            )

        self._frames_seen += 1
        dist = self._dist_steps[min(self._frames_seen - 1, len(self._dist_steps) - 1)]
        raw_assignments: list[Assignment] = []

        try:
            _, _, _, _, raw_assignments = detect_figures(
                frame,
                board_coords=self._assignment_board_pixels,
                max_assign_dist=dist,
                return_assignments=True,
                debug_assignments=self._debug_assignments,
                labels_order=self._labels_order,
                draw_assignments=False,
            )
        except Exception:
            logger.exception("Fehler bei der Zuordnung der Figuren zum Brett.")

        stable_assignments: list[Assignment] = []
        if self._stabilizer is not None:
            self._stabilizer.update(raw_assignments)
            stable_assignments = self._stabilizer.stable_assignments(
                min_ratio=self._min_ratio,
                min_samples=self._min_samples,
            )

        ready = self._frames_seen >= self._min_samples
        stable_board = _assignments_to_player_board(stable_assignments) if ready else _empty_player_board()

        return BoardAssignmentObservation(
            stable_board=stable_board,
            stable_assignments=stable_assignments,
            raw_assignments=raw_assignments,
            frames_seen=self._frames_seen,
            board_source=self._active_source,
            ready=ready,
        )


def assign_figures_to_board(
    centroids: Sequence[tuple[int, int]],
    colors: Sequence[str],
    board_coords: Mapping[str, tuple[float, float]],
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
    used_dets: set[int] = set()
    used_labels: set[int] = set()

    while np.isfinite(D).any():
        i_raw, j_raw = np.unravel_index(np.argmin(D), D.shape)
        i, j = int(i_raw), int(j_raw)
        dist = float(D[i, j])
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
            centroid_arr = np.round(self.centroids[lbl][-1]).astype(int)
            centroid = (int(centroid_arr[0]), int(centroid_arr[1]))
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
    board_coords: Mapping[str, tuple[float, float]] | None = None,
    max_assign_dist: float | None = None,
    labels_order: Sequence[str] | None = None,
    *,
    return_assignments: Literal[False] = False,
    draw_assignments: bool = False,
    debug_assignments: bool = False,
    params: Mapping[str, int | float] | None = None,
) -> DetectionResult: ...


@overload
def detect_figures(
    frame: np.ndarray,
    tracker: Any = None,
    board_coords: Mapping[str, tuple[float, float]] | None = None,
    max_assign_dist: float | None = None,
    labels_order: Sequence[str] | None = None,
    *,
    return_assignments: Literal[True],
    draw_assignments: bool = False,
    debug_assignments: bool = False,
    params: Mapping[str, int | float] | None = None,
) -> DetectionResultWithAssignments: ...


def detect_figures(
    frame: np.ndarray,
    tracker: Any = None,
    board_coords: Mapping[str, tuple[float, float]] | None = None,
    max_assign_dist: float | None = None,
    labels_order: Sequence[str] | None = None,
    *,
    return_assignments: bool = False,
    draw_assignments: bool = False,
    debug_assignments: bool = False,
    params: Mapping[str, int | float] | None = None,
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
    - params: optionale Parameter fuer Blur/Threshold/Hough/Farbtrennung.
    """
    p = normalize_figure_params(params)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur_ksize = int(p["blur_ksize"])
    blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    min_radius = int(p["min_radius"])
    max_radius = int(p["max_radius"])
    thresh_block = int(p["thresh_block"])
    thresh_c = int(p["thresh_c"])

    # Adaptives Schwellwertverfahren
    thresh = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        thresh_block,
        thresh_c,
    )

    centroids: list[tuple[int, int]] = []
    colors: list[str] = []
    black_count, white_count = 0, 0

    # Kreisdetektion mit Hough-Transformation
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=float(p["hough_dp"]),
        minDist=int(p["hough_min_dist"]),
        param1=float(p["hough_param1"]),
        param2=float(p["hough_param2"]),
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    detected_circles: np.ndarray | None = None

    if circles is not None:
        detected_circles = np.round(circles[0, :]).astype(int)

        for x_raw, y_raw, r_raw in detected_circles:
            x, y, r = int(x_raw), int(y_raw), int(r_raw)
            center = (x, y)
            centroids.append(center)

            # Maske fuer lokale Helligkeitsmessung erzeugen
            mask = np.zeros(blurred.shape, dtype="uint8")
            cv2.circle(mask, center, r, 255, -1)
            masked_pixels = blurred[mask > 0]
            mean_val = float(masked_pixels.mean()) if masked_pixels.size else 255.0

            if mean_val < float(p["brightness_split"]):
                color = "schwarz"
                black_count += 1
            else:
                color = "weiss"
                white_count += 1
            colors.append(color)

    if len(centroids) > 0:
        sorted_idx = sorted(range(len(centroids)), key=lambda i: (centroids[i][0], centroids[i][1]))
        centroids = [centroids[i] for i in sorted_idx]
        colors = [colors[i] for i in sorted_idx]
        if detected_circles is not None:
            detected_circles = detected_circles[np.array(sorted_idx)]

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

    if detected_circles is None or len(centroids) == 0:
        if return_assignments:
            return frame, gray, blurred, thresh, assignments
        return frame, gray, blurred, thresh

    draw_detections(frame, detected_circles, colors, black_count, white_count)
    # if tracker is not None:
    #     draw_ids(frame, tracker)

    if draw_assignments and assignments:
        draw_assignment_labels(frame, assignments, font_scale=0.6)

    if return_assignments:
        return frame, gray, blurred, thresh, assignments
    return frame, gray, blurred, thresh


def run_live_parameter_tuning(
    *,
    camera_index: int = CAMERA_INDEX,
    window_name: str = "Figuren-Detektor (Live-Tuning)",
) -> None:
    """
    Live-Tuning fuer die Figuren-Detektion mit Trackbars.

    Zeigt das annotierte Bild sowie den Schwellwert- und Blur-Output.
    Druecke 'q' oder ESC, um zu beenden.
    """
    params_win = "Figure Params"
    thresh_win = "Figure Threshold"

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.namedWindow(thresh_win, cv2.WINDOW_NORMAL)
    cv2.namedWindow(params_win, cv2.WINDOW_NORMAL)

    def add_slider(name: str, init: int, maxval: int) -> None:
        cv2.createTrackbar(name, params_win, init, maxval, lambda _: None)
        cv2.setTrackbarPos(name, params_win, init)

    add_slider("blur_ksize", int(DEFAULT_FIGURE_PARAMS["blur_ksize"]), 31)  # Blur-Kernel (ungerade)
    add_slider("thresh_block", int(DEFAULT_FIGURE_PARAMS["thresh_block"]), 121)  # AdaptiveThreshold-Block
    add_slider("thresh_c", int(DEFAULT_FIGURE_PARAMS["thresh_c"]), 50)  # AdaptiveThreshold-C
    add_slider("min_radius", int(DEFAULT_FIGURE_PARAMS["min_radius"]), 120)  # Min-Kreisradius (px)
    add_slider("max_radius", int(DEFAULT_FIGURE_PARAMS["max_radius"]), 200)  # Max-Kreisradius (px)
    add_slider("hough_dp_x10", int(round(float(DEFAULT_FIGURE_PARAMS["hough_dp"]) * 10.0)), 40)  # dp*10
    add_slider("hough_min_dist", int(DEFAULT_FIGURE_PARAMS["hough_min_dist"]), 300)  # Min Zentrum-Abstand
    add_slider("hough_param1", int(DEFAULT_FIGURE_PARAMS["hough_param1"]), 400)  # Canny high threshold
    add_slider("hough_param2", int(DEFAULT_FIGURE_PARAMS["hough_param2"]), 200)  # Hough vote threshold
    add_slider("brightness_split", int(DEFAULT_FIGURE_PARAMS["brightness_split"]), 255)  # Schwarz/Weiss-Schwelle
    add_slider("save_config", 0, 1)  # Aktuelle Parameter in figure_detector.py speichern

    board_coords: dict[str, tuple[float, float]] | None = None
    assign_dist: float | None = None
    labels_order: list[str] | None = None

    calibration_loaded = False

    try:
        with open_camera(camera_index=camera_index) as cam:
            print("Kamera geoeffnet, starte Loop...", flush=True)
            while True:
                ret, frame = cam.read()
                if not ret:
                    logger.warning("Kameraframe konnte nicht gelesen werden, breche ab.")
                    break

                if not calibration_loaded:
                    try:
                        from gaming_robot_arm.vision.mill_board_detector import detect_board_positions
                        positions, _ = detect_board_positions(frame, debug=False)
                        if len(positions) == len(BOARD_LABELS):
                            detected = {lbl: (float(x), float(y)) for lbl, (x, y) in zip(BOARD_LABELS, positions)}
                            board_coords = _with_assignment_offset(detected)
                            assert board_coords is not None
                            assign_dist = estimate_assign_distance(board_coords)
                            labels_order = list(BOARD_LABELS)
                            calibration_loaded = True
                            logger.info("Live-Tuning: Brett live erkannt (%d Punkte).", len(board_coords))
                        # else: retry next frame
                    except Exception:
                        calibration_loaded = True  # stop retrying on hard error
                        logger.warning("Live-Tuning: Live-Brettdetektion fehlgeschlagen.", exc_info=True)

                raw_params = {
                    "blur_ksize": cv2.getTrackbarPos("blur_ksize", params_win),
                    "thresh_block": cv2.getTrackbarPos("thresh_block", params_win),
                    "thresh_c": cv2.getTrackbarPos("thresh_c", params_win),
                    "min_radius": cv2.getTrackbarPos("min_radius", params_win),
                    "max_radius": cv2.getTrackbarPos("max_radius", params_win),
                    "hough_dp": cv2.getTrackbarPos("hough_dp_x10", params_win) / 10.0,
                    "hough_min_dist": cv2.getTrackbarPos("hough_min_dist", params_win),
                    "hough_param1": cv2.getTrackbarPos("hough_param1", params_win),
                    "hough_param2": cv2.getTrackbarPos("hough_param2", params_win),
                    "brightness_split": cv2.getTrackbarPos("brightness_split", params_win),
                }
                live_params = normalize_figure_params(raw_params)

                # Normalisierung ggf. zurueck in die Slider schreiben (ungerade Kernel, max>=min).
                slider_sync = {
                    "blur_ksize": int(live_params["blur_ksize"]),
                    "thresh_block": int(live_params["thresh_block"]),
                    "max_radius": int(live_params["max_radius"]),
                }
                for slider, val in slider_sync.items():
                    if cv2.getTrackbarPos(slider, params_win) != val:
                        cv2.setTrackbarPos(slider, params_win, val)

                if cv2.getTrackbarPos("save_config", params_win) == 1:
                    if write_figure_params_to_detector(live_params):
                        DEFAULT_FIGURE_PARAMS.update(live_params)
                        print("Figure-Parameter in figure_detector_config.json gespeichert.")
                    cv2.setTrackbarPos("save_config", params_win, 0)

                processed, _, blurred, thresh = detect_figures(
                    frame,
                    board_coords=board_coords,
                    max_assign_dist=assign_dist,
                    labels_order=labels_order,
                    draw_assignments=bool(board_coords),
                    params=live_params,
                )
                if board_coords is not None:
                    for bx, by in board_coords.values():
                        cv2.circle(processed, (int(bx), int(by)), 4, (0, 255, 255), -1)

                overlay = (
                    f"blur={live_params['blur_ksize']} block={live_params['thresh_block']} "
                    f"C={live_params['thresh_c']} split={live_params['brightness_split']} "
                    f"| dp={float(live_params['hough_dp']):.1f} minDist={live_params['hough_min_dist']} "
                    f"p1={live_params['hough_param1']} p2={live_params['hough_param2']} "
                    f"| r={live_params['min_radius']}-{live_params['max_radius']} | q/esc=beenden"
                )
                cv2.putText(
                    processed,
                    overlay,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                cv2.imshow(window_name, processed)
                cv2.imshow(thresh_win, thresh)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    logger.info("Live-Tuning beendet.")
                    break
    finally:
        cv2.destroyAllWindows()


def run_live_assignment_test(
    *,
    camera_index: int = CAMERA_INDEX,
    max_assign_dist: float | None = None,
    window_name: str = "Figuren-Detektor (A1-C8)",
    use_stabilizer: bool = True,
    min_ratio: float = 0.5,
    min_samples: int | None = None,
    debug_assignments: bool = False,
    board_source: BoardCoordSource = "live",
) -> None:
    """
    Live-Test: markiert erkannte Figuren und versieht sie mit Feld-Labels (A1-C8).

    board_source:
    - calibration: nutzt gespeicherte Kalibrier-Brettpunkte.
    - live: nutzt pro Startframe den Mill-Board-Detector.
    - auto: vergleicht beide Quellen und waehlt robust die passendere.

    Druecke 'q', um die Anzeige zu beenden. Gewertet werden nur Figuren, die in
    mindestens min_ratio der letzten Sekunde erkannt wurden (nutzt FRAME_RATE).
    Wenn use_stabilizer=False, werden die Roh-Zuordnungen je Frame gezeichnet.
    """
    try:
        with open_camera(camera_index=camera_index) as cam:
            fps = get_effective_camera_fps(cam, configured_fps=FRAME_RATE)
            window_frames = max(1, int(round(fps)))
            min_samples = window_frames if min_samples is None else max(1, int(min_samples))

            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            ret0, frame0 = cam.read()
            if not ret0:
                logger.error("Kameraframe konnte nicht gelesen werden, breche ab.")
                return

            calibration_coords: dict[str, tuple[float, float]] | None = None
            labels_hint: list[str] | None = None
            board_coords, labels_order, active_source = _select_assignment_board_coords(
                frame0,
                calibration_coords=calibration_coords,
                labels_order=labels_hint,
                board_source=board_source,
            )
            if board_coords is None:
                logger.error("Keine Brettkoordinaten verfuegbar (board_source=%s).", board_source)
                return

            stabilizer = AssignmentStabilizer(labels_order, window=window_frames) if use_stabilizer else None
            assign_dist = max_assign_dist or estimate_assign_distance(board_coords)

            # Wenn die Kalibrierung klar nicht zur Aufloesung passt, ist Zuordnung sehr wahrscheinlich 0.
            xs = [p[0] for p in board_coords.values()]
            ys = [p[1] for p in board_coords.values()]
            inside_ratio = _inside_ratio(board_coords, width=frame0.shape[1], height=frame0.shape[0])
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
                "Starte Live-Test fuer Figuren-Zuordnung (quelle=%s, max_dist=%.1fpx). 'q' beendet.",
                active_source,
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
    board_pixels: Mapping[str, tuple[float, float]],
    attempts: int = 6,
    session: FigureDetectionSession | None = None,
    labels_order: Sequence[str] | None = None,
    camera_index: int = CAMERA_INDEX,
    *,
    debug_assignments: bool = False,
    board_source: BoardCoordSource = "live",
) -> List[Assignment]:
    """
    Nimmt mehrere Bilder auf und versucht, Figuren robust den Brettpositionen zuzuordnen.

    - Nutzt einen adaptiven Zuordnungsradius basierend auf den Brettabstaenden.
    - Erhoeht schrittweise die Distanzschwelle, falls keine Zuordnung gelingt.
    - Kann optional in eine laufende RecordingSession oder einen Live-Kamerakanal mitschreiben.
    - Ein Label zaehlt nur, wenn es in mindestens der Haelfte der betrachteten Frames
      innerhalb der letzten Sekunde erkannt wurde (FRAME_RATE bestimmt die Fensterlaenge).
    """
    labels_order = list(labels_order) if labels_order else sorted(board_pixels.keys())
    last_observation = BoardAssignmentObservation(
        stable_board=_empty_player_board(),
        stable_assignments=[],
        raw_assignments=[],
        frames_seen=0,
        board_source="none",
        ready=False,
    )

    try:
        with (
            open_camera(camera_index=camera_index) if session is None else nullcontext(session.camera)
        ) as cam:
            fps = get_effective_camera_fps(cam, configured_fps=FRAME_RATE)
            window_frames = max(1, int(round(fps)))
            stream = BoardAssignmentStream(
                board_pixels,
                labels_order=labels_order,
                board_source=board_source,
                window_frames=window_frames,
                min_samples=window_frames,
                min_ratio=0.5,
                debug_assignments=debug_assignments,
            )
            frames_to_capture = max(window_frames, attempts)

            for idx in range(frames_to_capture):
                ret, frame = cam.read()
                if not ret:
                    logger.warning("Kameraframe konnte nicht gelesen werden (Versuch %s).", idx + 1)
                    continue
                if session is not None:
                    session.write(frame)
                last_observation = stream.process_frame(frame)

    except RuntimeError as exc:
        logger.warning("Kamera konnte nicht geoeffnet werden: %s", exc)
        return []

    if not last_observation.ready:
        return []
    return last_observation.stable_assignments


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Live-Tuning fuer Figuren-Detektion; mit --assignments startet der Live-Test fuer Brettzuordnung.",
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
        default=None,
        help="Fenstertitel fuer die Live-Ansicht (modusabhaengiger Standard, falls nicht gesetzt).",
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
    parser.add_argument(
        "--board-source",
        choices=("auto", "calibration", "live"),
        default="live",
        help="Quelle fuer Brettkoordinaten bei Figuren-Zuordnung.",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--assignments",
        action="store_true",
        help="Startet den Live-Test mit Brettzuordnung statt des Default-Live-Tunings.",
    )
    mode_group.add_argument(
        "--tune",
        action="store_true",
        help="Alias fuer den Default-Modus mit Live-Trackbars.",
    )
    args = parser.parse_args()

    if args.assignments:
        run_live_assignment_test(
            camera_index=args.camera_index,
            max_assign_dist=args.max_assign_dist,
            window_name=args.window_name or "Figuren-Detektor (A1-C8)",
            use_stabilizer=not args.raw,
            min_ratio=args.min_ratio,
            min_samples=args.min_samples,
            debug_assignments=args.debug_assignments,
            board_source=args.board_source,
        )
        return

    run_live_parameter_tuning(
        camera_index=args.camera_index,
        window_name=args.window_name or "Figuren-Detektor (Live-Tuning)",
    )


if __name__ == "__main__":
    print("Starte...", flush=True)
    main()
