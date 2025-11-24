# board_detector.py
import cv2
import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class Line:
    """Repraesentiert eine erkannte Brettlinie mit Orientierung, Steigung und Achsenabschnitt."""

    orientation: str  # 'H' = horizontal dominant, 'V' = vertical dominant
    slope: float
    intercept: float

    @property
    def anchor(self) -> float:
        """Liefert den Sortierwert der Linie (y@x=0 fuer horizontale, x@y=0 fuer vertikale Linien)."""
        return self.intercept

# ---- Pfad für Standalone-Test ----
IMG_PATH = r"C:\Users\nando\OneDrive\Documents\Uni\Bachelor\PP-BA\Praxisprojekt\Graphics\Screenshots\bw_board.png"


# ---------- Hilfsfunktionen ----------
def angle_deg(x1, y1, x2, y2):
    """Berechnet den Winkel eines Segmentes in Grad relativ zur positiven x-Achse."""
    return np.degrees(np.arctan2((y2 - y1), (x2 - x1)))


def separate_segments_by_orientation(segments, tolerance_deg):
    """Teilt Segmentkoordinaten in nahezu horizontale und vertikale Gruppen nach Winkeltoleranz auf."""
    horizontals, verticals = [], []
    for (x1, y1, x2, y2) in segments:
        a = angle_deg(x1, y1, x2, y2)
        a = (a + 180.0) % 180.0  # 0..180
        if min(abs(a - 0), abs(a - 180)) <= tolerance_deg:
            horizontals.append((x1, y1, x2, y2))
        elif min(abs(a - 90), abs(a - 270)) <= tolerance_deg:
            verticals.append((x1, y1, x2, y2))
    return horizontals, verticals

def peak_centers_from_hist(vals, bins, want_k=7, nms=2):
    """Bestimme Peak-Zentren eines 1D-Histogramms via Non-Max-Suppression."""
    if len(vals) == 0:
        return []

    hist, edges = np.histogram(vals, bins=bins)
    if not np.any(hist):
        return []

    sorted_idx = np.argsort(hist)[::-1]
    suppressed = np.zeros_like(hist, dtype=bool)
    peaks = []

    for idx in sorted_idx:
        if hist[idx] <= 0 or suppressed[idx]:
            continue
        peaks.append(idx)
        if len(peaks) == want_k:
            break

        lo = max(0, idx - nms)
        hi = min(len(hist), idx + nms + 1)
        suppressed[lo:hi] = True

    if not peaks:
        return []

    peaks = np.array(peaks, dtype=int)
    centers = 0.5 * (edges[peaks] + edges[peaks + 1])
    centers.sort()
    return centers.tolist()

def fit_line_from_segments(segments, vertical=False):
    """Fittet eine Ausgleichsgerade aus den Segmentpunkten (horizontal bzw. vertikal dominiert)."""
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

def draw_infinite_line(img, line, color, thickness=2):
    """Zeichnet die berechnete Linie als unendliche Verlaengerung quer durch das Bild."""
    h, w = img.shape[:2]
    m, b = line.slope, line.intercept
    if line.orientation == 'H':
        x1, x2 = 0, w - 1
        y1 = int(round(m * x1 + b))
        y2 = int(round(m * x2 + b))
        cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    else:
        y1, y2 = 0, h - 1
        x1 = int(round(m * y1 + b))
        x2 = int(round(m * y2 + b))
        cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def line_anchor_value(line):
    """Sortier-Skalar je Linie: y@x=0 (horizontale) bzw. x@y=0 (vertikale)."""
    return line.intercept

def ensure_lines(lines, expected=7, axis='H'):
    """
    Passt die Anzahl der Linien auf den Erwartungswert an, ohne vorhandene Linien zu verschieben.
    Fehlende Linien werden lokal ergänzt, überschüssige Linien sanft ausgedünnt.
    """
    orientation = 'H' if axis == 'H' else 'V'
    sorted_lines = sorted(lines, key=line_anchor_value)

    if len(sorted_lines) == expected:
        return sorted_lines

    if not sorted_lines:
        # Voller Fallback: künstliche Gitterlinien alle 50px
        return [Line(orientation, 0.0, float(i * 50)) for i in range(expected)]

    if len(sorted_lines) < expected:
        lines_adjusted = sorted_lines.copy()

        def insert_between(idx):
            l1, l2 = lines_adjusted[idx], lines_adjusted[idx + 1]
            new_anchor = 0.5 * (l1.anchor + l2.anchor)
            new_slope = 0.5 * (l1.slope + l2.slope)
            lines_adjusted.insert(idx + 1, Line(orientation, float(new_slope), float(new_anchor)))

        while len(lines_adjusted) < expected:
            anchors = np.array([ln.anchor for ln in lines_adjusted], dtype=float)
            gaps = np.diff(anchors)

            if len(gaps) == 0:
                # Nur eine Linie vorhanden → künstliche Nachbarn ober/unterhalb erzeugen
                offset = 40.0 if axis == 'H' else 40.0
                delta = offset * (1 if len(lines_adjusted) % 2 == 0 else -1)
                base = anchors[0] + delta
                lines_adjusted.append(Line(orientation, float(lines_adjusted[0].slope), float(base)))
                lines_adjusted = sorted(lines_adjusted, key=line_anchor_value)
                continue

            # Größte Lücke suchen und dort eine zusätzliche Linie einsetzen
            idx = int(np.argmax(gaps))
            insert_between(idx)

        return lines_adjusted

    # len(sorted_lines) > expected → reale Linien mittels Gruppierung auswählen
    indices = [int(group[len(group) // 2]) for group in np.array_split(np.arange(len(sorted_lines)), expected)]
    return [sorted_lines[idx] for idx in indices]

def intersect(hline, vline):
    """Berechnet den Schnittpunkt einer horizontalen und einer vertikalen Linie."""
    mh, bh = hline.slope, hline.intercept
    mv, bv = vline.slope, vline.intercept
    A = 1.0 - mh * mv
    B = mh * bv + bh
    if abs(A) < 1e-9:
        # Parallel / numerisch instabil → fallback
        y = bh
        x = mv * y + bv
    else:
        y = B / A
        x = mv * y + bv
    return int(round(x)), int(round(y))


# ---------- Hauptfunktion ----------
def detect_board_positions(frame_bgr, debug=True):
    """Detektiert ein Muehle-Brett, liefert alle 24 Spielfeldpunkte und ein optional beschriftetes Bild."""
    H_img, W_img = frame_bgr.shape[:2]
    annotated = frame_bgr.copy()

    # 1) Vorverarbeitung
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny adaptiv aus Median
    v = np.median(blur)
    lower = int(max(0, 0.66 * v))
    upper = int(min(255, 1.33 * v))
    edges = cv2.Canny(blur, lower, upper, apertureSize=3)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    # 2) Probabilistic Hough → echte Segmente
    segs = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=80, minLineLength=max(W_img, H_img) // 7, maxLineGap=15
    )
    if segs is None:
        print("⚠️ Keine Linien erkannt.")
        return [], annotated

    # Rohsegmente (grün)
    for (x1, y1, x2, y2) in segs[:, 0]:
        cv2.line(annotated, (x1, y1), (x2, y2), (0, 255, 0), 1)

    # 3) Winkelbasierte Selektion: nahezu horizontal / vertikal
    horiz_segs, vert_segs = separate_segments_by_orientation(segs[:, 0], tolerance_deg=7.0)

    # Sanft lockern, falls nötig
    if len(horiz_segs) < 7 or len(vert_segs) < 7:
        horiz_segs, vert_segs = separate_segments_by_orientation(segs[:, 0], tolerance_deg=10.0)

    # 4) 1D-Histogramm der Segmentlagen (Mittelpunkte) → 7 Peaks
    y_mids = [0.5 * (y1 + y2) for (_, y1, _, y2) in horiz_segs]
    x_mids = [0.5 * (x1 + x2) for (x1, _, x2, _) in vert_segs]

    H_centers = peak_centers_from_hist(y_mids, bins=max(14, H_img // 20), want_k=7, nms=2)
    V_centers = peak_centers_from_hist(x_mids, bins=max(14, W_img // 20), want_k=7, nms=2)

    # 5) Segmente den Peaks zuordnen und pro Peak Linie fitten (Least-Squares)
    def group_by_peak(segments, centers, vertical=False):
        """Gruppiert Segmente anhand des naechstliegenden Peak-Zentrums."""
        groups = [[] for _ in centers]
        for seg in segments:
            if not centers:
                continue
            if not vertical:
                val = 0.5 * (seg[1] + seg[3])
            else:
                val = 0.5 * (seg[0] + seg[2])
            i = int(np.argmin([abs(val - c) for c in centers]))
            groups[i].append(seg)
        return groups

    H_groups = group_by_peak(horiz_segs, H_centers, vertical=False)
    V_groups = group_by_peak(vert_segs,  V_centers, vertical=True)

    H_lines = []
    for g in H_groups:
        if len(g) == 0: continue
        ln = fit_line_from_segments(g, vertical=False)
        H_lines.append(ln)
        draw_infinite_line(annotated, ln, (255, 255, 0), 1)  # cyan: fit je Peak

    V_lines = []
    for g in V_groups:
        if len(g) == 0: continue
        ln = fit_line_from_segments(g, vertical=True)
        V_lines.append(ln)
        draw_infinite_line(annotated, ln, (255, 255, 0), 1)

    # Nach Ankerwert sortieren (oben→unten, links→rechts)
    H_lines = sorted(H_lines, key=line_anchor_value)
    V_lines = sorted(V_lines, key=line_anchor_value)

    # 6) Sicherstellen, dass genau 7 Linien pro Richtung existieren (robuster Fallback)
    H_lines = ensure_lines(H_lines, expected=7, axis='H')
    V_lines = ensure_lines(V_lines, expected=7, axis='V')

    # Finale Linien (blau)
    for ln in H_lines: draw_infinite_line(annotated, ln, (255, 0, 0), 2)
    for ln in V_lines: draw_infinite_line(annotated, ln, (255, 0, 0), 2)

   # 7) 24 gültige Mühle-Positionen --------------------
    # Linien-Indices:
    # H0–H6, V0–V6 (da Python 0-indiziert)
    outer = (0, 6)
    middle = (1, 5)
    inner = (2, 4)
    center = 3  # mittlere Verbindungsachse

    positions = []

    # --- Äußeres Quadrat (8 Punkte) ---
    positions += [
        intersect(H_lines[outer[0]], V_lines[outer[0]]),  # oben links
        intersect(H_lines[outer[0]], V_lines[outer[1]]),  # oben rechts
        intersect(H_lines[outer[1]], V_lines[outer[0]]),  # unten links
        intersect(H_lines[outer[1]], V_lines[outer[1]]),  # unten rechts
        intersect(H_lines[outer[0]], V_lines[center]),    # Mitte oben
        intersect(H_lines[outer[1]], V_lines[center]),    # Mitte unten
        intersect(H_lines[center], V_lines[outer[0]]),    # Mitte links
        intersect(H_lines[center], V_lines[outer[1]])     # Mitte rechts
    ]

    # --- Mittleres Quadrat (8 Punkte) ---
    positions += [
        intersect(H_lines[middle[0]], V_lines[middle[0]]),
        intersect(H_lines[middle[0]], V_lines[middle[1]]),
        intersect(H_lines[middle[1]], V_lines[middle[0]]),
        intersect(H_lines[middle[1]], V_lines[middle[1]]),
        intersect(H_lines[middle[0]], V_lines[center]),
        intersect(H_lines[middle[1]], V_lines[center]),
        intersect(H_lines[center], V_lines[middle[0]]),
        intersect(H_lines[center], V_lines[middle[1]])
    ]

    # --- Inneres Quadrat (8 Punkte) ---
    positions += [
        intersect(H_lines[inner[0]], V_lines[inner[0]]),
        intersect(H_lines[inner[0]], V_lines[inner[1]]),
        intersect(H_lines[inner[1]], V_lines[inner[0]]),
        intersect(H_lines[inner[1]], V_lines[inner[1]]),
        intersect(H_lines[inner[0]], V_lines[center]),
        intersect(H_lines[inner[1]], V_lines[center]),
        intersect(H_lines[center], V_lines[inner[0]]),
        intersect(H_lines[center], V_lines[inner[1]])
    ]


    # === Koordinatensystem A1–C8 auf Basis der 7+7 Linien ===
    # Indizes: H0..H6 (oben→unten), V0..V6 (links→rechts); center = 3
    outer  = (0, 6)
    middle = (1, 5)
    inner  = (2, 4)
    center = 3

    def quad_positions(H_lines, V_lines, iT, iB, iL, iR, iC):
        """Liefert acht Eck- und Kantenpunkte eines Quadrats in festgelegter Reihenfolge."""
        return [
            intersect(H_lines[iT], V_lines[iL]),    # 1
            intersect(H_lines[iT], V_lines[iC]),    # 2
            intersect(H_lines[iT], V_lines[iR]),    # 3
            intersect(H_lines[iC], V_lines[iR]),    # 4
            intersect(H_lines[iB], V_lines[iR]),    # 5
            intersect(H_lines[iB], V_lines[iC]),    # 6
            intersect(H_lines[iB], V_lines[iL]),    # 7
            intersect(H_lines[iC], V_lines[iL]),    # 8
    ]

    def build_coords_dict(H_lines, V_lines):
        """Baut ein Dictionary von Spielfeld-Labels auf die berechneten Koordinaten."""
        coords = {}
        for label, (iT, iB, iL, iR) in zip(
            ["A","B","C"],
            [outer + (outer[0], outer[1]),
            middle + (middle[0], middle[1]),
            inner  + (inner[0],  inner[1])]
        ):
            pts = quad_positions(H_lines, V_lines, iT, iB, iL, iR, center)
            for i, (x,y) in enumerate(pts, start=1):
                coords[f"{label}{i}"] = (int(x), int(y))
        return coords

    def draw_labels(frame, coords_dict):
        """Visualisiert alle gefundenen Positionen als Punkte mit Textlabel im Bild."""
        out = frame.copy()
        for key, (x,y) in coords_dict.items():
            cv2.circle(out, (x,y), 6, (0,0,255), -1)                   # Punkt
            cv2.putText(out, key, (x-20, y-10),                        # Label
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 164, 0), 2, cv2.LINE_AA)
        return out

    coords = build_coords_dict(H_lines, V_lines)
    annotated = draw_labels(annotated, coords)   # annotated ist dein Debug-Frame
    print(f"{len(coords)} Brettpositionen (sollte 24 sein)")


    # 8) Debug: Punkte zeichnen
    # for (x, y) in positions:
    #     cv2.circle(annotated, (int(x), int(y)), 5, (0, 0, 255), -1)

    return positions, annotated


# ---------- Standalone-Test ----------
if __name__ == "__main__":
    try:
        from vision.recording import open_camera
    except ModuleNotFoundError:
        # Allow running the file directly by adding the project root to sys.path
        import sys
        from pathlib import Path

        project_root = Path(__file__).resolve().parents[1]
        if str(project_root) not in sys.path:
            sys.path.append(str(project_root))
        from vision.recording import open_camera

    # Debug-Option: einen Live-Frame der Kamera einlesen (bei Bedarf einkommentieren)
    with open_camera() as cam:
        ok, frame = cam.read()
        if not ok:
            raise RuntimeError("Kameraframe konnte nicht gelesen werden.")
        img = frame
    
    # Alternativ: statisches Testbild laden
    # img = cv2.imread(IMG_PATH)
    
    if img is None:
        raise RuntimeError(f"Bild nicht gefunden: {IMG_PATH}")
    
    pts, vis = detect_board_positions(img, debug=True)
    
    print(f"Gefundene Brett-Positionen: {len(pts)}")
    cv2.imshow("Board Detection Debug", vis)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
