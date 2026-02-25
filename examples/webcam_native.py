#!/usr/bin/env python3
"""
Oeffnet eine verbundene Webcam in nativer Aufloesung und zeigt den Stream.
Keine Projekt-Imports oder Abhaengigkeiten ausser OpenCV.
"""

import argparse
import sys


def _resolve_backend(cv2, backend_name: str):
    backends = {
        "auto": None,
        "any": cv2.CAP_ANY,
        "dshow": cv2.CAP_DSHOW,
        "msmf": cv2.CAP_MSMF,
        "v4l2": cv2.CAP_V4L2,
    }
    if backend_name not in backends:
        raise ValueError(f"Unbekanntes Backend: {backend_name}")
    if backend_name == "auto":
        if sys.platform.startswith("win"):
            return cv2.CAP_MSMF
        return cv2.CAP_ANY
    return backends[backend_name]


def _open_capture(cv2, index: int, backend):
    if backend is None or backend == cv2.CAP_ANY:
        return cv2.VideoCapture(index)
    return cv2.VideoCapture(index, backend)


def _backend_label(cv2, backend):
    if backend is None:
        return "auto"
    if backend == cv2.CAP_ANY:
        return "any"
    if backend == cv2.CAP_DSHOW:
        return "dshow"
    if backend == cv2.CAP_MSMF:
        return "msmf"
    if backend == cv2.CAP_V4L2:
        return "v4l2"
    return str(backend)


def _candidate_backends(cv2, preferred):
    if sys.platform.startswith("win"):
        order = [preferred, cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY]
    else:
        order = [preferred, cv2.CAP_ANY]
    seen = set()
    result = []
    for backend in order:
        key = backend if backend is not None else "auto"
        if key in seen:
            continue
        seen.add(key)
        result.append(backend)
    return result


def _open_capture_with_fallback(cv2, index: int, preferred):
    for backend in _candidate_backends(cv2, preferred):
        cap = _open_capture(cv2, index, backend)
        if cap.isOpened():
            return cap, backend
        cap.release()
    return None, None


def _detect_cameras(cv2, scan_max: int, backend):
    available = []
    for idx in range(scan_max):
        cap = _open_capture(cv2, idx, backend)
        if not cap.isOpened():
            cap.release()
            continue
        ok, frame = cap.read()
        if ok and frame is not None:
            height, width = frame.shape[:2]
            available.append((idx, width, height))
        cap.release()
    return available


def _prompt_for_index(available):
    print("Mehrere Webcams erkannt:")
    for idx, width, height in available:
        print(f"  [{idx}] {width}x{height}")
    while True:
        choice = input("Kamera-Index waehlen: ").strip()
        try:
            idx = int(choice)
        except ValueError:
            print("Bitte einen gueltigen ganzzahligen Index eingeben.")
            continue
        if any(idx == item[0] for item in available):
            return idx
        print("Index nicht in der erkannten Liste. Bitte erneut versuchen.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Oeffnet eine Webcam in nativer Aufloesung und zeigt den Stream."
    )
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="Kamera-Index (wenn weggelassen: erkennen und bei mehreren Auswahl abfragen)",
    )
    parser.add_argument(
        "--scan-max",
        type=int,
        default=10,
        help="Wie viele Indizes bei Auto-Erkennung geprueft werden (Standard: 10)",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "any", "dshow", "msmf", "v4l2"],
        default="auto",
        help="Zu verwendendes Video-Backend (Standard: auto)",
    )
    args = parser.parse_args()

    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover - runtime dependency
        print("OpenCV wird benoetigt. Installation: pip install opencv-python", file=sys.stderr)
        print(f"Import-Fehler: {exc}", file=sys.stderr)
        return 1

    backend = _resolve_backend(cv2, args.backend)
    scan_backend = backend
    if backend == cv2.CAP_DSHOW and sys.platform.startswith("win"):
        # DSHOW kann beim Scannen indexbezogene Capture-Warnungen ausgeben.
        scan_backend = cv2.CAP_MSMF

    selected_index = args.index
    if selected_index is None:
        available = _detect_cameras(cv2, args.scan_max, scan_backend)
        if not available:
            print("Keine Webcams erkannt.", file=sys.stderr)
            return 2
        if len(available) == 1:
            selected_index = available[0][0]
        else:
            selected_index = _prompt_for_index(available)

    cap, used_backend = _open_capture_with_fallback(cv2, selected_index, backend)
    if cap is None or not cap.isOpened():
        print(f"Konnte Webcam mit Index {selected_index} nicht oeffnen", file=sys.stderr)
        return 2
    if used_backend != backend:
        print(
            f"Weiche auf Backend '{_backend_label(cv2, used_backend)}' aus.",
            file=sys.stderr,
        )

    # Fordert 1920x1080 an; Kamera/Treiber koennen den naechstliegenden Modus waehlen.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    # Liest einen Frame zur Bestimmung der tatsaechlichen Aufloesung.
    ok, frame = cap.read()
    if not ok or frame is None:
        print("Konnte keinen Frame von der Webcam lesen.", file=sys.stderr)
        cap.release()
        return 3

    height, width = frame.shape[:2]
    print(
        f"Webcam geoeffnet mit {width}x{height} "
        f"(backend: {_backend_label(cv2, used_backend)})"
    )

    window_name = "Webcam (q zum Beenden)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        if frame is None:
            ok, frame = cap.read()
            if not ok:
                break
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        ok, frame = cap.read()
        if not ok:
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
