from __future__ import annotations

"""
Rastert den uArm-Arbeitsraum und protokolliert die maximale X-Reichweite
fuer jede (y, z)-Kombination. Nutzt nur check_pos_is_limit, es werden
keine echten Bewegungen ausgefuehrt.
"""

import json
from pathlib import Path
from typing import List, Optional

try:
    from control.uarm_controller import UArmController
except ModuleNotFoundError:
    import pathlib
    import sys

    repo_root = pathlib.Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    from control.uarm_controller import UArmController


def measure_workspace(
    start_x: int = 400,
    start_z: int = 0,
    x_step: int = -1,
    y_step: int = 1,
    z_step: int = 1,
    x_backoff: int = 5,
) -> tuple[List[List[Optional[int]]], List[List[Optional[int]]], bool]:
    """
    Durchlaeuft z aufwaerts, y aufwaerts, x abwaerts und merkt sich fuer jedes
    (y, z) den ersten legalen x-Wert. Stoppt, sobald fuer einen z-Wert keine
    legale (x, y)-Kombination existiert.
    """
    controller = UArmController()
    swift = controller.connect()

    # Stelle sicher, dass der x-Schritt negativ ist (abwaerts suchen).
    x_step_neg = x_step if x_step < 0 else -abs(x_step) if x_step != 0 else -1

    columns: List[List[Optional[int]]] = []
    z = start_z
    aborted = False
    max_legal_range: List[List[Optional[int]]] = []
    min_legal_range: List[List[Optional[int]]] = []
    legal_y = 0
    legal_z = 0
    try:
        while True:
            y = 0
            prev_x_start = start_x
            column: List[Optional[int]] = []
            found_any = False

            while True:
                x = prev_x_start
                legal_x: Optional[int] = None

                while x >= 0:
                    allowed = swift.check_pos_is_limit(pos=[x, y, z])
                    # check_pos_is_limit gibt False zurueck, wenn die Position erlaubt ist
                    if allowed is False:
                        legal_x = x
                        break
                    x += x_step_neg

                if legal_x is None:
                    break

                found_any = True
                column.append(legal_x)
                print(f"z={z:03d}, y={y:03d} -> legal_x={legal_x}", flush=True)
                y += y_step
                prev_x_start = legal_x + x_backoff
            else:
                # Keine legale Position mehr fuer dieses z
                print(f"z={z:03d}, y={y:03d} -> legal_x=None (stoppe y-Schleife)", flush=True)

            if not found_any:
                break

            columns.append(column)
            z += z_step

        # Max-Range in rechteckiges Array schreiben
        legal_y = max((len(col) for col in columns), default=0)
        legal_z = len(columns)
        max_legal_range = [[None for _ in range(legal_z)] for _ in range(legal_y)]
        for z_idx, col in enumerate(columns):
            for y_idx, x_val in enumerate(col):
                max_legal_range[y_idx][z_idx] = x_val

        # Min-Range ermitteln: fuer jede (y, z) ab x=0 aufwaerts suchen
        min_legal_range = [[None for _ in range(legal_z)] for _ in range(legal_y)]
        x_step_pos = abs(x_step_neg)
        for z_idx in range(legal_z):
            z_val = start_z + z_idx * z_step
            col = columns[z_idx]
            for y_idx, max_x in enumerate(col):
                if max_x is None:
                    continue
                y_val = y_idx * y_step
                x = 0
                min_val: Optional[int] = None
                while x <= max_x:
                    allowed = swift.check_pos_is_limit(pos=[x, y_val, z_val])
                    # check_pos_is_limit gibt False zurueck, wenn die Position erlaubt ist
                    if allowed is False:
                        min_val = x
                        break
                    x += x_step_pos
                min_legal_range[y_idx][z_idx] = min_val
    except KeyboardInterrupt:
        aborted = True
        print("Abbruch durch Benutzer, bisherige Daten werden gespeichert.", flush=True)
    finally:
        controller.disconnect()

    return max_legal_range, min_legal_range, aborted


if __name__ == "__main__":
    max_legal_range, min_legal_range, aborted = measure_workspace()
    legal_y = len(max_legal_range)
    legal_z = len(max_legal_range[0]) if max_legal_range else 0
    print(f"legal_y={legal_y}, legal_z={legal_z}, aborted={aborted}")
    print("max_legal_range[y][z] = x_max (None => keine Position)")
    for y_idx, row in enumerate(max_legal_range):
        print(f"y={y_idx:03d}: {row}")
    print("min_legal_range[y][z] = x_min (None => keine Position)")
    for y_idx, row in enumerate(min_legal_range):
        print(f"y={y_idx:03d}: {row}")

    # Ausgabe in Datei "max_legal_range" im data-Verzeichnis ablegen
    data_dir = Path(__file__).resolve().parents[1] / "data" / "movement_ranges"
    data_dir.mkdir(parents=True, exist_ok=True)
    out_path = data_dir / "max_legal_range"
    payload = {
        "legal_y": legal_y,
        "legal_z": legal_z,
        "aborted": aborted,
        "max_legal_range": max_legal_range,
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)
    print(f"Max-Ergebnis gespeichert in {out_path}")

    out_path_min = data_dir / "min_legal_range"
    payload_min = {
        "legal_y": legal_y,
        "legal_z": legal_z,
        "aborted": aborted,
        "min_legal_range": min_legal_range,
    }
    with out_path_min.open("w", encoding="utf-8") as f:
        json.dump(payload_min, f, ensure_ascii=True, indent=2)
    print(f"Min-Ergebnis gespeichert in {out_path_min}")
