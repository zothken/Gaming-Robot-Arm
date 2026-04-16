# Gaming Robot Arm

Dieses Projekt verbindet Computer Vision mit einem uArm Swift Pro: Eine Kamera erkennt Spielsteine auf einem Brett, ordnet sie Feldern zu und der Roboter kann Figuren sicher aufnehmen und umsetzen.

## Gesamtarchitektur

Die Laufzeit teilt sich in Vision, Kalibrierung und Robotersteuerung. Die Kalibrierung liefert die Abbildung von Pixeln auf Roboterkoordinaten, die Runtime verarbeitet Frames und kann Bewegungen ausloesen.

```
Kamera
  -> gaming_robot_arm.vision.recording (Kamera-Stream + Videoaufnahme)
  -> gaming_robot_arm.vision.figure_detector (Kreiserkennung + Farbklassifikation)
  -> gaming_robot_arm.runtime (Loop + Handler)
  -> gaming_robot_arm.control.UArmController (uArm Swift API)

Kalibrierung:
  gaming_robot_arm.vision.mill_board_detector -> gaming_robot_arm.calibration.calibration -> gaming_robot_arm/calibration/*.json
  gaming_robot_arm.utils.homography (img_to_robot) nutzt die gespeicherte Homography
```

## Module und Dateien

### Projektwurzel

| Modul/Datei | Funktion |
| --- | --- |
| `main.py` | Startpunkt/Launcher mit Modi fuer Vision-Loop und spielbare Mill-Partie. |
| `gaming_robot_arm/` | Python-Paket (Runtime, Vision, Control, Utils, Kalibrierung). |
| `pyproject.toml` | Paket-Metadaten und Python-Version (>=3.10). |
| `requirements.txt` | Core-Abhaengigkeiten (inkl. editable Paket + uArm-SDK von GitHub). |
| `requirements-ml.txt` | Optionale ML-Abhaengigkeiten (PyTorch fuer Neural-Mill-Training/Inferenz). |

### Paket `gaming_robot_arm/`

| Modul/Datei | Funktion |
| --- | --- |
| `gaming_robot_arm/config.py` | Zentrale Einstellungen fuer Kamera, uArm-Grenzen, Pfade und Board-Parameter. |
| `gaming_robot_arm/runtime.py` | Orchestriert Kamera-Loop, Detection und optionale Robotik. |

### Paket `gaming_robot_arm/utils/`

| Modul/Datei | Funktion |
| --- | --- |
| `gaming_robot_arm/utils/homography.py` | Laden/Umrechnen Pixel -> Roboterkoordinaten. |
| `gaming_robot_arm/utils/logger.py` | Logging-Setup fuer alle Module. |
| `gaming_robot_arm/utils/timing.py` | FPS-Tracker fuer Loop-Diagnose. |

### Paket `gaming_robot_arm/calibration/`

| Modul/Datei | Funktion |
| --- | --- |
| `gaming_robot_arm/calibration/calibration.py` | Interaktive Erfassung von Brett-Pixeln und Homography-Fit. |

### Paket `gaming_robot_arm/vision/`

| Modul/Datei | Funktion |
| --- | --- |
| `gaming_robot_arm/vision/figure_detector.py` | Erkennung runder Figuren, Farbklassifikation, Zuordnung zu Brettlabels. |
| `gaming_robot_arm/vision/mill_board_detector.py` | Brettlinien-Detektion und Schnittpunkte (A1-C8) fuer die Kalibrierung. |
| `gaming_robot_arm/vision/recording.py` | Kamera-Handling, Frame-Lesen, MP4-Aufzeichnung, Live-Preview. |
| `gaming_robot_arm/vision/visualization.py` | Zeichnet Detections, IDs und Debug-Frames. |

### Paket `gaming_robot_arm/control/`

| Modul/Datei | Funktion |
| --- | --- |
| `gaming_robot_arm/control/uarm_controller.py` | Abstraktions-Wrapper der uArm Swift API mit sicheren Bewegungen und Grenzen. |

### Paket `gaming_robot_arm/games/`

| Modul/Datei | Funktion |
| --- | --- |
| `gaming_robot_arm/games/common/interfaces.py` | Gemeinsame Schnittstellen fuer Spiel-Logik. |
| `gaming_robot_arm/games/mill/core/board.py` | Brett-Labels, Nachbarschaften und Mill-Linien. |
| `gaming_robot_arm/games/mill/core/rules.py` | Regeln fuer Nine Men's Morris (Mill). |
| `gaming_robot_arm/games/mill/core/settings.py` | Umschaltbare Regel-Einstellungen (z.B. Flying, Remis-Regeln) fuer spaeteres GUI-Menue. |
| `gaming_robot_arm/games/mill/core/session.py` | Sitzungscontainer fuer Zustand + Zughistorie inkl. KI-Anbindung. |
| `gaming_robot_arm/games/mill/ai/builtin.py` | Interne KIs (Heuristik + Alpha-Beta, ohne externe Engine/Installationen). |
| `gaming_robot_arm/games/mill/runtime/game_loop.py` | Spielbare Kommandozeilen-Partie mit Moduswahl und optionaler Vision/Roboter-Anbindung. |
| `gaming_robot_arm/games/mill/core/state.py` | Zustandscontainer fuer Mill. |

### Paket `examples/`

| Modul/Datei | Funktion |
| --- | --- |
| `examples/move_uArm.py` | Interaktives Bewegen (Koordinaten oder Brettlabel) inkl. optionaler Aufnahme. |
| `examples/move_figures.py` | Figur aufnehmen und zwischen zwei Brettpositionen umsetzen. |

### Daten und Ausgaben

| Modul/Datei | Funktion |
| --- | --- |
| `gaming_robot_arm/calibration/*.json` | Kalibrierungsdaten (Board-Pixel und Homography). |
| `Aufnahmen/` | Standardziel fuer Videoaufnahmen der Runtime. |

### Paket `gaming_robot_arm/old/`

| Modul/Datei | Funktion |
| --- | --- |
| `gaming_robot_arm/old/tracker.py` | Altes Tracking (wird derzeit nicht von der Runtime genutzt). |
| `gaming_robot_arm/old/board_detector.py` | Alternative/ausfuehrliche Brettlinien-Detektion (Hough + Geometrie). |

## Installation

1. **Voraussetzungen**
   - Python >= 3.10 inkl. `pip` (Pruefung: `python --version`).
   - Git (zum Klonen des Repos).
   - uArm Swift Pro via USB (Treiber/Seriell-Port muss vom Betriebssystem erkannt werden).
   - Kamera (USB/HDMI), die von OpenCV gelesen werden kann.

2. **Repository klonen**

   ```bash
   git clone <REPO-URL> gaming-robot-arm
   cd gaming-robot-arm
   ```

3. **Virtuelle Umgebung erstellen (empfohlen)**

   ```bash
   python -m venv .venv
   ```

   Aktivieren:

   - Windows (PowerShell):

     ```bash
     .venv\Scripts\Activate.ps1
     ```

   - Windows (cmd):

     ```bash
     .venv\Scripts\activate.bat
     ```

   - Linux/macOS:

     ```bash
     source .venv/bin/activate
     ```

4. **Paketwerkzeuge aktualisieren**

   ```bash
   python -m pip install --upgrade pip setuptools wheel
   ```

5. **Kernabhaengigkeiten installieren**

   ```bash
   python -m pip install -r requirements.txt
   ```

   Hinweis: `requirements.txt` installiert das uArm SDK direkt von GitHub. Dazu ist Netzwerkzugriff erforderlich.

6. **Optional: ML-Abhaengigkeiten installieren (nur fuer Neural Mill)**

   ```bash
   # Standard (kann CUDA-Pakete mitziehen):
   python -m pip install -r requirements-ml.txt
   ```

   Linux nur CPU (kleiner/fuer CI oft sinnvoll):

   ```bash
   python -m pip install --index-url https://download.pytorch.org/whl/cpu -r requirements-ml.txt
   ```

7. **Projekt als Paket installieren (empfohlen)**

   ```bash
   python -m pip install -e .
   ```

   Dadurch funktionieren Imports wie `gaming_robot_arm.config` oder `gaming_robot_arm.calibration` auch beim direkten Start von Skripten.

8. **Hardware verbinden und Ports pruefen**
   - Kamera anschliessen und im OS pruefen (z.B. Kamera-App).
   - uArm per USB anschliessen. Falls der Port nicht automatisch erkannt wird, `UARM_PORT` in `gaming_robot_arm/config.py` setzen.

9. **Projekt konfigurieren**
   - `gaming_robot_arm/config.py` anpassen:
     - `CAMERA_INDEX`, `FRAME_WIDTH`/`FRAME_HEIGHT` (optional; `None` = native Kameraaufloesung), `FRAME_RATE` (optional; `None` = native Kamera-FPS)
     - `SAFE_Z`, `REST_POS`
   - `gaming_robot_arm/calibration/mill_default_calibration.py` anpassen:
     - `MILL_UARM_POSITIONS` (A1-C8 Brettkoordinaten)
     - `MILL_WHITE_RESERVE_POSITIONS` und `MILL_BLACK_RESERVE_POSITIONS` (3x3 Vorratskoordinaten fuer Setzzuege)
     - `MILL_PICK_Z`, `MILL_PLACE_Z` (Greif-/Ablagehoehen auf dem Brett)
     - `MILL_RESERVE_PICK_Z` (Pickhoehe fuer Reservepositionen)
   - Optional: `BOARD_LINE_PARAMS` in `gaming_robot_arm/vision/mill_board_detector.py` fuer die Brett-Detektion feinjustieren.

10. **Kalibrierung durchfuehren**

   ```bash
   python -m gaming_robot_arm.calibration.calibration
   ```

   - **Option 1**: Brett-Pixel erfassen (A1-C8) und `gaming_robot_arm/calibration/cam_to_robot_homography.json` (nur `board_pixels`) erzeugen/aktualisieren.
   - **Option 2**: Homography fitten (mindestens 4 Punktpaare). Ergebnis wird in `gaming_robot_arm/calibration/cam_to_robot_homography.json` unter `H` gespeichert.
   - **Option 3**: Vorhandene Kalibrierungsdateien auflisten.

11. **Installation verifizieren (empfohlen)**
   - Kamera-Test: `python -m gaming_robot_arm.vision.recording` (Live-Vorschau, Stopp mit `q`).
   - Runtime starten: `python main.py` (Standard: `--mode vision-loop`).
   - Spielbare Mill-CLI starten: `python main.py --mode play-mill --game-mode human-vs-ai`.
   - Roboter-Test: `python examples/move_uArm.py` oder `python examples/move_figures.py`.

### Vision-Trigger in spielbarer Muehle

- Bei `--human-input vision` ist `--vision-trigger auto` der Standard.
- `--vision-trigger auto` beobachtet das Brett kontinuierlich, wartet auf ein ruhiges/stabiles Bild und fuehrt den KI-Zug nur aus, wenn genau ein legaler menschlicher Zug erkannt wurde.
- Bei unklarer, mehrdeutiger oder instabiler Beobachtung faellt die Runtime konservativ auf den manuellen Vision-Scan per Enter zurueck.
- `--vision-trigger manual` behaelt das bisherige Verhalten bei: Zug auf dem realen Brett ausfuehren, dann per Enter einen Vision-Scan ausloesen.

Beispiel:

```bash
python main.py --mode play-mill --game-mode human-vs-ai --human-input vision --vision-trigger auto
```

## Mill-KI

Fuer Mill stehen interne Zug-Provider zur Verfuegung
(`gaming_robot_arm/games/mill/ai/builtin.py`):

```python
from gaming_robot_arm.games.mill import AlphaBetaMillAI, HeuristicMillAI, MillGameSession, MillRules

session = MillGameSession(rules=MillRules())
heuristic_ai = HeuristicMillAI(seed=42)
alpha_beta_ai = AlphaBetaMillAI(depth=3, seed=42)
move = session.choose_ai_move(alpha_beta_ai)
session.apply_move(move)
```

Vergleichstest (10 Spiele, wechselnde Farben):

```bash
gra-mill-benchmark --games 10 --depth 3
```

Der Vergleichstest ist generisch und kann beliebige Zug-Provider gegeneinander testen:

```bash
gra-mill-benchmark --ai-a heuristic --ai-b alphabeta --ai-b-arg depth=4 --games 10
gra-mill-benchmark --list-ai
```

### Bewertungslogik der Mill-KI

Die Gewichte sind Heuristiken, keine geloesten Spielwerte. Sie bilden strategische
Prioritaeten aus den Quellen ab und werden an die konkrete Bretttopologie dieses
Repos angepasst: [PlayMorris](https://www.playmorris.com/rules),
[boardgames.zone](https://boardgames.zone/morris/rules) und
[Kartik Kukreja](https://kartikkukreja.wordpress.com/2014/03/17/heuristicevaluation-function-for-nine-mens-morris/).

| Feature | Bedeutung | Placement | Movement | Flying | Warum |
| --- | --- | --- | --- | --- | --- |
| `piece_delta` | Materialvorteil | `900 -> 1200` | `1500` | `1800` | Material bleibt immer wichtig; im Flying entscheidet ein Capture oft direkt die Partie. |
| `closed_mill_delta` | Bereits geschlossene Muehlen | `70 -> 150` | `190` | `170` | Fruehe Muehlen sind nuetzlich, sollen im Opening aber nicht alle anderen Entwicklungsziele ueberdecken. |
| `open_mill_delta` | Zwei-in-einer-Reihe mit offenem Abschluss | `140 -> 130` | `90` | `130` | Offene Muehlen sind im Placement und Flying wertvoller als starre Frueh-Muehlen. |
| `double_mill_delta` | Wiederholt reformierbare Muehlen / Shared-Piece-Strukturen | `180 -> 150` | `180` | `220` | Doppelangriffe und Muehlenmotoren sind in allen spaeteren Phasen zentrale Gewinnmuster. |
| `future_mobility_delta` | Summe freier Nachbarn der eigenen Steine | `55 -> 35` | `0` | `0` | Im Setzspiel geht es zuerst darum, spaetere Beweglichkeit und Anschlussfelder aufzubauen. |
| `legal_mobility_delta` | Anzahl legaler Zuege | `0` | `18` | `8` | Im Movement zaehlen echte Zugoptionen; im Flying bleibt Mobilitaet relevant, aber weniger als Capture-/Schutzlogik. |
| `blocked_delta` | Gegenspieler einbauen / eigene Steine nicht einsperren | `20 -> 25` | `50` | `0` | Blockaden werden erst im Movement wirklich stark; im Flying verliert Adjazenz als Limit fast ihre Wirkung. |
| `protected_piece_delta` | Steine, die in einer geschlossenen Muehle geschuetzt sind | `0` | `0` | `80` | Im Flying steigen Wert und Ueberlebenswirkung geschuetzter Steine deutlich. |

Im Brettmodell dieses Repos sind die ringverbindenden Punkte die geraden Labels.
Besonders flexibel sind `B2`, `B4`, `B6` und `B8`, weil sie vier Nachbarn haben;
Ecken wie `A1` haben nur zwei. Das ist die konkrete Uebersetzung dessen, was die
Quellen als "midpoints" oder "intersections" beschreiben.

**Placement**

Ziel ist es, Optionen aufzubauen, Verbindungsfelder zu besetzen, Doppelangriffe
vorzubereiten und den Gegner in schlechte Entwicklung zu druecken. Deshalb liegen
`future_mobility_delta` und `open_mill_delta` vor fruehen `closed_mill_delta`-
Belohnungen. `double_mill_delta` ist bewusst hoch gewichtet, weil Shared-Piece-
Strukturen im Opening haeufig staerker sind als eine einzelne sofortige Muehle.
`blocked_delta` bleibt niedrig, weil im Setzspiel noch keine echte Bewegungsblockade
entsteht.

**Movement**

Im Mittelspiel steigen wiederholt reformierbare Muehlen, Zugzwang und echte
Mobilitaet. Daher sind `double_mill_delta`, `blocked_delta` und
`legal_mobility_delta` deutlich hoeher als im Placement. `future_mobility_delta`
faellt weg, weil jetzt nicht mehr potenzielle Nachbarschaften, sondern reale legale
Zuege zaehlen. `closed_mill_delta` bleibt wichtig, dominiert aber nicht blind ueber
den Aufbau eines stabilen Muehlenmotors.

**Flying**

Im Flying geht es vor allem um Capture-Rennen, Schutz der verbleibenden Steine und
das Vermeiden taktisch schlechter Muehlenschluesse. `blocked_delta` wird daher auf
`0` gesetzt, weil Adjazenz fuer fliegende Spieler kaum noch einschraenkt.
`protected_piece_delta` und `double_mill_delta` steigen, waehrend `piece_delta` am
staerksten bleibt: Wer hier einen Stein verliert, kippt die Partie oft sofort.

### Neuronales Mill-Training (PyTorch)

Schritt 1: Lehrerdaten per Selbstspiel mit `AlphaBetaMillAI` erzeugen:

```bash
gra-mill-generate-teacher --games 500 --teacher-depth 3 --output data/mill_teacher.jsonl
```

Schritt 2: Policy/Value-Modell (PyTorch, Mini-Batches + Checkpoints) trainieren:

```bash
gra-mill-train --data data/mill_teacher.jsonl --output models/mill_torch_v1.pt --epochs 12 --batch-size 128
```

Schritt 3: Neuronale KI gegen Basisgegner vergleichen:

```bash
gra-mill-benchmark --ai-a neural --ai-a-arg model_path=models/mill_torch_v1.pt --ai-b alphabeta --ai-b-arg depth=4 --games 20
```

Hinweis zu Regelkonsistenz: fuer Datengenerierung, Trainingsevaluation und Vergleichstest sollten dieselben Mill-Regelschalter genutzt werden (`--enable-flying`, `--enable-threefold-repetition`, `--enable-no-capture-draw`).

Regelschalter fuer ein spaeteres GUI-Menue:

- Backend-Einstellungen: `gaming_robot_arm/games/mill/core/settings.py` (`MillRuleSettings`)
- Projekt-Standardwerte: `gaming_robot_arm/games/mill/core/settings.py` (`MILL_*` Konstanten)

Beispiel (ohne GUI, interne KI):

```python
from gaming_robot_arm.games.mill.core.settings import (
    MILL_ENABLE_FLYING,
    MILL_ENABLE_NO_CAPTURE_DRAW,
    MILL_ENABLE_THREEFOLD_REPETITION,
    MILL_NO_CAPTURE_DRAW_PLIES,
)
from gaming_robot_arm.games.mill import (
    AlphaBetaMillAI,
    MillGameSession,
    MillRuleSettings,
    MillRules,
)

rules = MillRules(
    settings=MillRuleSettings(
        enable_flying=MILL_ENABLE_FLYING,
        enable_threefold_repetition=MILL_ENABLE_THREEFOLD_REPETITION,
        enable_no_capture_draw=MILL_ENABLE_NO_CAPTURE_DRAW,
        no_capture_draw_plies=MILL_NO_CAPTURE_DRAW_PLIES,
    )
)
session = MillGameSession(rules=rules)
ai = AlphaBetaMillAI(depth=3)
move = session.choose_ai_move(ai)
session.apply_move(move)
```

## Fehlerbehebung (Zuordnung = 0)

Wenn `gaming_robot_arm/vision/figure_detector.py` Kreise einzeichnet, aber **Roh**- und **stabile** Zuordnungen dauerhaft `0` bleiben,
passt sehr wahrscheinlich die Kalibrierung (`gaming_robot_arm/calibration/*board_pixels*`) nicht zur aktuellen Kamera-Aufloesung.

- Stelle sicher, dass Kalibrierung und Runtime mit derselben Aufloesung laufen (ggf. `FRAME_WIDTH/FRAME_HEIGHT` in `gaming_robot_arm/config.py` setzen).
- Kalibrierung neu ausfuehren: `python -m gaming_robot_arm.calibration.calibration` → Option 1.
- Debug-Protokolle aktivieren: `python -m gaming_robot_arm.vision.figure_detector --assignments --debug-assignments`
