# Gaming Robot Arm

Dieses Projekt realisiert ein vollstaendig spielbares Muehlespiel mit einem uArm Swift Pro Roboter. Eine grafische Benutzeroberflaeche (PySide6) erlaubt das Konfigurieren und Starten von Partien. Eine Kamera erkennt Spielsteine auf dem Brett per Computer Vision, der KI-Gegner berechnet Zuege per Minimax-Algorithmus, und der Roboter setzt die Figuren physisch um. Optionale Sprachsteuerung erlaubt dem menschlichen Spieler, Zuege per Mikrofon einzugeben.

## Gesamtarchitektur

Die Laufzeit teilt sich in Vision, Kalibrierung und Robotersteuerung. Die Kalibrierung liefert die Abbildung von Pixeln auf Roboterkoordinaten, die Runtime verarbeitet Frames und kann Bewegungen ausloesen.

```
Kamera
  -> gaming_robot_arm.vision.recording (Kamera-Stream + Videoaufnahme)
  -> gaming_robot_arm.vision.figure_detector (Kreiserkennung + Farbklassifikation)
  -> gaming_robot_arm.runtime (Loop + Handler)
  -> gaming_robot_arm.control.UArmController (uArm Swift API)

Kalibrierung:
  gaming_robot_arm.vision.mill_board_detector -> gaming_robot_arm.calibration.live_calibration -> gaming_robot_arm/calibration/*.json
  gaming_robot_arm.utils.homography (img_to_robot) nutzt die gespeicherte Homography

Spielbare Muehle:
  games/mill/runtime/game_loop.py
    -> PlayerController (players.py)          -- human/ai/voice
    -> MillVisionBridge (vision_bridge.py)    -- Kamera -> Brettbelegung -> Zug
    -> VoiceBridge (voice_bridge.py)          -- Mikrofon -> Text -> Zug
    -> MillRobotBridge (robot_bridge.py)      -- Zug -> uArm-Ausfuehrung
```

## Module und Dateien

### Projektwurzel

| Modul/Datei | Funktion |
| --- | --- |
| `main.py` | Startpunkt/Launcher mit Modi fuer Vision-Loop und spielbare Mill-Partie. |
| `gaming_robot_arm/` | Python-Paket (Runtime, Vision, Control, Utils, Kalibrierung, Spiele). |
| `pyproject.toml` | Paket-Metadaten, Python-Version (>=3.10) und alle Abhaengigkeiten (Kern + optionale Extras `hardware`, `ml`, `speech`, `ui`). |

### Paket `gaming_robot_arm/`

| Modul/Datei | Funktion |
| --- | --- |
| `gaming_robot_arm/__main__.py` | Einstiegspunkt fuer `python -m gaming_robot_arm`. |
| `gaming_robot_arm/app.py` | Argument-Parser und Modus-Dispatcher fuer UI, Vision-Loop und play-mill. |
| `gaming_robot_arm/config.py` | Zentrale Einstellungen fuer Kamera, uArm-Grenzen, Pfade und Board-Parameter. |
| `gaming_robot_arm/runtime.py` | Orchestriert Kamera-Loop, Detection und optionale Robotik. |

### Paket `gaming_robot_arm/utils/`

| Modul/Datei | Funktion |
| --- | --- |
| `gaming_robot_arm/utils/homography.py` | Laden/Umrechnen Pixel -> Roboterkoordinaten via gespeicherter H-Matrix. |
| `gaming_robot_arm/utils/logger.py` | Logging-Setup fuer alle Module. |
| `gaming_robot_arm/utils/timing.py` | FPS-Tracker fuer Loop-Diagnose. |
| `gaming_robot_arm/utils/cli.py` | Gemeinsame CLI-Helfer (Brett-Label-Abfrage, Aufnahme-Prompt) fuer Beispielskripte. |

### Paket `gaming_robot_arm/calibration/`

| Modul/Datei | Funktion |
| --- | --- |
| `gaming_robot_arm/calibration/live_calibration.py` | Interaktive Erfassung von Brett-Pixeln, Homography-Fit und Live-Bretterkennung aus Kameraframes. |
| `gaming_robot_arm/calibration/mill_default_calibration.py` | Feste uArm-XY-Koordinaten (mm) fuer alle 24 Brettlabels sowie Reservepositionen je Farbe. |

### Paket `gaming_robot_arm/vision/`

| Modul/Datei | Funktion |
| --- | --- |
| `gaming_robot_arm/vision/figure_detector.py` | Erkennung runder Figuren, Farbklassifikation, stabile Zuordnung zu Brettlabels, Live-Tuning. |
| `gaming_robot_arm/vision/mill_board_detector.py` | Konturbasierte Brettdetektion (drei Quadrate A/B/C) und 24 Feldpositionen mit EMA-Glaettung. |
| `gaming_robot_arm/vision/detector_config.py` | Persistenz der Figure-Detector-Parameter (figure_detector_config.json). |
| `gaming_robot_arm/vision/recording.py` | Kamera-Handling, Frame-Lesen, MP4-Aufzeichnung, Live-Preview. |
| `gaming_robot_arm/vision/visualization.py` | Zeichnet Detections, Feld-Labels und Debug-Frames. |

### Paket `gaming_robot_arm/control/`

| Modul/Datei | Funktion |
| --- | --- |
| `gaming_robot_arm/control/uarm_controller.py` | Abstraktions-Wrapper der uArm Swift API mit sicheren Bewegungen, Greifer und Notstopp. |

### Paket `gaming_robot_arm/games/common/`

| Modul/Datei | Funktion |
| --- | --- |
| `gaming_robot_arm/games/common/interfaces.py` | Gemeinsame Schnittstellen fuer Spiel-Logik (Move, Player, Rules). |

### Paket `gaming_robot_arm/games/mill/core/`

| Modul/Datei | Funktion |
| --- | --- |
| `gaming_robot_arm/games/mill/core/board.py` | Brett-Labels (A1-C8), Nachbarschaften (ADJACENT) und alle 16 Muehlenkombinationen (MILLS). |
| `gaming_robot_arm/games/mill/core/constants.py` | Gemeinsame Konstanten: Spieler-Tuple und Steinzahl pro Seite. |
| `gaming_robot_arm/games/mill/core/rules.py` | Vollstaendige Regelimplementierung (Setzphase, Bewegungsphase, Flying, Schlagzwang, Remisregeln). |
| `gaming_robot_arm/games/mill/core/settings.py` | Umschaltbare Regel-Einstellungen (Flying, Dreifachwiederholung, Zugzwang-Remis). |
| `gaming_robot_arm/games/mill/core/session.py` | Sitzungscontainer fuer Zustand und Zughistorie mit KI-Anbindung. |
| `gaming_robot_arm/games/mill/core/state.py` | Unveraenderlicher Zustandscontainer (Board, to_move, placed, Zughistorie). |

### Paket `gaming_robot_arm/games/mill/ai/`

| Modul/Datei | Funktion |
| --- | --- |
| `gaming_robot_arm/games/mill/ai/builtin.py` | Interne KIs (Heuristik + Alpha-Beta mit Transpositionstabelle), keine externe Abhaengigkeit. |
| `gaming_robot_arm/games/mill/ai/neural.py` | Neuronale KI auf Basis eines PyTorch-Policy/Value-Modells mit Temperatur-Sampling. |

### Paket `gaming_robot_arm/games/mill/runtime/`

| Modul/Datei | Funktion |
| --- | --- |
| `gaming_robot_arm/games/mill/runtime/game_loop.py` | Spielbare Kommandozeilen-Partie mit Vision-, Roboter- und Sprachanbindung. |
| `gaming_robot_arm/games/mill/runtime/players.py` | Aufbau und Aufloesung von Spieler-Controllern (human/ai/voice) und uArm-Spielerzuweisung. |
| `gaming_robot_arm/games/mill/runtime/robot_bridge.py` | Fuehrt Muehle-Zuege mit dem uArm aus (Pick-and-Place, Reserve- und Capture-Slots). |
| `gaming_robot_arm/games/mill/runtime/vision_bridge.py` | Liest stabile Brettbelegung aus Kameraframes und erkennt ausgefuehrte menschliche Zuege. |
| `gaming_robot_arm/games/mill/runtime/voice_bridge.py` | Koordiniert STT-, Befehlsverarbeitungs- und Zug-Mapping-Threads fuer Sprachsteuerung. |
| `gaming_robot_arm/games/mill/runtime/stt.py` | Echtzeit-Spracherkennung via RealtimeSTT/Whisper, legt Transkripte in Befehlsqueue. |
| `gaming_robot_arm/games/mill/runtime/mill_commands.py` | Brett-Label-Vokabular und Whisper-Priming-Prompt fuer die Sprachsteuerung. |
| `gaming_robot_arm/games/mill/runtime/command_process.py` | Befehlserkennung via spaCy-Lemmatisierung und rapidfuzz-Fuzzy-Matching auf Brettlabels. |

### Paket `gaming_robot_arm/games/mill/ml/`

| Modul/Datei | Funktion |
| --- | --- |
| `gaming_robot_arm/games/mill/ml/model.py` | Zwei-Turm-Policy/Value-Netzwerk (PyTorch), Checkpoint-Speicherung und -Laden. |
| `gaming_robot_arm/games/mill/ml/features.py` | Kodierung von Zustand (83-dim) und Zuegen (77-dim) als Float-Vektoren fuer das Modell. |
| `gaming_robot_arm/games/mill/ml/training.py` | Mini-Batch-Trainingsschleife mit Policy-Cross-Entropy- und Value-MSE-Loss. |
| `gaming_robot_arm/games/mill/ml/dataset.py` | Laedt JSONL-Trainingsdaten, validiert Shapes und baut Mini-Batches fuer PyTorch. |
| `gaming_robot_arm/games/mill/ml/selfplay.py` | Hilfsfunktionen fuer Selbstspiel-Datengenerierung (Zug-Key, Ziel-Index-Suche). |
| `gaming_robot_arm/games/mill/ml/evolution.py` | Helfer fuer evolutionaere Optimierung: Gewichts-Klonen und lineare Interpolation. |
| `gaming_robot_arm/games/mill/ml/checkpoints.py` | Re-exportiert Checkpoint-Funktionen aus ml.model fuer abwaertskompatible Imports. |

### Paket `gaming_robot_arm/games/mill/cli/`

| Modul/Datei | Funktion |
| --- | --- |
| `gaming_robot_arm/games/mill/cli/play.py` | CLI-Argumente und Einstiegspunkt fuer spielbare Muehle-Sitzungen. |
| `gaming_robot_arm/games/mill/cli/benchmark.py` | Generischer Kopf-an-Kopf-Benchmark fuer beliebige Zug-Provider-KIs. |
| `gaming_robot_arm/games/mill/cli/train.py` | Trainiert ein PyTorch-Policy/Value-Modell aus JSONL-Teacher-Daten. |
| `gaming_robot_arm/games/mill/cli/generate_teacher_data.py` | Erzeugt Lehrerdaten per AlphaBeta-Selbstspiel (gra-mill-generate-teacher). |
| `gaming_robot_arm/games/mill/cli/generate_selfplay_data.py` | Erzeugt Trainingsdaten aus neuronalen Selbstspiel-Partien (gra-mill-generate-selfplay). |
| `gaming_robot_arm/games/mill/cli/selfplay_loop.py` | Orchestriert iterative Schleifen: Selbstspiel -> Training -> Bewertung -> Promotion. |
| `gaming_robot_arm/games/mill/cli/evolve_population.py` | Evolutionaeres Training der Neural-KI gegen AlphaBeta ohne Gradientenverfahren. |
| `gaming_robot_arm/games/mill/cli/train_watchdog.py` | Watchdog: startet Datenerzeugungs- und Trainingskommandos neu, falls sie abstuerzen. |
| `gaming_robot_arm/games/mill/cli/inspect_checkpoint.py` | Gibt Metadaten und Gewichte eines gespeicherten PyTorch-Checkpoints aus. |

### Paket `gaming_robot_arm/ui/launcher/`

| Modul/Datei | Funktion |
| --- | --- |
| `gaming_robot_arm/ui/launcher/window.py` | PySide6-Hauptfenster des Desktop-Launchers. |
| `gaming_robot_arm/ui/launcher/command_builder.py` | Baut CLI-Kommandozeilen aus Launcher-Formularwerten. |
| `gaming_robot_arm/ui/launcher/settings.py` | Persistenz und Datentransfer fuer Launcher-Einstellungen (JSON). |
| `gaming_robot_arm/ui/launcher/preview.py` | Lazy Loader fuer Kamera-Preview-Integrationen im Launcher. |
| `gaming_robot_arm/ui/launcher/process_runner.py` | QProcess-Starthelfer: startet und ueberwacht CLI-Unterprozesse aus dem Launcher. |

### Paket `examples/`

| Modul/Datei | Funktion |
| --- | --- |
| `examples/move_uArm.py` | Interaktives Bewegen (Koordinaten oder Brettlabel) inkl. optionaler Aufnahme. |
| `examples/move_figures.py` | Figur aufnehmen und zwischen zwei Brettpositionen umsetzen. |

### Daten und Ausgaben

| Pfad | Funktion |
| --- | --- |
| `gaming_robot_arm/calibration/*.json` | Kalibrierungsdaten (Board-Pixel und Homography-Matrix). |
| `gaming_robot_arm/vision/figure_detector_config.json` | Gespeicherte Figuren-Detektor-Parameter aus dem Live-Tuning. |
| `Aufnahmen/` | Standardziel fuer Videoaufnahmen der Runtime. |
| `data/` | Trainingsdaten (JSONL) und Benchmark-Ausgaben. |
| `models/` | Gespeicherte PyTorch-Checkpoints (z.B. `models/champion/mill_champion.pt`). |

## Installation

1. **Voraussetzungen**
   - Python >= 3.10 inkl. `pip` (Pruefung: `python --version`).
   - Git (zum Klonen des Repos).
   - uArm Swift Pro via USB (Treiber/Seriell-Port muss vom Betriebssystem erkannt werden).
   - Für Visuíon-Steuerung: Kamera (USB/HDMI), die von OpenCV gelesen werden kann.

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

5. **Paket installieren**

   ```bash
   python -m pip install -e .
   ```

   Damit werden die Kernabhaengigkeiten (`numpy`, `opencv-python`, `pyserial`) installiert und das Paket im Editable-Modus eingebunden, sodass Imports wie `gaming_robot_arm.config` ueberall funktionieren.

6. **Optional: uArm-SDK installieren**

   Erforderlich fuer alle Roboter-Features (Pick-and-Place, Kalibrierung, Robot-Bridge). Das SDK wird direkt von GitHub bezogen und benoetigt Netzwerkzugriff:

   ```bash
   python -m pip install -e ".[hardware]"
   ```

7. **Optional: ML-Abhaengigkeiten installieren (nur fuer Neural Mill)**

   ```bash
   # Standard (zieht ggf. CUDA-Pakete mit):
   python -m pip install -e ".[ml]"
   ```

   Linux nur CPU (kleiner, fuer CI empfohlen):

   ```bash
   python -m pip install --index-url https://download.pytorch.org/whl/cpu -e ".[ml]"
   ```

8. **Optional: Sprachsteuerung einrichten**

   ```bash
   python -m pip install -e ".[speech]"
   ```

9. **Optional: Desktop-UI installieren**

   ```bash
   python -m pip install -e ".[ui]"
   ```

10. **Alle Extras auf einmal installieren**

    ```bash
    python -m pip install -e ".[hardware,ml,speech,ui]"
    ```

    Dieser Befehl installiert auch die Kernabhaengigkeiten aus Schritt 5 — wer Schritt 10 ausfuehrt, kann Schritt 5 ueberspringen.

11. **Hardware verbinden und Ports pruefen**
    - Kamera anschliessen und im OS pruefen (z.B. Kamera-App).
    - uArm per USB anschliessen. Falls der Port nicht automatisch erkannt wird, `UARM_PORT` in `gaming_robot_arm/config.py` setzen.

12. **Projekt konfigurieren**
    - `gaming_robot_arm/config.py` anpassen:
      - `CAMERA_INDEX`, `FRAME_WIDTH`/`FRAME_HEIGHT` (optional; `None` = native Kameraaufloesung), `FRAME_RATE` (optional; `None` = native Kamera-FPS)
      - `SAFE_Z`, `REST_POS`
    - `gaming_robot_arm/calibration/mill_default_calibration.py` anpassen:
      - `MILL_UARM_POSITIONS` (A1-C8 Brettkoordinaten)
      - `MILL_WHITE_RESERVE_POSITIONS` und `MILL_BLACK_RESERVE_POSITIONS` (3x3 Vorratskoordinaten fuer Setzzuege)
      - `MILL_PICK_Z`, `MILL_PLACE_Z` (Greif-/Ablagehoehen auf dem Brett)
      - `MILL_RESERVE_PICK_Z` (Pickhoehe fuer Reservepositionen)
    - Optional: `BOARD_LINE_PARAMS` in `gaming_robot_arm/vision/mill_board_detector.py` fuer die Brett-Detektion feinjustieren.

13. **Kalibrierung durchfuehren**

    ```bash
    python -m gaming_robot_arm.calibration.live_calibration
    ```

    - **Option 1**: Brett-Pixel erfassen (A1-C8) und `gaming_robot_arm/calibration/cam_to_robot_homography.json` (nur `board_pixels`) erzeugen/aktualisieren.
    - **Option 2**: Homography fitten (mindestens 4 Punktpaare). Ergebnis wird in `gaming_robot_arm/calibration/cam_to_robot_homography.json` unter `H` gespeichert.
    - **Option 3**: Vorhandene Kalibrierungsdateien auflisten.

14. **Installation verifizieren (empfohlen)**
    - Kamera-Test: `python -m gaming_robot_arm.vision.recording` (Live-Vorschau, Stopp mit `q`).
    - Runtime starten: `python main.py` (Standard: `--mode ui`).
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

### Sprachsteuerung

Mit `--human-input voice` kann ein menschlicher Spieler Zuege sprechen statt tippen:

```bash
python main.py --mode play-mill --game-mode human-vs-ai --human-input voice
```

Die Spracherkennung laeuft ueber RealtimeSTT/Whisper. Zuege werden als Brett-Label-Paare gesprochen (z.B. "A1 nach B2") oder als Zugnummer (z.B. "drei"). Beim Schlag muss das Capture-Feld angegeben werden (z.B. "A1 B2 C3").

Abhaengigkeiten: `RealtimeSTT`, `pyaudio`, `spacy` (Modell `de_core_news_sm`), `rapidfuzz`.

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

Schritt 4 (optional): Iterative Selbstspiel-Schleife starten:

```bash
gra-mill-selfplay-loop --champion models/champion/mill_champion.pt
```

Schritt 5 (optional): Evolutionaeres Training ohne Gradientenverfahren:

```bash
gra-mill-evolve-population --generations 50 --population 20
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
- Kalibrierung neu ausfuehren: `python -m gaming_robot_arm.calibration.live_calibration` → Option 1.
- Debug-Protokolle aktivieren: `python -m gaming_robot_arm.vision.figure_detector --assignments --debug-assignments`
