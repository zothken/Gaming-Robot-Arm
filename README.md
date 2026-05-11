# Gaming Robot Arm

Dieses Projekt realisiert ein vollständig spielbares Mühlespiel mit einem uArm Swift Pro Roboter. Eine grafische Benutzeroberfläche (PySide6) erlaubt das Konfigurieren und Starten von Partien. Eine Kamera erkennt Spielsteine auf dem Brett per Computer Vision, der KI-Gegner berechnet Züge per Minimax-Algorithmus, und der Roboter setzt die Figuren physisch um. Optionale Sprachsteuerung erlaubt dem menschlichen Spieler, Züge per Mikrofon einzugeben.

## Gesamtarchitektur

Die Laufzeit teilt sich in Vision, Kalibrierung und Robotersteuerung. Die Kalibrierung liefert die Abbildung von Pixeln auf Roboterkoordinaten, der Mill-Game-Loop verarbeitet Frames und löst Bewegungen aus.

```
Kalibrierung:
  gaming_robot_arm.vision.mill_board_detector -> gaming_robot_arm.calibration.live_calibration (fit_homography_from_correspondences)
  gaming_robot_arm.games.mill.runtime.robot_bridge.img_to_robot wandelt Pixel -> Roboter-XY

Spielbare Mühle:
  games/mill/runtime/game_loop.py
    -> PlayerController (players.py)          -- human/ai/voice
    -> MillVisionBridge (vision_bridge.py)    -- Kamera -> Brettbelegung -> Zug
    -> VoiceBridge (voice_bridge.py)          -- Mikrofon -> Text -> Zug
    -> MillRobotBridge (robot_bridge.py)      -- Zug -> uArm-Ausführung
        -> gaming_robot_arm.control.UArmController (uArm Swift API)
```

## Module und Dateien

### Projektwurzel

| Modul/Datei | Funktion |
| --- | --- |
| `main.py` | Startpunkt/Launcher mit Modi für UI und spielbare Mill-Partie. |
| `gaming_robot_arm/` | Python-Paket (Vision, Control, Kalibrierung, Spiele, UI). |
| `pyproject.toml` | Paket-Metadaten, Python-Version (>=3.10) und alle Abhängigkeiten (Kern + optionale Extras `hardware`, `speech`, `ui`). |

### Paket `gaming_robot_arm/`

| Modul/Datei | Funktion |
| --- | --- |
| `gaming_robot_arm/__main__.py` | Einstiegspunkt für `python -m gaming_robot_arm`. |
| `gaming_robot_arm/app.py` | Argument-Parser und Modus-Dispatcher für UI und play-mill. |
| `gaming_robot_arm/config.py` | Zentrale Einstellungen für Kamera, uArm-Grenzen, Pfade und Board-Parameter. |
| `gaming_robot_arm/logger.py` | Logging-Setup für alle Module. |

### Paket `gaming_robot_arm/calibration/`

| Modul/Datei | Funktion |
| --- | --- |
| `gaming_robot_arm/calibration/live_calibration.py` | Interaktive Erfassung von Brett-Pixeln, Homography-Fit und Live-Bretterkennung aus Kameraframes. |
| `gaming_robot_arm/calibration/mill_default_calibration.py` | Feste uArm-XY-Koordinaten (mm) für alle 24 Brettlabels sowie Reservepositionen je Farbe. |

### Paket `gaming_robot_arm/vision/`

| Modul/Datei | Funktion |
| --- | --- |
| `gaming_robot_arm/vision/figure_detector.py` | Erkennung runder Figuren, Farbklassifikation, stabile Zuordnung zu Brettlabels, Live-Tuning. |
| `gaming_robot_arm/vision/mill_board_detector.py` | Konturbasierte Brettdetektion (drei Quadrate A/B/C) und 24 Feldpositionen mit EMA-Glättung. |
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
| `gaming_robot_arm/games/common/interfaces.py` | Gemeinsame Schnittstellen für Spiel-Logik (Move, Player, Rules). |

### Paket `gaming_robot_arm/games/mill/core/`

| Modul/Datei | Funktion |
| --- | --- |
| `gaming_robot_arm/games/mill/core/board.py` | Brett-Labels (A1-C8), Nachbarschaften (ADJACENT) und alle 16 Mühlenkombinationen (MILLS). |
| `gaming_robot_arm/games/mill/core/constants.py` | Gemeinsame Konstanten: Spieler-Tuple und Steinzahl pro Seite. |
| `gaming_robot_arm/games/mill/core/rules.py` | Vollständige Regelimplementierung (Setzphase, Bewegungsphase, Flying, Schlagzwang, Remisregeln). |
| `gaming_robot_arm/games/mill/core/settings.py` | Umschaltbare Regel-Einstellungen (Flying, Dreifachwiederholung, Zugzwang-Remis). |
| `gaming_robot_arm/games/mill/core/session.py` | Sitzungscontainer für Zustand und Zughistorie mit KI-Anbindung. |
| `gaming_robot_arm/games/mill/core/state.py` | Unveränderlicher Zustandscontainer (Board, to_move, placed, Zughistorie). |

### Paket `gaming_robot_arm/games/mill/ai/`

| Modul/Datei | Funktion |
| --- | --- |
| `gaming_robot_arm/games/mill/ai/builtin.py` | Interne KIs (Heuristik + Alpha-Beta mit Transpositionstabelle), keine externe Abhängigkeit. |

### Paket `gaming_robot_arm/games/mill/runtime/`

| Modul/Datei | Funktion |
| --- | --- |
| `gaming_robot_arm/games/mill/runtime/game_loop.py` | Spielbare Kommandozeilen-Partie mit Vision-, Roboter- und Sprachanbindung. |
| `gaming_robot_arm/games/mill/runtime/players.py` | Aufbau und Auflösung von Spieler-Controllern (human/ai/voice) und uArm-Spielerzuweisung. |
| `gaming_robot_arm/games/mill/runtime/robot_bridge.py` | Führt Mühle-Züge mit dem uArm aus (Pick-and-Place, Reserve- und Capture-Slots). |
| `gaming_robot_arm/games/mill/runtime/vision_bridge.py` | Liest stabile Brettbelegung aus Kameraframes und erkennt ausgeführte menschliche Züge. |
| `gaming_robot_arm/games/mill/runtime/voice_bridge.py` | Koordiniert STT-, Befehlsverarbeitungs- und Zug-Mapping-Threads für Sprachsteuerung. |
| `gaming_robot_arm/games/mill/runtime/stt.py` | Echtzeit-Spracherkennung via RealtimeSTT/Whisper, legt Transkripte in Befehlsqueue. |
| `gaming_robot_arm/games/mill/runtime/mill_commands.py` | Brett-Label-Vokabular und Whisper-Priming-Prompt für die Sprachsteuerung. |
| `gaming_robot_arm/games/mill/runtime/command_process.py` | Befehlserkennung via spaCy-Lemmatisierung und rapidfuzz-Fuzzy-Matching auf Brettlabels. |
| `gaming_robot_arm/games/mill/runtime/signals.py` | Sentinel-Werte (z.B. `UndoSignal`) aus den menschlichen Eingabekanälen. |

### Paket `gaming_robot_arm/games/mill/cli/`

| Modul/Datei | Funktion |
| --- | --- |
| `gaming_robot_arm/games/mill/cli/play.py` | CLI-Argumente und Einstiegspunkt für spielbare Mühle-Sitzungen (`gra-mill-play`). |
| `gaming_robot_arm/games/mill/cli/benchmark.py` | Generischer Kopf-an-Kopf-Benchmark für beliebige Zug-Provider-KIs (`gra-mill-benchmark`). |

### Paket `gaming_robot_arm/ui/launcher/`

| Modul/Datei | Funktion |
| --- | --- |
| `gaming_robot_arm/ui/launcher/window.py` | PySide6-Hauptfenster des Desktop-Launchers. |
| `gaming_robot_arm/ui/launcher/command_builder.py` | Baut CLI-Kommandozeilen aus Launcher-Formularwerten. |
| `gaming_robot_arm/ui/launcher/settings.py` | Persistenz und Datentransfer für Launcher-Einstellungen (JSON). |
| `gaming_robot_arm/ui/launcher/preview.py` | Lazy Loader für Kamera-Preview-Integrationen im Launcher. |
| `gaming_robot_arm/ui/launcher/process_runner.py` | QProcess-Starthelfer: startet und überwacht CLI-Unterprozesse aus dem Launcher. |

### Paket `examples/`

| Modul/Datei | Funktion |
| --- | --- |
| `examples/move_uArm.py` | Interaktives Bewegen (Koordinaten oder Brettlabel) inkl. optionaler Aufnahme. |
| `examples/move_figures.py` | Figur aufnehmen und zwischen zwei Brettpositionen umsetzen. |
| `examples/webcam_native.py` | Öffnet die Webcam in nativer Auflösung und zeigt den Stream (nur OpenCV). |
| `examples/cam.py` | Minimaler Kamera-Test mit fest gesetzter Auflösung (1080p MJPG). |
| `examples/test_voice_commands.py` | Interaktiver Test der Sprachsteuerung ohne laufendes Spiel. |
| `examples/cli_helpers.py` | Gemeinsame CLI-Helfer (Brett-Label-Abfrage, Aufnahme-Prompt) für Beispielskripte. |

### Daten und Ausgaben

| Pfad | Funktion |
| --- | --- |
| `gaming_robot_arm/calibration/*.json` | Kalibrierungsdaten (Board-Pixel und Homography-Matrix). |
| `gaming_robot_arm/vision/figure_detector_config.json` | Gespeicherte Figuren-Detektor-Parameter aus dem Live-Tuning. |
| `Aufnahmen/` | Standardziel für Videoaufnahmen. |

## Installation

1. **Voraussetzungen**
   - Python >= 3.10 inkl. `pip` (Prüfung: `python --version`).
   - Git (zum Klonen des Repos).
   - uArm Swift Pro via USB (Treiber/Seriell-Port muss vom Betriebssystem erkannt werden).
   - Für Vision-Steuerung: Kamera (USB/HDMI), die von OpenCV gelesen werden kann.

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

   Damit werden die Kernabhängigkeiten (`numpy`, `opencv-python`, `pyserial`) installiert und das Paket im Editable-Modus eingebunden, sodass Imports wie `gaming_robot_arm.config` überall funktionieren.

6. **Optional: uArm-SDK installieren**

   Erforderlich für alle Roboter-Features (Pick-and-Place, Kalibrierung, Robot-Bridge). Das SDK wird direkt von GitHub bezogen und benötigt Netzwerkzugriff:

   ```bash
   python -m pip install -e ".[hardware]"
   ```

7. **Optional: Sprachsteuerung einrichten**

   ```bash
   python -m pip install -e ".[speech]"
   ```

8. **Optional: Desktop-UI installieren**

   ```bash
   python -m pip install -e ".[ui]"
   ```

9. **Alle Extras auf einmal installieren**

   ```bash
   python -m pip install -e ".[hardware,speech,ui]"
   ```

   Dieser Befehl installiert auch die Kernabhängigkeiten aus Schritt 5 — wer Schritt 9 ausführt, kann Schritt 5-8 überspringen.

10. **Hardware verbinden und Ports prüfen**
    - Kamera anschließen und im OS prüfen (z.B. Kamera-App).
    - uArm per USB anschließen. Falls der Port nicht automatisch erkannt wird, `UARM_PORT` in `gaming_robot_arm/config.py` setzen.

11. **Projekt konfigurieren**
    - `gaming_robot_arm/config.py` anpassen:
      - `CAMERA_INDEX`, `FRAME_WIDTH`/`FRAME_HEIGHT` (optional; `None` = native Kameraauflösung), `FRAME_RATE` (optional; `None` = native Kamera-FPS)
      - `SAFE_Z`, `REST_POS`
    - `gaming_robot_arm/calibration/mill_default_calibration.py` anpassen:
      - `MILL_UARM_POSITIONS` (A1-C8 Brettkoordinaten)
      - `MILL_WHITE_RESERVE_POSITIONS` und `MILL_BLACK_RESERVE_POSITIONS` (3x3 Vorratskoordinaten für Setzzüge)
      - `MILL_PICK_Z`, `MILL_PLACE_Z` (Greif-/Ablagehöhen auf dem Brett)
      - `MILL_RESERVE_PICK_Z` (Pickhöhe für Reservepositionen)
    - Optional: `BOARD_LINE_PARAMS` in `gaming_robot_arm/vision/mill_board_detector.py` für die Brett-Detektion feinjustieren.

12. **Installation verifizieren (empfohlen)**
    - Kamera-Test: `python -m gaming_robot_arm.vision.recording` (Live-Vorschau, Stopp mit `q`).
    - Runtime starten: `python main.py` (Standard: `--mode ui`).
    - Spielbare Mill-CLI starten: `python main.py --mode play-mill --game-mode human-vs-ai`.
    - Roboter-Test: `python examples/move_uArm.py` oder `python examples/move_figures.py`.

### Vision-Trigger in spielbarer Mühle

- Bei `--human-input vision` ist `--vision-trigger auto` der Standard.
- `--vision-trigger auto` beobachtet das Brett kontinuierlich, wartet auf ein ruhiges/stabiles Bild und führt den KI-Zug nur aus, wenn genau ein legaler menschlicher Zug erkannt wurde.
- Bei unklarer, mehrdeutiger oder instabiler Beobachtung fällt die Runtime konservativ auf den manuellen Vision-Scan per Enter zurück.
- `--vision-trigger manual` behält das bisherige Verhalten bei: Zug auf dem realen Brett ausführen, dann per Enter einen Vision-Scan auslösen.

Beispiel:

```bash
python main.py --mode play-mill --game-mode human-vs-ai --human-input vision --vision-trigger auto
```

### Sprachsteuerung

Mit `--human-input voice` kann ein menschlicher Spieler Züge sprechen statt tippen:

```bash
python main.py --mode play-mill --game-mode human-vs-ai --human-input voice
```

Die Spracherkennung läuft über RealtimeSTT/Whisper. Züge werden als Brett-Label-Paare gesprochen (z.B. "A1 nach B2") oder als Zugnummer (z.B. "drei"). Beim Schlag muss das Capture-Feld angegeben werden (z.B. "A1 B2 C3").

Abhängigkeiten: `RealtimeSTT`, `pyaudio`, `spacy` (Modell `de_core_news_sm`), `rapidfuzz`.

## Mill-KI

Für Mill stehen interne Zug-Provider zur Verfügung
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

Die Gewichte sind Heuristiken, keine gelösten Spielwerte. Sie bilden strategische
Prioritäten aus den Quellen ab und werden an die konkrete Bretttopologie dieses
Repos angepasst: [PlayMorris](https://www.playmorris.com/rules),
[boardgames.zone](https://boardgames.zone/morris/rules) und
[Kartik Kukreja](https://kartikkukreja.wordpress.com/2014/03/17/heuristicevaluation-function-for-nine-mens-morris/).

| Feature | Bedeutung | Placement | Movement | Flying | Warum |
| --- | --- | --- | --- | --- | --- |
| `piece_delta` | Materialvorteil | `900 -> 1200` | `1500` | `1800` | Material bleibt immer wichtig; im Flying entscheidet ein Capture oft direkt die Partie. |
| `closed_mill_delta` | Bereits geschlossene Mühlen | `70 -> 150` | `190` | `170` | Frühe Mühlen sind nützlich, sollen im Opening aber nicht alle anderen Entwicklungsziele überdecken. |
| `open_mill_delta` | Zwei-in-einer-Reihe mit offenem Abschluss | `140 -> 130` | `90` | `130` | Offene Mühlen sind im Placement und Flying wertvoller als starre Früh-Mühlen. |
| `double_mill_delta` | Wiederholt reformierbare Mühlen / Shared-Piece-Strukturen | `180 -> 150` | `180` | `220` | Doppelangriffe und Mühlenmotoren sind in allen späteren Phasen zentrale Gewinnmuster. |
| `future_mobility_delta` | Summe freier Nachbarn der eigenen Steine | `55 -> 35` | `0` | `0` | Im Setzspiel geht es zuerst darum, spätere Beweglichkeit und Anschlussfelder aufzubauen. |
| `legal_mobility_delta` | Anzahl legaler Züge | `0` | `18` | `8` | Im Movement zählen echte Zugoptionen; im Flying bleibt Mobilität relevant, aber weniger als Capture-/Schutzlogik. |
| `blocked_delta` | Gegenspieler einbauen / eigene Steine nicht einsperren | `20 -> 25` | `50` | `0` | Blockaden werden erst im Movement wirklich stark; im Flying verliert Adjazenz als Limit fast ihre Wirkung. |
| `protected_piece_delta` | Steine, die in einer geschlossenen Mühle geschützt sind | `0` | `0` | `80` | Im Flying steigen Wert und Überlebenswirkung geschützter Steine deutlich. |

Im Brettmodell dieses Repos sind die ringverbindenden Punkte die geraden Labels.
Besonders flexibel sind `B2`, `B4`, `B6` und `B8`, weil sie vier Nachbarn haben;
Ecken wie `A1` haben nur zwei. Das ist die konkrete Übersetzung dessen, was die
Quellen als "midpoints" oder "intersections" beschreiben.

**Placement**

Ziel ist es, Optionen aufzubauen, Verbindungsfelder zu besetzen, Doppelangriffe
vorzubereiten und den Gegner in schlechte Entwicklung zu drücken. Deshalb liegen
`future_mobility_delta` und `open_mill_delta` vor frühen `closed_mill_delta`-
Belohnungen. `double_mill_delta` ist bewusst hoch gewichtet, weil Shared-Piece-
Strukturen im Opening häufig stärker sind als eine einzelne sofortige Mühle.
`blocked_delta` bleibt niedrig, weil im Setzspiel noch keine echte Bewegungsblockade
entsteht.

**Movement**

Im Mittelspiel steigen wiederholt reformierbare Mühlen, Zugzwang und echte
Mobilität. Daher sind `double_mill_delta`, `blocked_delta` und
`legal_mobility_delta` deutlich höher als im Placement. `future_mobility_delta`
fällt weg, weil jetzt nicht mehr potenzielle Nachbarschaften, sondern reale legale
Züge zählen. `closed_mill_delta` bleibt wichtig, dominiert aber nicht blind über
den Aufbau eines stabilen Mühlenmotors.

**Flying**

Im Flying geht es vor allem um Capture-Rennen, Schutz der verbleibenden Steine und
das Vermeiden taktisch schlechter Mühlenschlüsse. `blocked_delta` wird daher auf
`0` gesetzt, weil Adjazenz für fliegende Spieler kaum noch einschränkt.
`protected_piece_delta` und `double_mill_delta` steigen, während `piece_delta` am
stärksten bleibt: Wer hier einen Stein verliert, kippt die Partie oft sofort.

Hinweis zu Regelkonsistenz: Vergleichstest (`gra-mill-benchmark`) und Spiel (`gra-mill-play`) verwenden unterschiedlich benannte Schalter für dieselben Regeln — Benchmark: `--enable-flying`, `--enable-threefold-repetition`, `--enable-no-capture-draw`; Play: `--flying`/`--no-flying`, `--threefold-repetition`/`--no-threefold-repetition`, `--no-capture-draw`/`--no-no-capture-draw`. Beide nutzen `--no-capture-draw-plies` für die Schwelle.

Regelschalter für ein späteres GUI-Menü:

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
passt sehr wahrscheinlich die Kalibrierung (`gaming_robot_arm/calibration/*board_pixels*`) nicht zur aktuellen Kamera-Auflösung.

- Stelle sicher, dass Kalibrierung und Runtime mit derselben Auflösung laufen (ggf. `FRAME_WIDTH/FRAME_HEIGHT` in `gaming_robot_arm/config.py` setzen).
- Kalibrierung neu ausführen: `python -m gaming_robot_arm.calibration.live_calibration` → Option 1.
- Debug-Protokolle aktivieren: `python -m gaming_robot_arm.vision.figure_detector --assignments --debug-assignments`
