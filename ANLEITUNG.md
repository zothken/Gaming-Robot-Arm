# Bedienungsanleitung — Gaming Robot Arm

Diese Anleitung führt dich Schritt für Schritt durch das System: vom allerersten Spiel bis zur vollständigen Nutzung mit Kamera, Spracheingabe und Roboterarm.

> **Technische Installation:** Siehe [README.md](README.md)

---

## Inhaltsverzeichnis

1. [Was ist das System?](#1-was-ist-das-system)
2. [Voraussetzungen & App starten](#2-voraussetzungen--app-starten)
3. [Das Startmenü](#3-das-startmenü)
4. [Erstes Spiel — Schritt für Schritt](#4-erstes-spiel--schritt-für-schritt)
5. [Das Spielbrett verstehen](#5-das-spielbrett-verstehen)
6. [Spielmodi & Spielerkonfiguration](#6-spielmodi--spielerkonfiguration)
7. [Eingabemethoden](#7-eingabemethoden)
8. [Spiel unterbrechen & fortsetzen](#8-spiel-unterbrechen--fortsetzen)
9. [Einstellungen im Detail](#9-einstellungen-im-detail)
10. [Spiel mit Roboterarm (uArm)](#10-spiel-mit-roboterarm-uarm)
11. [Spielvideo aufzeichnen](#11-spielvideo-aufzeichnen)
12. [Dev Mode](#12-dev-mode)
13. [Häufige Fragen & nicht-intuitive Verhaltensweisen](#13-häufige-fragen--nicht-intuitive-verhaltensweisen)
14. [Kurzreferenz Tastatureingaben](#14-kurzreferenz-tastatureingaben)

---

## 1. Was ist das System?

Das System ermöglicht es, das Brettspiel **Mühle (Nine Men's Morris)** in verschiedenen Varianten zu spielen:

- **Mensch gegen KI** oder **Mensch gegen Mensch**
- Züge eingeben per **Tastatur**, **Kamera** (Vision) oder **Sprache**
- Optional: ein **uArm Swift Pro Roboterarm** führt die Züge physisch auf einem echten Brett aus

**Was man mindestens braucht:** Einen PC mit installierter Software — mehr nicht. Kamera, Mikrofon und Roboter sind optional und erweitern die Erfahrung.

**Wichtig zu wissen:** Die grafische Oberfläche ist ein **Launcher**. Das eigentliche Spiel läuft im Hintergrund als Terminal-Prozess und die Ausgabe erscheint im **Log-Panel** rechts in der Oberfläche. Es öffnet sich kein separates Spielfenster.

---

## 2. Voraussetzungen & App starten

Stelle sicher, dass die Installation gemäß [README.md](README.md) abgeschlossen ist.

**App starten:**

```
python main.py
```

oder über die VS Code Run-Konfiguration.

**Fensteraufbau:**

```
┌────────────────────────────┬──────────────────────────────────┐
│  Linkes Panel              │  Rechtes Panel                   │
│  (Steuerung & Seiten)      │  (Status, Kamera, Log-Ausgabe)   │
│                            │                                  │
│  - Startmenü               │  - Statusanzeige                 │
│  - Spielkonfiguration      │  - Kamera-Vorschau (Dev Mode)    │
│  - Einstellungen           │  - Command Preview (Dev Mode)    │
│  - Dev Mode                │  - Log-Ausgabe des Spiels        │
└────────────────────────────┴──────────────────────────────────┘
```

Die Trennlinie zwischen den Panels lässt sich durch Ziehen verschieben.

---

## 3. Das Startmenü

Nach dem Start erscheint das Hauptmenü mit fünf Schaltflächen:

| Schaltfläche | Beschreibung |
|---|---|
| **Spiel Starten** | Öffnet die Spielkonfiguration für eine neue Partie |
| **Spiel fortsetzen** | Lädt den letzten gespeicherten Spielstand (nur aktiv wenn vorhanden) |
| **Einstellungen** | Dauerhaft gespeicherte Konfiguration für Hardware und Spielregeln |
| **Dev Mode** | Entwicklerwerkzeuge: Kamera-Vorschau, Detektor-Tuning, Fehlersuche |
| **Beenden** | App schließen |

**Hinweis:** „Spiel fortsetzen" ist ausgegraut, wenn noch kein Spielstand existiert. Das ist kein Fehler — nach dem ersten abgeschlossenen Zug in einer Partie wird automatisch gespeichert.

---

## 4. Erstes Spiel — Schritt für Schritt

Der schnellste Einstieg ohne Hardware: Mensch gegen KI, Eingabe per Tastatur.

**Schritt 1:** Im Startmenü auf **Spiel Starten** klicken.

**Schritt 2:** Spieler konfigurieren:
- *Spieler Weiß:* auf **Human** klicken
- *Spieler Schwarz:* auf **uArm** klicken (das bedeutet hier: KI übernimmt, nicht der Roboter — der Name ist historisch)

**Schritt 3:** Eingabemethode für den menschlichen Spieler: **Tastatur** auswählen.

**Schritt 4:** Auf **Jetzt starten** klicken.

Das Spiel startet. Die Ausgabe erscheint im Log-Panel rechts:

```
=== WEISS (Ply 1) ===
Aktuelle Brettbelegung:
[A1/O][A2/O][A3/O]...

Legale Züge:
  [01] setze A1
  [02] setze A2
  ...

Bitte Zugnummer eingeben:
```

**Schritt 5:** Eine Zahl eintippen (z.B. `1`) und Enter drücken. Das war der erste Zug.

Die KI antwortet automatisch, dann ist wieder Weiß dran. So geht es abwechselnd weiter.

**Spiel beenden:** Warten bis ein Spieler gewinnt, oder `q` + Enter eingeben um abzubrechen.

**Undo:** `z` + Enter — nimmt den letzten eigenen Zug zurück (falls erlaubt).

---

## 5. Das Spielbrett verstehen

### Brettstruktur

Das Mühle-Brett hat **24 Felder** in drei konzentrischen Ringen:

```
A1 ———————— A2 ———————— A3
|            |            |
|   B1 ————— B2 ————— B3  |
|   |         |         |  |
|   |  C1 — C2 — C3   |  |
|   |   |         |   |  |
|   |  C8 — C7 — C6   |  |
|   |         |         |  |
|   B8 ————— B7 ————— B6  |
|            |            |
A8 ———————— A7 ———————— A6
             |
            A5... (Mitte links)
```

- **Ring A** = äußerer Ring (A1–A8)
- **Ring B** = mittlerer Ring (B1–B8)
- **Ring C** = innerer Ring (C1–C8)
- Nummerierung 1–8 im Uhrzeigersinn, beginnend oben-links

Die tatsächliche Feldanordnung siehst du in der ASCII-Darstellung im Terminal. Felder werden als `[A1/O]` (leer), `[A1/W]` (Weiß) oder `[A1/B]` (Schwarz) angezeigt.

### Spielphasen

**Setzphase:** Jeder Spieler hat 9 Steine. Abwechselnd wird je ein Stein auf ein beliebiges freies Feld gesetzt. Insgesamt 18 Halbzüge.

**Bewegungsphase:** Alle Steine sind gesetzt. Jetzt wird je ein Stein auf ein benachbartes freies Feld verschoben (nur entlang der Linien).

**Flugphase** (wenn Regel aktiv): Wer noch genau 3 Steine hat, darf auf ein beliebiges freies Feld springen, nicht nur benachbarte.

### Mühlen & Schlagen

Eine **Mühle** entsteht, wenn drei eigene Steine in einer Reihe liegen (entlang der Linien des Brettes). Wer eine Mühle schließt, darf sofort einen beliebigen gegnerischen Stein schlagen (entfernen) — außer Steine, die selbst in einer Mühle liegen, wenn andere verfügbar sind.

**Gewonnen** hat, wer den Gegner auf weniger als 3 Steine reduziert oder ihn in eine Situation bringt, in der er keinen legalen Zug mehr hat.

---

## 6. Spielmodi & Spielerkonfiguration

### Spielmodi

| Modus | Weiß | Schwarz |
|---|---|---|
| Mensch vs. KI | Human | uArm |
| KI vs. Mensch | uArm | Human |
| Mensch vs. Mensch | Human | Human |
| KI vs. KI | uArm | uArm |

"uArm" in der Konfiguration bedeutet: Diese Seite wird von der KI gesteuert. Der physische Roboterarm greift nur dann ein, wenn er angeschlossen ist und die entsprechende Seite auf Robot-Kontrolle gesetzt ist.

### Eingabemethode

Die Eingabemethode (Tastatur, Kamera, Sprache) gilt für **menschliche Spieler**. Bei KI-Spielern hat diese Auswahl keine Wirkung.

---

## 7. Eingabemethoden

### 7a. Tastatur

Die einfachste Methode — kein Zusatzhardware nötig.

Im Terminal erscheint für jeden Zug eine nummerierte Liste legaler Züge. Die gewünschte Zahl eingeben und Enter drücken.

**Steuerung im laufenden Spiel:**

| Eingabe | Aktion |
|---|---|
| `1`–`N` + Enter | Zug mit dieser Nummer ausführen |
| `z` + Enter | Letzten Zug zurücknehmen |
| `q` + Enter | Partie abbrechen |

### 7b. Kamera / Vision

**Voraussetzungen:**
- Webcam angeschlossen
- Physisches Mühle-Brett mit weißen und schwarzen Spielsteinen aufgebaut
- Kameraindex in den Einstellungen korrekt gesetzt (Standard: 1)

**Ablauf Auto-Trigger (empfohlen):**

1. System akquiriert eine **Baseline** — es wartet, bis das Brett für mehrere Frames ruhig ist (kein Bewegungsunschärfe, keine Hand im Bild). Das dauert einige Sekunden. Timeout: 60 Sekunden (deaktivierbar in Einstellungen).
2. Zug physisch auf dem Brett ausführen (Stein setzen oder verschieben).
3. Hand vollständig aus dem Kamerabild nehmen.
4. System erkennt die Veränderung, bestätigt sie mehrfach und ordnet sie einem legalen Zug zu.
5. Zug wird automatisch ausgeführt.

Wenn Vision unsicher ist (mehrere Züge kommen in Frage), fordert das System zum manuellen Scan auf: Enter drücken.

**Ablauf Manuell-Trigger:**

1. Zug auf dem Brett ausführen.
2. Enter drücken → Kamera scannt das Brett.
3. Zug wird erkannt und ausgeführt.

**Tipp — Live-Vorschau:** In den Einstellungen (Tab *Wahrnehmung*) „Live-Vorschau mit Detector-Overlay" aktivieren. Ein separates Fenster zeigt das Kamerabild mit eingezeichneten Feldmarkierungen (A1–C8 als gelbe Punkte) und den erkannten Steinen. So lässt sich sofort prüfen, ob das System das Brett korrekt sieht.

**Typische Probleme und Abhilfe:**

- Brett nicht erkannt → Live-Vorschau aktivieren; Kameraindex prüfen; Brett besser beleuchten
- Zu viele Fehldetektionen → Vision-Trigger auf "Manuell" stellen
- Baseline-Timeout → Timeout deaktivieren oder Hand schneller aus dem Bild nehmen

### 7c. Sprache / Voice

**Voraussetzungen:**
- Mikrofon angeschlossen
- Pakete für Spracherkennung installiert (siehe README: `speech`-Extra)

**Wie man Züge spricht:**

Das System zeigt im Terminal die legalen Züge an und wartet auf eine Sprachäußerung. Folgende Formate werden erkannt:

| Situation | Was sprechen | Beispiel |
|---|---|---|
| Stein setzen (Setzphase) | Zielfeld | „A1" oder „B drei" |
| Stein bewegen | Quell- und Zielfeld | „A1 B2" |
| Stein bewegen + schlagen | Drei Felder | „A1 B2 C3" |
| Zugnummer | Deutsche Zahlwörter | „drei" für Zug Nr. 3 |
| Zug zurücknehmen | „zurück" | Bestätigung mit „ja" / „nein" folgt |

**Timeout:** Pro Zug 60 Sekunden. Danach erscheint eine Fehlermeldung. Timeout in den Einstellungen deaktivierbar.

**Rücknahme per Sprache:** Nach „zurück" fragt das System: *Bitte 'ja' oder 'nein' sagen zur Bestätigung.* Antwort innerhalb von 10 Sekunden geben.

**Tipp:** Feldbuchstaben klar aussprechen. „A" wie „Anton", Ziffern einzeln. Das System verwendet Fuzzy-Matching, versteht also auch leicht undeutliche Aussprache.

---

## 8. Spiel unterbrechen & fortsetzen

Das System speichert nach **jedem Zug** automatisch den Spielstand in `.mill_autosave.json` im Projektordner.

**Unterbrechen:**
- App einfach schließen — der aktuelle Stand ist bereits gespeichert.
- Alternativ: laufendes Spiel mit dem **Stop**-Button (erscheint während des Spiels im rechten Panel) beenden.

**Fortsetzen:**
1. App neu starten.
2. Im Startmenü: **Spiel fortsetzen** klicken (nur aktiv wenn Spielstand vorhanden).
3. Konfiguration wie zuvor wählen, dann **Jetzt starten**.

Auf der Spielkonfigurationsseite gibt es auch eine Checkbox **„Letzten Spielstand laden"** — damit lässt sich ein Spielstand laden, auch ohne den Startmenü-Button zu benutzen.

**Physisches Brett wiederherstellen:** Falls ein Roboter angeschlossen ist, kann die Checkbox **„Spielbrett physisch wiederherstellen (uArm)"** aktiviert werden. Der Roboter platziert dann alle Steine entsprechend dem gespeicherten Spielstand auf dem Brett, bevor das Spiel fortgesetzt wird.

**Wichtig:** Es gibt immer nur **einen** Spielstand. Ein neues Spiel überschreibt den alten automatisch.

---

## 9. Einstellungen im Detail

Erreichbar über **Einstellungen** im Startmenü. Alle Änderungen werden dauerhaft gespeichert.

### Tab: Wahrnehmung

Einstellungen für Kamera und Bilderkennung.

**Kamera:**

| Einstellung | Bedeutung |
|---|---|
| Kameraindex | Welche Kamera verwendet wird. 0 = erste Kamera, 1 = zweite, etc. Falsch gesetzt → kein Bild |
| Spiel aufzeichnen (Video) | Speichert das Kamerabild als MP4 in `Aufnahmen/` |

**Vision-Brücke:**

| Einstellung | Bedeutung |
|---|---|
| Scan-Versuche | Wie oft Vision bei Unsicherheit wiederholt (Standard: 6) |
| Vision-Trigger | Auto = erkennt Züge automatisch; Manuell = Enter-Taste löst Scan aus |
| Baseline-Timeout deaktivieren | System wartet unbegrenzt auf ruhiges Brett (nützlich bei schlechtem Licht) |
| Debug-Logging für Vision-Zuordnung | Ausführliche Vision-Logs im Terminal — hilfreich bei Erkennungsproblemen |
| Live-Vorschau mit Detector-Overlay | Öffnet separates Fenster mit Kamerabild und eingezeichneten Feldmarkierungen |

**Sprach-Brücke:**

| Einstellung | Bedeutung |
|---|---|
| Spracherkennungs-Timeout deaktivieren | System wartet unbegrenzt auf gültigen Sprachbefehl |

**Pre-Move-Warnung (nur mit Roboter relevant):**

| Einstellung | Bedeutung |
|---|---|
| Kamerawächter vor uArm-Zug | Roboter wartet, bis das Brett ruhig ist, bevor er sich bewegt |
| Vision-Gate Timeout (s) | Maximale Wartezeit für ruhiges Brett (Standard: 10 s) |
| Fallback-Pause (s) | Feste Pause vor Roboterzug, wenn kein Kamerawächter aktiv ist (Standard: 2 s; 0 = deaktiviert) |

### Tab: Mühle

Einstellungen für Spielregeln und Partiedauer.

| Einstellung | Bedeutung |
|---|---|
| Max. Halbzüge | Partie endet nach N Halbzügen unentschieden. 0 = unbegrenzt (Standard) |
| Flying-Regel aktivieren | Ein Spieler mit genau 3 Steinen darf auf beliebiges freies Feld springen |
| Remis bei Dreifachwiederholung | Unentschieden wenn dieselbe Stellung dreimal erreicht wird |
| Remis ohne Schlagserie | Unentschieden nach N Halbzügen ohne Schlag |
| Remisgrenze (Halbzüge) | Schwellenwert für obige Regel (Standard: 200) |

### Tab: KI

Einstellungen für die künstliche Intelligenz.

| Einstellung | Bedeutung |
|---|---|
| Backend | `heuristic` = schnell, gute Spielstärke; `alphabeta` = stärker, aber langsamer |
| AlphaBeta-Tiefe | Suchtiefe (Standard: 3). Höher = stärker, aber merklich langsamer ab Tiefe 5+ |
| Seed | Zufalls-Seed für reproduzierbare Spielzüge bei Gleichstand (Standard: 42) |
| Zufällige Tie-Breaks | Bei gleich bewerteten Zügen wählt die KI zufällig → weniger vorhersehbares Spielverhalten |

**Empfehlung:** Für ein anspruchsvolles Spiel `alphabeta` mit Tiefe 4 wählen. Für schnellere Partien `heuristic` oder `alphabeta` Tiefe 2.

### Tab: uArm

Einstellungen für den Roboterarm.

| Einstellung | Bedeutung |
|---|---|
| Serieller Port | Leer = automatische Erkennung (empfohlen); manuell eintragen wenn automatisch fehlschlägt |
| Robotergeschwindigkeit | Geschwindigkeit bei Pick-Place-Bewegungen (Standard: 500; höher = schneller) |
| Brett-Mapping | `default` = voreingestellte Koordinaten; `homography` = perspektivisch berechnetes Mapping |

---

## 10. Spiel mit Roboterarm (uArm)

Der uArm Swift Pro kann die Züge physisch auf dem Brett ausführen — Steine aufnehmen und platzieren.

**Voraussetzungen:**
- uArm per USB angeschlossen
- Physisches Mühle-Brett in der kalibrierten Position aufgebaut
- Kameraindex korrekt (für Pre-Move-Kamerawächter)

**Konfiguration:**

In der Spielkonfiguration die uArm-Kontrolle zuweisen:
- *Spieler Weiß:* uArm — Roboter spielt Weiß
- *Spieler Schwarz:* uArm — Roboter spielt Schwarz
- Beide auf uArm → vollautomatisches Spiel (zum Zuschauen)

**Brett-Mapping:**
- `default` — Funktioniert ohne weitere Kalibrierung, wenn das Brett an der vorgesehenen Position steht.
- `homography` — Für angepasste Brett-Positionen; benötigt Kalibrierungsdaten aus dem Kalibrierungs-Prozess.

**Wichtig — Sicherheit:**

> **Warnung:** Bevor der Roboter einen Zug ausführt, erscheint im Terminal die Meldung:
> `Warnung: uArm bewegt sich gleich - Brettbereich freihalten!`
> Hände und Gegenstände sofort aus dem Bewegungsbereich des Roboterarms entfernen.

Der Roboter bewegt sich nach einer kurzen Pause (Fallback-Pause) oder sobald der Kamerawächter das Brett als ruhig erkennt.

**Empfehlung:** Kamerawächter in den Einstellungen aktivieren (Tab *Wahrnehmung* → „Kamerawächter vor uArm-Zug"). Das verhindert, dass der Roboter in ein noch bewegtes Brett fährt.

---

## 11. Spielvideo aufzeichnen

Das System kann das Kamerabild während einer Partie als MP4 aufzeichnen.

**Aktivieren:** In den Einstellungen, Tab *Wahrnehmung* → Checkbox **„Spiel aufzeichnen (Video)"** aktivieren.

**Bedingung:** Nur verfügbar wenn die Eingabemethode „Kamera" aktiv ist (das System muss eine Kamera nutzen).

**Speicherort:** Die Videos werden automatisch im Ordner `Aufnahmen/` im Projektverzeichnis gespeichert. Der Dateiname enthält Datum und Uhrzeit.

---

## 12. Dev Mode

Der Dev Mode ist für Nutzer gedacht, die das System kalibrieren, die Bilderkennung tunen oder Fehler suchen möchten.

**Aufruf:** Startmenü → **Dev Mode**

### Kamera-Vorschau mit Overlay

Im rechten Panel erscheint ein Live-Kamerabild. Über ein Dropdown-Menü kann das angezeigte Overlay gewechselt werden:

| Overlay | Was es zeigt |
|---|---|
| Rohbild | Nur das Kamerabild, kein Overlay |
| Board Detector | Erkannte Brett-Konturen (drei konzentrische Quadrate A/B/C) |
| Figure Detector | Erkannte Steine mit Feldbeschriftungen (gelbe Punkte mit A1–C8) |

Mit dem Figure Detector Overlay lässt sich schnell prüfen, ob alle 24 Felder korrekt erkannt werden und ob Steine den richtigen Feldern zugeordnet sind.

### Command Preview

Zeigt den exakten Kommandozeilenaufruf, der beim Klick auf „Jetzt starten" ausgeführt wird. Nützlich um zu verstehen, welche Parameter an das Spiel übergeben werden.

### Tuning-Panel

Schieberegler für die Detektor-Parameter (Farbschwellen, Konturen, EMA-Glättung) mit Echtzeit-Feedback im Kamera-Vorschaufenster. Änderungen an den Detektoren können hier live getestet werden.

### Process Input & Log Output

Während ein Spiel läuft, können im Feld **Process Input** Texteingaben direkt an das Terminal des Spiels gesendet werden (z.B. Zugnummern eintippen). Das **Log Output** Feld zeigt die vollständige Terminal-Ausgabe in Echtzeit.

---

## 13. Häufige Fragen & nicht-intuitive Verhaltensweisen

**„Das Spiel öffnet kein eigenes Fenster"**
Das ist richtig so. Die Spielausgabe erscheint im Log-Panel rechts in der Launcher-Oberfläche. Es gibt kein separates Spielfenster.

**„Spiel fortsetzen ist ausgegraut"**
Es ist noch kein Spielstand vorhanden. Der Spielstand wird erst nach dem ersten Zug einer neuen Partie angelegt. Nach Neuinstallation ist der Button immer zunächst ausgegraut.

**„Die KI antwortet sehr langsam"**
Im Tab *KI* die AlphaBeta-Tiefe reduzieren (z.B. auf 3) oder das Backend auf `heuristic` umstellen. `heuristic` ist deutlich schneller bei akzeptabler Spielstärke.

**„Partie endet nach kurzer Zeit unentschieden"**
Die Einstellung „Max. Halbzüge" ist aktiv und wurde erreicht. Tab *Mühle* → Max. Halbzüge auf `0` setzen für eine unbegrenzte Partie.

**„Vision erkennt meine Züge nicht"**
1. Live-Vorschau aktivieren und prüfen ob alle Felder korrekt markiert sind.
2. Kameraindex überprüfen (0, 1, 2 ausprobieren).
3. Beleuchtung verbessern — gleichmäßiges, blendfreies Licht hilft der Erkennung.
4. Vision-Trigger auf „Manuell" stellen und per Enter-Taste scannen lassen.
5. Scan-Versuche erhöhen (Tab *Wahrnehmung*).

**„Baseline-Akquisition dauert sehr lange oder läuft ab"**
Das System wartet auf ein ruhiges Brett (keine Bewegung im Bild). Hand vollständig aus dem Kamerabild nehmen. Bei Problemen: Baseline-Timeout in den Einstellungen deaktivieren.

**„Spracheingabe reagiert nicht"**
Prüfen ob die Sprach-Pakete installiert sind (README: `speech`-Extra). Mikrofon-Zugriff im Betriebssystem erlaubt? Timeout ist 60 Sekunden — nach Ablauf erscheint eine Fehlermeldung.

**„Der Roboter bewegt sich trotzdem obwohl ich die Warnung gehört habe"**
Richtig — die Warnung erscheint kurz bevor der Roboter fährt. Es gibt eine kurze Pause (Fallback-Pause, Standard: 2 Sekunden), dann bewegt sich der Arm. Den Bereich sofort nach der Warnung freimachen.

**„Roboter verbindet sich nicht"**
Im Tab *uArm* den seriellen Port manuell eintragen. Auf Windows typischerweise `COM3`, `COM4` etc. (im Geräte-Manager nachsehen). Bei leer gelassenem Feld versucht das System automatisch zu verbinden.

**„Einstellungen gehen nach Neustart verloren"**
Das sollte nicht passieren — Einstellungen werden in `.gaming_robot_arm_launcher.json` gespeichert. Falls die Datei fehlt oder gelöscht wurde, werden Standardwerte verwendet.

**„Ich sehe keinen Stop-Button"**
Der Stop-Button erscheint nur während ein Spiel läuft. Er ist im rechten Panel sichtbar, sobald das Spiel gestartet wurde.

---

## 14. Kurzreferenz Tastatureingaben

Im laufenden Spiel (Eingabe im Log-Panel oder Process-Input-Feld):

| Eingabe | Aktion |
|---|---|
| `1`–`N` + Enter | Zug mit dieser Nummer ausführen |
| `z` + Enter | Letzten Zug zurücknehmen |
| `q` + Enter | Partie abbrechen |
| Enter (Vision-Modus, Manuell-Trigger) | Kamera-Scan auslösen |

Im Vision-Modus mit Auto-Trigger wird Enter nur als Fallback benötigt, wenn die automatische Erkennung unsicher ist.
