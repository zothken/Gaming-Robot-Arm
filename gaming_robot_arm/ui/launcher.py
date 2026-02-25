"""PySide6-basierter Desktop-Launcher fuer die bestehenden CLI-Modi."""

from __future__ import annotations

import json
import shlex
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from PySide6.QtCore import QProcess, QProcessEnvironment, QTimer, Qt
from PySide6.QtGui import QCloseEvent, QFont, QImage, QPixmap, QTextCursor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSplitter,
    QStackedWidget,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from gaming_robot_arm.config import (
    CAMERA_INDEX,
    MILL_ENABLE_FLYING,
    MILL_ENABLE_NO_CAPTURE_DRAW,
    MILL_ENABLE_THREEFOLD_REPETITION,
    MILL_NO_CAPTURE_DRAW_PLIES,
    UARM_PORT,
)

try:
    import cv2
except ModuleNotFoundError:  # pragma: no cover - optionaler Laufzeitpfad
    cv2 = None


@dataclass(slots=True)
class LauncherSettings:
    mode: str = "vision-loop"
    camera_index: int = CAMERA_INDEX

    mill_mode: str = "human-vs-ai"
    mill_human_color: str = "W"
    mill_human_input: str = "manual"
    mill_max_plies: int = 400

    mill_flying: bool = MILL_ENABLE_FLYING
    mill_threefold_repetition: bool = MILL_ENABLE_THREEFOLD_REPETITION
    mill_no_capture_draw: bool = MILL_ENABLE_NO_CAPTURE_DRAW
    mill_no_capture_draw_plies: int = MILL_NO_CAPTURE_DRAW_PLIES

    mill_ai: str = "alphabeta"
    mill_ai_depth: int = 3
    mill_ai_model: str = "models/champion/mill_champion.pt"
    mill_ai_temperature: float = 0.0
    mill_ai_device: str = "auto"
    mill_random_tiebreak: bool = True
    mill_seed: int = 42

    mill_vision_attempts: int = 6
    mill_debug_vision: bool = False

    mill_uarm_port: str = "" if UARM_PORT is None else str(UARM_PORT)
    mill_uarm_move_both_players: bool = False
    mill_robot_speed: int = 500
    mill_robot_board_map: str = "default"
    mill_white_reserve: str = ""
    mill_black_reserve: str = ""
    mill_capture_bin: str = ""

    @classmethod
    def from_payload(cls, payload: object) -> "LauncherSettings":
        base = asdict(cls())
        if isinstance(payload, dict):
            for key, value in payload.items():
                if key in base:
                    base[key] = value
        try:
            return cls(**base)
        except TypeError:
            return cls()


class LauncherWindow(QMainWindow):
    def __init__(self, entry_script: Path) -> None:
        super().__init__()
        self.entry_script = Path(entry_script).resolve()
        self.project_root = self.entry_script.parent
        self.settings_file = self.project_root / ".gaming_robot_arm_launcher.json"

        self._suppress_form_updates = False
        self._widgets: dict[str, QWidget] = {}
        self._mode_buttons: dict[str, QRadioButton] = {}
        self._process: QProcess | None = None

        self._status_label: QLabel | None = None
        self._summary_label: QLabel | None = None
        self._hint_label: QLabel | None = None
        self._settings_file_label: QLabel | None = None
        self._command_preview: QPlainTextEdit | None = None
        self._log_output: QPlainTextEdit | None = None
        self._stdin_input: QLineEdit | None = None
        self._start_button: QPushButton | None = None
        self._stop_button: QPushButton | None = None
        self._send_button: QPushButton | None = None
        self._left_pages: QStackedWidget | None = None
        self._settings_category_list: QListWidget | None = None
        self._settings_pages: QStackedWidget | None = None
        self._body_splitter: QSplitter | None = None
        self._left_panel: QFrame | None = None
        self._right_panel: QFrame | None = None
        self._camera_preview_label: QLabel | None = None
        self._camera_preview_overlay_combo: QComboBox | None = None
        self._camera_preview_timer: QTimer | None = None
        self._camera_capture = None
        self._camera_preview_index: int | None = None
        self._board_overlay_detector = None
        self._figure_overlay_detector = None
        self._board_pixels_loader = None
        self._camera_figure_board_coords_cache: dict[tuple[int, int], dict[str, tuple[int, int]] | None] = {}
        self._camera_figure_board_coords_warned: set[tuple[int, int]] = set()
        self._camera_overlay_error_key: str | None = None

        self._setup_process()
        self._build_ui()
        self._apply_theme()

        self._apply_settings_to_widgets(self._load_settings())
        self._show_home_screen()
        self._refresh_context()
        self._refresh_command_preview()
        self._sync_runtime_controls()
        self._set_status("Bereit")

    def _setup_process(self) -> None:
        self._process = QProcess(self)
        self._process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self._process.readyReadStandardOutput.connect(self._read_process_output)
        self._process.finished.connect(self._on_process_finished)
        self._process.errorOccurred.connect(self._on_process_error)
        self._process.stateChanged.connect(self._on_process_state_changed)

    def _build_ui(self) -> None:
        self.setWindowTitle("Gaming Robot Arm Leitstand")
        self.resize(1420, 900)
        self.setMinimumSize(1180, 760)

        central = QWidget(self)
        central.setObjectName("CentralRoot")
        self.setCentralWidget(central)

        outer = QVBoxLayout(central)
        outer.setContentsMargins(16, 16, 16, 16)
        outer.setSpacing(12)

        header = QFrame()
        header.setObjectName("HeaderPanel")
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(18, 14, 18, 14)
        header_layout.setSpacing(3)

        title = QLabel("Gaming Robot Arm Leitstand")
        title.setObjectName("HeaderTitle")
        subtitle = QLabel(
            "Startmenü und Einstellungsoberfläche für Vision-Loop und spielbare Mühle auf Basis des bestehenden Backends."
        )
        subtitle.setObjectName("HeaderSubtitle")
        subtitle.setWordWrap(True)

        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)
        outer.addWidget(header)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        outer.addWidget(splitter, 1)
        self._body_splitter = splitter

        left_panel = QFrame()
        left_panel.setObjectName("Panel")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(8)

        right_panel = QFrame()
        right_panel.setObjectName("Panel")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(10, 10, 10, 10)
        right_layout.setSpacing(10)
        self._left_panel = left_panel
        self._right_panel = right_panel

        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([520, 820])

        self._build_left_panel(left_layout)
        self._build_right_panel(right_layout)

    def _build_left_panel(self, parent_layout: QVBoxLayout) -> None:
        pages = QStackedWidget()
        pages.setObjectName("LeftPages")
        pages.addWidget(self._build_home_screen())
        pages.addWidget(self._build_launch_screen())
        pages.addWidget(self._build_settings_screen())
        pages.addWidget(self._build_dev_screen())
        self._left_pages = pages
        parent_layout.addWidget(pages, 1)

    def _build_scrollable_page(self, builder) -> QScrollArea:
        inner = QWidget()
        inner.setObjectName("TabContent")
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)
        builder(layout)
        layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setObjectName("TabScrollArea")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.viewport().setObjectName("TabScrollViewport")
        scroll.setWidget(inner)
        return scroll

    def _build_home_screen(self) -> QWidget:
        page = QWidget()
        page.setObjectName("StartScreen")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(0)
        layout.addStretch(1)

        menu_card = QFrame()
        menu_card.setObjectName("MainMenuCard")
        menu_card.setMaximumWidth(420)
        menu_layout = QVBoxLayout(menu_card)
        menu_layout.setContentsMargins(18, 18, 18, 18)
        menu_layout.setSpacing(12)

        play_btn = QPushButton("Spiel Starten")
        play_btn.setObjectName("MenuPrimaryButton")
        play_btn.clicked.connect(self._show_launch_screen)
        settings_btn = QPushButton("Einstellungen")
        settings_btn.setObjectName("MenuSecondaryButton")
        settings_btn.clicked.connect(self._show_settings_screen)
        dev_btn = QPushButton("Dev Mode")
        dev_btn.setObjectName("MenuDevButton")
        dev_btn.clicked.connect(self._show_dev_screen)
        exit_btn = QPushButton("Beenden")
        exit_btn.setObjectName("MenuDangerButton")
        exit_btn.clicked.connect(self.close)

        menu_layout.addWidget(play_btn)
        menu_layout.addWidget(settings_btn)
        menu_layout.addWidget(dev_btn)
        menu_layout.addWidget(exit_btn)

        row = QHBoxLayout()
        row.setSpacing(0)
        row.addStretch(1)
        row.addWidget(menu_card)
        row.addStretch(1)
        layout.addLayout(row)
        layout.addStretch(1)
        return page

    def _build_launch_screen(self) -> QWidget:
        return self._build_scrollable_page(self._build_launch_content)

    def _build_settings_screen(self) -> QWidget:
        page = QWidget()
        page.setObjectName("SettingsScreen")
        root = QHBoxLayout(page)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        sidebar = QFrame()
        sidebar.setObjectName("SettingsSidebar")
        sidebar.setMinimumWidth(190)
        sidebar.setMaximumWidth(240)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(10, 10, 10, 10)
        sidebar_layout.setSpacing(10)

        sidebar_title = QLabel("Einstellungen")
        sidebar_title.setObjectName("SectionLabel")
        sidebar_layout.addWidget(sidebar_title)

        category_list = QListWidget()
        category_list.setObjectName("SettingsCategoryList")
        category_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        category_list.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        category_list.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        category_list.setSpacing(6)
        category_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        category_list.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        category_list.addItems(["Allgemein", "Mühle", "KI", "Robotik + Vision"])
        category_list.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        row_heights = [category_list.sizeHintForRow(i) for i in range(category_list.count())]
        # Qt liefert hier vor dem ersten Rendern teils zu kleine Row-Hints und ignoriert
        # dabei Stylesheet-Padding/Margins. Daher erzwingen wir eine konservative Mindesthoehe.
        styled_row_min_height = 54
        row_height = max([h for h in row_heights if h > 0], default=styled_row_min_height)
        row_height = max(row_height, styled_row_min_height)
        total_height = (
            category_list.frameWidth() * 2
            + row_height * category_list.count()
            + category_list.spacing() * max(0, category_list.count() - 1)
            + 8
        )
        category_list.setMinimumHeight(total_height)
        category_list.setMaximumHeight(total_height)
        self._settings_category_list = category_list
        sidebar_layout.addWidget(category_list)

        back_box = self._group_box("Navigation")
        back_layout = QVBoxLayout(back_box)
        back_layout.setContentsMargins(10, 12, 10, 10)
        back_layout.setSpacing(6)
        back_btn = QPushButton("Zurück")
        back_btn.setObjectName("MenuSecondaryButton")
        back_btn.clicked.connect(self._show_home_screen)
        back_layout.addWidget(back_btn)
        sidebar_layout.addWidget(back_box)
        sidebar_layout.addStretch(1)

        pages = QStackedWidget()
        pages.setObjectName("SettingsPages")
        pages.addWidget(self._build_scrollable_page(self._build_general_tab))
        pages.addWidget(self._build_scrollable_page(self._build_mill_tab))
        pages.addWidget(self._build_scrollable_page(self._build_ai_tab))
        pages.addWidget(self._build_scrollable_page(self._build_bridge_tab))
        self._settings_pages = pages

        category_list.currentRowChanged.connect(pages.setCurrentIndex)
        pages.currentChanged.connect(category_list.setCurrentRow)
        category_list.setCurrentRow(0)

        root.addWidget(sidebar)
        root.addWidget(pages, 1)
        return page

    def _build_dev_screen(self) -> QWidget:
        return self._build_scrollable_page(self._build_dev_content)

    def _build_dev_content(self, layout: QVBoxLayout) -> None:
        nav_box = self._group_box("Navigation")
        nav_layout = QHBoxLayout(nav_box)
        nav_layout.setContentsMargins(12, 12, 12, 10)
        nav_layout.setSpacing(8)
        back_btn = QPushButton("Zurück zum Startbildschirm")
        back_btn.setObjectName("MenuSecondaryButton")
        back_btn.clicked.connect(self._show_home_screen)
        nav_layout.addWidget(back_btn)
        nav_layout.addStretch(1)
        layout.addWidget(nav_box)

        intro_box = self._group_box("Dev Mode")
        intro_layout = QVBoxLayout(intro_box)
        intro_layout.setContentsMargins(12, 14, 12, 12)
        intro_layout.setSpacing(8)
        intro = QLabel(
            "Hier liegen Entwicklerfunktionen, die nicht im normalen Nutzerfluss sichtbar sein sollen. "
            "Aktuell findest du rechts die Kameravorschau mit optionalen Detector-Overlays, "
            "Befehlsvorschau, Prozess-Eingabe und Live-Ausgabe."
        )
        intro.setWordWrap(True)
        intro.setObjectName("MutedText")
        intro_layout.addWidget(intro)
        layout.addWidget(intro_box)

        shortcuts = self._group_box("Schnellzugriffe")
        shortcuts_layout = QVBoxLayout(shortcuts)
        shortcuts_layout.setContentsMargins(12, 14, 12, 12)
        shortcuts_layout.setSpacing(8)
        go_launch_btn = QPushButton("Zu Spiel Starten")
        go_launch_btn.setObjectName("MenuSecondaryButton")
        go_launch_btn.clicked.connect(self._show_launch_screen)
        go_settings_btn = QPushButton("Zu Einstellungen")
        go_settings_btn.setObjectName("MenuSecondaryButton")
        go_settings_btn.clicked.connect(self._show_settings_screen)
        shortcuts_layout.addWidget(go_launch_btn)
        shortcuts_layout.addWidget(go_settings_btn)
        layout.addWidget(shortcuts)

    def _build_launch_content(self, layout: QVBoxLayout) -> None:
        nav_box = self._group_box("Navigation")
        nav_layout = QHBoxLayout(nav_box)
        nav_layout.setContentsMargins(12, 12, 12, 10)
        nav_layout.setSpacing(8)
        back_btn = QPushButton("Zurück zum Startbildschirm")
        back_btn.setObjectName("MenuSecondaryButton")
        back_btn.clicked.connect(self._show_home_screen)
        nav_layout.addWidget(back_btn)
        nav_layout.addStretch(1)
        layout.addWidget(nav_box)

        mode_box = self._group_box("Modusauswahl")
        mode_layout = QVBoxLayout(mode_box)
        mode_layout.setContentsMargins(12, 14, 12, 12)
        mode_layout.setSpacing(10)

        mode_group = QButtonGroup(self)
        mode_group.setExclusive(True)

        vision_card, vision_radio = self._mode_card(
            "vision-loop",
            "Vision-Laufzeit",
            "Startet den klassischen Vision-/Runtime-Loop (OpenCV + Robotersteuerung).",
        )
        mill_card, mill_radio = self._mode_card(
            "play-mill",
            "Spielbare Mühle",
            "Startet die vorhandene Mühle-CLI mit optionaler KI-, Vision- und uArm-Anbindung.",
        )

        mode_group.addButton(vision_radio)
        mode_group.addButton(mill_radio)
        mode_layout.addWidget(vision_card)
        mode_layout.addWidget(mill_card)
        layout.addWidget(mode_box)

        action_box = self._group_box("Schnellaktionen")
        action_layout = QVBoxLayout(action_box)
        action_layout.setContentsMargins(12, 14, 12, 12)
        action_layout.setSpacing(8)

        actions_row = QHBoxLayout()
        actions_row.setSpacing(8)
        self._start_button = QPushButton("Ausgewählten Modus starten")
        self._start_button.setObjectName("PrimaryButton")
        self._start_button.clicked.connect(self._start_process)
        self._stop_button = QPushButton("Stoppen")
        self._stop_button.setObjectName("DangerButton")
        self._stop_button.clicked.connect(self._stop_process)

        save_btn = QPushButton("Einstellungen speichern")
        save_btn.clicked.connect(self._save_settings)
        load_btn = QPushButton("Gespeicherte laden")
        load_btn.clicked.connect(self._reload_saved_settings)

        actions_row.addWidget(self._start_button)
        actions_row.addWidget(self._stop_button)
        actions_row.addWidget(save_btn)
        actions_row.addWidget(load_btn)
        actions_row.addStretch(1)
        action_layout.addLayout(actions_row)

        self._summary_label = QLabel("")
        self._summary_label.setWordWrap(True)
        self._summary_label.setObjectName("MutedText")
        self._hint_label = QLabel("")
        self._hint_label.setWordWrap(True)
        self._hint_label.setObjectName("MutedText")
        action_layout.addWidget(self._summary_label)
        action_layout.addWidget(self._hint_label)
        layout.addWidget(action_box)

        state_box = self._group_box("Launcher-Zustand")
        state_layout = QVBoxLayout(state_box)
        state_layout.setContentsMargins(12, 14, 12, 12)
        state_layout.setSpacing(8)
        state_layout.addWidget(self._section_label("Einstellungsdatei"))
        self._settings_file_label = QLabel(str(self.settings_file))
        self._settings_file_label.setWordWrap(True)
        self._settings_file_label.setObjectName("MutedText")
        state_layout.addWidget(self._settings_file_label)

        reset_btn = QPushButton("Standardwerte wiederherstellen")
        reset_btn.clicked.connect(self._reset_defaults)
        state_layout.addWidget(reset_btn, 0, Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(state_box)

        self._mode_buttons = {"vision-loop": vision_radio, "play-mill": mill_radio}
        for radio in self._mode_buttons.values():
            radio.toggled.connect(self._on_form_change)

    def _mode_card(self, value: str, title: str, description: str) -> tuple[QFrame, QRadioButton]:
        card = QFrame()
        card.setObjectName("ModeCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(10, 10, 10, 10)
        card_layout.setSpacing(4)

        radio = QRadioButton(title)
        radio.setProperty("mode_value", value)
        desc = QLabel(description)
        desc.setWordWrap(True)
        desc.setObjectName("MutedText")

        card_layout.addWidget(radio)
        card_layout.addWidget(desc)
        return card, radio

    def _build_general_tab(self, layout: QVBoxLayout) -> None:
        runtime_box = self._group_box("Allgemeine Laufzeit")
        runtime_form = QFormLayout(runtime_box)
        runtime_form.setContentsMargins(12, 16, 12, 12)
        runtime_form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        runtime_form.setFormAlignment(Qt.AlignmentFlag.AlignTop)
        runtime_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        self._add_line_edit(runtime_form, "camera_index", "Kameraindex")
        self._add_note(runtime_form, "Wird für Vision-Loop und die Vision-Brücke im Mühle-Modus verwendet.")
        layout.addWidget(runtime_box)

        note_box = self._group_box("Hinweise")
        note_layout = QVBoxLayout(note_box)
        note_layout.setContentsMargins(12, 14, 12, 12)
        note_layout.setSpacing(6)
        note = QLabel(
            "Der Launcher startet die bestehenden CLI-Modi als Unterprozess. Dadurch bleibt die aktuelle Backend-Logik unverändert und die GUI ist eine saubere Steueroberfläche darüber."
        )
        note.setWordWrap(True)
        note.setObjectName("MutedText")
        note_layout.addWidget(note)
        layout.addWidget(note_box)

    def _build_mill_tab(self, layout: QVBoxLayout) -> None:
        game_box = self._group_box("Spielablauf")
        game_form = self._new_form_layout(game_box)
        self._add_combo(game_form, "mill_mode", "Spielmodus", ["human-vs-human", "human-vs-ai", "ai-vs-ai"])
        self._add_combo(game_form, "mill_human_color", "Mensch-Farbe", ["W", "B"])
        self._add_combo(game_form, "mill_human_input", "Mensch-Eingabe", ["manual", "vision"])
        self._add_line_edit(game_form, "mill_max_plies", "Max. Halbzüge")
        self._add_note(game_form, "0 = unbegrenzt (keine Begrenzung der Halbzüge).")
        layout.addWidget(game_box)

        rules_box = self._group_box("Regeln")
        rules_layout = QVBoxLayout(rules_box)
        rules_layout.setContentsMargins(12, 14, 12, 12)
        rules_layout.setSpacing(6)
        self._add_check(rules_layout, "mill_flying", "Flying-Regel aktivieren")
        self._add_check(rules_layout, "mill_threefold_repetition", "Remis bei Dreifachwiederholung aktivieren")
        self._add_check(rules_layout, "mill_no_capture_draw", "Remis ohne Schlagserie aktivieren")

        draw_form = QFormLayout()
        draw_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        self._add_line_edit(draw_form, "mill_no_capture_draw_plies", "Remisgrenze (Halbzüge)")
        rules_layout.addLayout(draw_form)
        layout.addWidget(rules_box)

    def _build_ai_tab(self, layout: QVBoxLayout) -> None:
        ai_box = self._group_box("KI-Einstellungen")
        ai_form = self._new_form_layout(ai_box)
        self._add_combo(ai_form, "mill_ai", "Backend", ["heuristic", "alphabeta", "neural"])
        self._add_line_edit(ai_form, "mill_ai_depth", "AlphaBeta-Tiefe")
        self._add_line_edit(ai_form, "mill_ai_model", "Modellpfad")
        self._add_line_edit(ai_form, "mill_ai_temperature", "Temperatur")
        self._add_line_edit(ai_form, "mill_ai_device", "Gerät")
        self._add_line_edit(ai_form, "mill_seed", "Seed")
        layout.addWidget(ai_box)

        ai_flags = self._group_box("KI-Optionen")
        ai_flags_layout = QVBoxLayout(ai_flags)
        ai_flags_layout.setContentsMargins(12, 14, 12, 12)
        ai_flags_layout.setSpacing(6)
        self._add_check(ai_flags_layout, "mill_random_tiebreak", "Zufällige Tie-Breaks bei gleicher Bewertung")
        note = QLabel("Der Modus 'neural' benötigt optionale ML-Abhängigkeiten und einen gültigen Checkpoint.")
        note.setWordWrap(True)
        note.setObjectName("MutedText")
        ai_flags_layout.addWidget(note)
        layout.addWidget(ai_flags)

    def _build_bridge_tab(self, layout: QVBoxLayout) -> None:
        vision_box = self._group_box("Vision-Brücke")
        vision_form = self._new_form_layout(vision_box)
        self._add_line_edit(vision_form, "mill_vision_attempts", "Scan-Versuche")
        self._add_check(vision_box.layout(), "mill_debug_vision", "Debug-Logging für Vision-Zuordnung")
        layout.addWidget(vision_box)

        robot_box = self._group_box("uArm-Brücke")
        robot_form = self._new_form_layout(robot_box)
        self._add_line_edit(robot_form, "mill_uarm_port", "Serieller Port")
        self._add_note(robot_form, "Leer lassen = Backend-Default / Auto-Erkennung.")
        self._add_line_edit(robot_form, "mill_robot_speed", "Robotergeschwindigkeit")
        self._add_combo(robot_form, "mill_robot_board_map", "Brett-Mapping", ["default", "homography"])
        self._add_line_edit(robot_form, "mill_white_reserve", "Weißer Vorrat (X,Y)")
        self._add_line_edit(robot_form, "mill_black_reserve", "Schwarzer Vorrat (X,Y)")
        self._add_line_edit(robot_form, "mill_capture_bin", "Ablage für Schläge (X,Y)")

        self._add_check(robot_form, "mill_uarm_move_both_players", "uArm führt Züge beider Seiten aus")
        layout.addWidget(robot_box)

    def _build_right_panel(self, layout: QVBoxLayout) -> None:
        status_row = QHBoxLayout()
        status_row.setSpacing(8)
        status_row.addWidget(self._section_label("Status"))
        self._status_label = QLabel("Bereit")
        self._status_label.setObjectName("StatusText")
        status_row.addWidget(self._status_label)
        status_row.addStretch(1)
        layout.addLayout(status_row)

        camera_box = self._group_box("Kameravorschau")
        camera_layout = QVBoxLayout(camera_box)
        camera_layout.setContentsMargins(12, 14, 12, 12)
        camera_layout.setSpacing(8)

        overlay_row = QHBoxLayout()
        overlay_row.setSpacing(8)
        overlay_label = QLabel("Overlay")
        overlay_label.setObjectName("MutedText")
        overlay_combo = QComboBox()
        overlay_combo.setEditable(False)
        overlay_combo.addItems(
            [
                "Rohbild",
                "Board Detector Overlay",
                "Figure Detector Overlay",
            ]
        )
        overlay_combo.currentTextChanged.connect(self._on_camera_preview_overlay_change)
        self._camera_preview_overlay_combo = overlay_combo
        overlay_row.addWidget(overlay_label)
        overlay_row.addWidget(overlay_combo, 1)
        camera_layout.addLayout(overlay_row)

        camera_placeholder = QLabel("Kameravorschau (Dev Mode)\n\nWird beim Öffnen des Dev Mode gestartet.")
        camera_placeholder.setObjectName("CameraPreviewPlaceholder")
        camera_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        camera_placeholder.setMinimumHeight(180)
        camera_layout.addWidget(camera_placeholder)
        self._camera_preview_label = camera_placeholder
        layout.addWidget(camera_box)

        command_box = self._group_box("Befehlsvorschau (Spielstart)")
        command_layout = QVBoxLayout(command_box)
        command_layout.setContentsMargins(12, 14, 12, 12)
        self._command_preview = QPlainTextEdit()
        self._command_preview.setReadOnly(True)
        self._command_preview.setObjectName("CommandPreview")
        self._command_preview.setFixedHeight(96)
        self._command_preview.setFont(self._mono_font())
        command_layout.addWidget(self._command_preview)
        layout.addWidget(command_box)

        input_box = self._group_box("Prozess-Eingabe")
        input_layout = QVBoxLayout(input_box)
        input_layout.setContentsMargins(12, 14, 12, 12)
        input_layout.setSpacing(8)

        input_note = QLabel(
            "Für manuelle Mühle-Eingaben (Zugnummer, Enter oder q). Die Eingabe wird an den laufenden Unterprozess gesendet."
        )
        input_note.setWordWrap(True)
        input_note.setObjectName("MutedText")
        input_layout.addWidget(input_note)

        input_row = QHBoxLayout()
        input_row.setSpacing(8)
        self._stdin_input = QLineEdit()
        self._stdin_input.setPlaceholderText("Eingabe an den Prozess senden …")
        self._stdin_input.returnPressed.connect(self._send_process_input)
        self._send_button = QPushButton("Senden")
        self._send_button.clicked.connect(self._send_process_input)
        send_empty_btn = QPushButton("Leere Zeile senden")
        send_empty_btn.clicked.connect(self._send_empty_line)
        input_row.addWidget(self._stdin_input, 1)
        input_row.addWidget(self._send_button)
        input_row.addWidget(send_empty_btn)
        input_layout.addLayout(input_row)
        layout.addWidget(input_box)

        log_box = self._group_box("Live-Ausgabe")
        log_layout = QVBoxLayout(log_box)
        log_layout.setContentsMargins(12, 14, 12, 12)
        log_layout.setSpacing(8)

        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)
        clear_btn = QPushButton("Log leeren")
        clear_btn.clicked.connect(self._clear_log)
        save_btn = QPushButton("Einstellungen speichern")
        save_btn.clicked.connect(self._save_settings)
        quick_stop = QPushButton("Stoppen")
        quick_stop.setObjectName("DangerButton")
        quick_stop.clicked.connect(self._stop_process)
        quick_start = QPushButton("Start")
        quick_start.setObjectName("PrimaryButton")
        quick_start.clicked.connect(self._start_process)

        toolbar.addWidget(clear_btn)
        toolbar.addWidget(save_btn)
        toolbar.addStretch(1)
        toolbar.addWidget(quick_stop)
        toolbar.addWidget(quick_start)
        log_layout.addLayout(toolbar)

        self._log_output = QPlainTextEdit()
        self._log_output.setReadOnly(True)
        self._log_output.setObjectName("LogOutput")
        self._log_output.setFont(self._mono_font())
        self._log_output.document().setMaximumBlockCount(6000)
        log_layout.addWidget(self._log_output, 1)
        layout.addWidget(log_box, 1)

    def _set_dev_panel_visible(self, visible: bool) -> None:
        if self._right_panel is None:
            return
        self._right_panel.setVisible(visible)
        if self._body_splitter is not None:
            if visible:
                self._body_splitter.setSizes([520, 820])
            else:
                self._body_splitter.setSizes([1, 0])
        if visible:
            self._start_camera_preview()
        else:
            self._stop_camera_preview()

    def _show_home_screen(self) -> None:
        if self._left_pages is not None:
            self._left_pages.setCurrentIndex(0)
        self._set_dev_panel_visible(False)

    def _show_launch_screen(self) -> None:
        if self._left_pages is not None:
            self._left_pages.setCurrentIndex(1)
        self._set_dev_panel_visible(False)
        self._refresh_context()
        self._refresh_command_preview()

    def _show_settings_screen(self) -> None:
        if self._left_pages is not None:
            self._left_pages.setCurrentIndex(2)
        self._set_dev_panel_visible(False)
        if self._settings_category_list is not None and self._settings_pages is not None:
            row = max(0, self._settings_category_list.currentRow())
            self._settings_category_list.setCurrentRow(row)
            self._settings_pages.setCurrentIndex(row)

    def _show_dev_screen(self) -> None:
        if self._left_pages is not None:
            self._left_pages.setCurrentIndex(3)
        self._set_dev_panel_visible(True)
        self._refresh_command_preview()

    def _camera_preview_enabled(self) -> bool:
        return self._right_panel is not None and self._right_panel.isVisible()

    def _camera_preview_interval_timer(self) -> QTimer:
        timer = self._camera_preview_timer
        if timer is None:
            timer = QTimer(self)
            timer.setInterval(33)
            timer.timeout.connect(self._update_camera_preview_frame)
            self._camera_preview_timer = timer
        return timer

    def _camera_preview_overlay_mode(self) -> str:
        combo = self._camera_preview_overlay_combo
        if not isinstance(combo, QComboBox):
            return "raw"
        text = combo.currentText().strip()
        if text.startswith("Board"):
            return "board"
        if text.startswith("Figure"):
            return "figure"
        return "raw"

    def _on_camera_preview_overlay_change(self, _text: str) -> None:
        self._camera_overlay_error_key = None
        if self._camera_preview_enabled():
            self._update_camera_preview_frame()

    def _set_camera_preview_message(self, text: str) -> None:
        label = self._camera_preview_label
        if label is None:
            return
        label.clear()
        label.setText(text)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def _set_camera_overlay_error_once(self, mode: str, exc: Exception) -> None:
        key = f"{mode}:{type(exc).__name__}:{exc}"
        if self._camera_overlay_error_key == key:
            return
        self._camera_overlay_error_key = key
        self._append_log(f"[launcher] Kamera-Overlay ({mode}) Fehler: {exc}\n")

    def _load_board_overlay_detector(self):
        detector = self._board_overlay_detector
        if callable(detector):
            return detector
        from gaming_robot_arm.games.mill.mill_board_detector import detect_board_positions

        self._board_overlay_detector = detect_board_positions
        return detect_board_positions

    def _load_figure_overlay_detector(self):
        detector = self._figure_overlay_detector
        if callable(detector):
            return detector
        from gaming_robot_arm.vision.figure_detector import detect_figures

        self._figure_overlay_detector = detect_figures
        return detect_figures

    def _load_board_pixels_loader(self):
        loader = self._board_pixels_loader
        if callable(loader):
            return loader
        from gaming_robot_arm.calibration.calibration import load_board_pixels

        self._board_pixels_loader = load_board_pixels
        return load_board_pixels

    def _camera_preview_board_coords(self, *, frame_width: int, frame_height: int) -> dict[str, tuple[int, int]] | None:
        key = (int(frame_width), int(frame_height))
        if key in self._camera_figure_board_coords_cache:
            return self._camera_figure_board_coords_cache[key]

        try:
            load_board_pixels = self._load_board_pixels_loader()
            board_coords = load_board_pixels(frame_size=key)
            normalized = {str(lbl): (int(x), int(y)) for lbl, (x, y) in board_coords.items()}
            self._camera_figure_board_coords_cache[key] = normalized
            return normalized
        except FileNotFoundError:
            self._camera_figure_board_coords_cache[key] = None
            if key not in self._camera_figure_board_coords_warned:
                self._camera_figure_board_coords_warned.add(key)
                self._append_log(
                    "[launcher] Figure-Overlay ohne Feldlabels: keine Brett-Kalibrierung gefunden "
                    "(board_pixels.json / cam_to_robot_homography.json).\n"
                )
            return None
        except Exception as exc:
            self._camera_figure_board_coords_cache[key] = None
            if key not in self._camera_figure_board_coords_warned:
                self._camera_figure_board_coords_warned.add(key)
                self._append_log(f"[launcher] Figure-Overlay: Brett-Kalibrierung konnte nicht geladen werden: {exc}\n")
            return None

    def _apply_camera_preview_overlay(self, frame, mode: str):
        if mode == "raw":
            return frame

        if cv2 is None:
            return frame

        if mode == "board":
            detect_board_positions = self._load_board_overlay_detector()
            board_result = detect_board_positions(frame, debug=False, return_bw=False)
            annotated = board_result[1]
            return annotated

        if mode == "figure":
            detect_figures = self._load_figure_overlay_detector()
            annotated_input = frame.copy()
            frame_height = int(getattr(annotated_input, "shape", [0, 0])[0])
            frame_width = int(getattr(annotated_input, "shape", [0, 0])[1])
            board_coords = None
            labels_order = None
            if frame_width > 0 and frame_height > 0:
                board_coords = self._camera_preview_board_coords(frame_width=frame_width, frame_height=frame_height)
                if board_coords:
                    labels_order = sorted(board_coords.keys())
            result = detect_figures(
                annotated_input,
                board_coords=board_coords,
                labels_order=labels_order,
                draw_assignments=bool(board_coords),
            )
            if isinstance(result, tuple) and len(result) > 0:
                annotated = result[0]
            else:
                annotated = annotated_input

            if board_coords is None:
                try:
                    cv2.putText(  # type: ignore[attr-defined]
                        annotated,
                        "Figure Overlay: keine Brett-Kalibrierung (ohne Feldlabels)",
                        (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX,  # type: ignore[attr-defined]
                        0.55,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,  # type: ignore[attr-defined]
                    )
                except Exception:
                    pass
            return annotated

        return frame

    def _camera_preview_current_index(self) -> int | None:
        widget = self._widgets.get("camera_index")
        if not isinstance(widget, QLineEdit):
            return None
        raw = widget.text().strip()
        if raw == "":
            self._set_camera_preview_message("Kameravorschau (Dev Mode)\n\nKein Kameraindex gesetzt.")
            return None
        try:
            value = int(raw)
        except ValueError:
            self._set_camera_preview_message("Kameravorschau (Dev Mode)\n\nUngültiger Kameraindex.")
            return None
        if value < 0:
            self._set_camera_preview_message("Kameravorschau (Dev Mode)\n\nKameraindex muss >= 0 sein.")
            return None
        return value

    def _start_camera_preview(self) -> None:
        if not self._camera_preview_enabled():
            return
        if self._camera_preview_label is None:
            return
        if cv2 is None:
            self._set_camera_preview_message("Kameravorschau (Dev Mode)\n\nOpenCV (cv2) ist nicht installiert.")
            return

        index = self._camera_preview_current_index()
        if index is None:
            self._stop_camera_preview(keep_label_text=True)
            return

        timer = self._camera_preview_interval_timer()
        if self._camera_capture is not None and self._camera_preview_index == index:
            if not timer.isActive():
                timer.start()
            return

        self._stop_camera_preview(keep_label_text=True)
        self._set_camera_preview_message(f"Kameravorschau (Dev Mode)\n\nVerbinde mit Kamera {index} ...")

        capture = cv2.VideoCapture(index)  # type: ignore[operator]
        try:
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # type: ignore[attr-defined]
        except Exception:
            pass

        if capture is None or not capture.isOpened():
            try:
                if capture is not None:
                    capture.release()
            except Exception:
                pass
            self._camera_capture = None
            self._camera_preview_index = None
            self._set_camera_preview_message(
                f"Kameravorschau (Dev Mode)\n\nKamera {index} konnte nicht geöffnet werden."
            )
            return

        self._camera_capture = capture
        self._camera_preview_index = index
        self._camera_figure_board_coords_cache.clear()
        self._camera_figure_board_coords_warned.clear()
        self._camera_overlay_error_key = None
        timer.start()
        self._update_camera_preview_frame()

    def _stop_camera_preview(self, *, keep_label_text: bool = False) -> None:
        if self._camera_preview_timer is not None and self._camera_preview_timer.isActive():
            self._camera_preview_timer.stop()

        capture = self._camera_capture
        self._camera_capture = None
        self._camera_preview_index = None
        if capture is not None:
            try:
                capture.release()
            except Exception:
                pass
        self._camera_overlay_error_key = None

        if not keep_label_text:
            self._set_camera_preview_message("Kameravorschau (Dev Mode)\n\nDev Mode öffnen, um die Vorschau zu starten.")

    def _update_camera_preview_frame(self) -> None:
        if not self._camera_preview_enabled():
            return
        label = self._camera_preview_label
        capture = self._camera_capture
        if label is None or capture is None or cv2 is None:
            return

        try:
            ok, frame = capture.read()
        except Exception as exc:
            self._set_camera_preview_message(f"Kameravorschau (Dev Mode)\n\nLesefehler: {exc}")
            return

        if not ok or frame is None:
            self._set_camera_preview_message("Kameravorschau (Dev Mode)\n\nKein Kamerabild verfügbar.")
            return

        overlay_mode = self._camera_preview_overlay_mode()
        if overlay_mode != "raw":
            try:
                frame = self._apply_camera_preview_overlay(frame, overlay_mode)
                self._camera_overlay_error_key = None
            except Exception as exc:
                self._set_camera_overlay_error_once(overlay_mode, exc)
                try:
                    cv2.putText(  # type: ignore[attr-defined]
                        frame,
                        f"Overlay-Fehler ({overlay_mode})",
                        (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX,  # type: ignore[attr-defined]
                        0.7,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,  # type: ignore[attr-defined]
                    )
                except Exception:
                    pass

        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # type: ignore[attr-defined]
        except Exception as exc:
            self._set_camera_preview_message(f"Kameravorschau (Dev Mode)\n\nKonvertierungsfehler: {exc}")
            return

        if len(rgb.shape) != 3:
            self._set_camera_preview_message("Kameravorschau (Dev Mode)\n\nUnerwartetes Bildformat.")
            return

        height, width, channels = rgb.shape
        bytes_per_line = channels * width
        image = QImage(rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888).copy()

        target_size = label.contentsRect().size()
        pixmap = QPixmap.fromImage(image)
        if target_size.width() > 0 and target_size.height() > 0:
            pixmap = pixmap.scaled(
                target_size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        label.setPixmap(pixmap)
        label.setText("")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def _group_box(self, title: str) -> QGroupBox:
        box = QGroupBox(title)
        box.setObjectName("SectionBox")
        return box

    def _section_label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("SectionLabel")
        return label

    def _mono_font(self) -> QFont:
        font = QFont("Consolas")
        if not font.exactMatch():
            font = QFont("Courier New")
        font.setStyleHint(QFont.StyleHint.Monospace)
        font.setPointSize(10)
        return font

    def _new_form_layout(self, parent: QGroupBox) -> QFormLayout:
        form = QFormLayout(parent)
        form.setContentsMargins(12, 16, 12, 12)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(8)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        form.setFormAlignment(Qt.AlignmentFlag.AlignTop)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        return form

    def _add_note(self, form: QFormLayout, text: str) -> None:
        note = QLabel(text)
        note.setWordWrap(True)
        note.setObjectName("MutedText")
        form.addRow(note)

    def _add_line_edit(self, form: QFormLayout, key: str, label: str) -> None:
        widget = QLineEdit()
        widget.textChanged.connect(self._on_form_change)
        self._widgets[key] = widget
        form.addRow(label, widget)

    def _add_combo(self, form: QFormLayout, key: str, label: str, values: list[str]) -> None:
        widget = QComboBox()
        widget.setEditable(False)
        widget.addItems(values)
        widget.currentTextChanged.connect(self._on_form_change)
        self._widgets[key] = widget
        form.addRow(label, widget)

    def _add_check(self, layout, key: str, label: str) -> None:
        widget = QCheckBox(label)
        widget.toggled.connect(self._on_form_change)
        self._widgets[key] = widget
        if isinstance(layout, QFormLayout):
            layout.addRow(widget)
            return
        layout.addWidget(widget)

    def _apply_theme(self) -> None:
        self.setStyleSheet(
            """
            QWidget#CentralRoot {
                background: #f4f7fb;
                color: #13233b;
            }
            QFrame#HeaderPanel {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                            stop:0 #0f766e, stop:1 #1d4ed8);
                border-radius: 14px;
            }
            QLabel#HeaderTitle {
                color: white;
                font-size: 22px;
                font-weight: 700;
            }
            QLabel#HeaderSubtitle {
                color: rgba(255,255,255,0.92);
                font-size: 12px;
            }
            QFrame#Panel {
                background: #ffffff;
                border: 1px solid #dbe5f0;
                border-radius: 14px;
            }
            QStackedWidget#LeftPages {
                background: transparent;
            }
            QWidget#StartScreen, QWidget#SettingsScreen {
                background: #ffffff;
            }
            QFrame#MainMenuCard {
                background: #ffffff;
                border: 1px solid #dbe5f0;
                border-radius: 14px;
            }
            QFrame#HeroCard {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                            stop:0 #f0fdfa, stop:1 #eff6ff);
                border: 1px solid #dbe5f0;
                border-radius: 12px;
            }
            QLabel#HeroTitle {
                color: #0f172a;
                font-size: 20px;
                font-weight: 700;
            }
            QFrame#SettingsSidebar {
                background: #f8fbff;
                border: 1px solid #dbe5f0;
                border-radius: 12px;
            }
            QStackedWidget#SettingsPages {
                background: #ffffff;
                border: 1px solid #dbe5f0;
                border-radius: 12px;
            }
            QWidget#TabContent {
                background: #ffffff;
            }
            QScrollArea#TabScrollArea {
                background: transparent;
                border: none;
            }
            QWidget#TabScrollViewport {
                background: #ffffff;
            }
            QTabWidget::pane {
                border: none;
                background: #ffffff;
            }
            QTabBar::tab {
                background: #eaf0f8;
                color: #334155;
                border: 1px solid #d7e2ef;
                border-bottom: none;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                padding: 8px 12px;
                margin-right: 4px;
            }
            QTabBar::tab:selected {
                background: white;
                color: #0f172a;
            }
            QListWidget#SettingsCategoryList {
                background: transparent;
                border: none;
                outline: none;
            }
            QListWidget#SettingsCategoryList::item {
                background: #eaf0f8;
                color: #334155;
                border: 1px solid #d7e2ef;
                border-radius: 8px;
                padding: 10px 12px;
                margin: 2px;
                min-height: 22px;
            }
            QListWidget#SettingsCategoryList::item:hover {
                background: #dee8f6;
                border-color: #c5d4e7;
            }
            QListWidget#SettingsCategoryList::item:selected {
                background: #1d4ed8;
                color: white;
                border-color: #1d4ed8;
            }
            QGroupBox#SectionBox {
                border: 1px solid #dbe5f0;
                border-radius: 12px;
                margin-top: 12px;
                background: #ffffff;
                font-weight: 600;
                color: #0f172a;
            }
            QGroupBox#SectionBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 6px;
            }
            QFrame#ModeCard {
                background: #f8fbff;
                border: 1px solid #dbe5f0;
                border-radius: 10px;
            }
            QLabel#MutedText {
                color: #55657f;
            }
            QLabel#SectionLabel {
                font-weight: 600;
                color: #0f172a;
            }
            QLabel#StatusText {
                color: #1d4ed8;
                font-weight: 600;
            }
            QLabel {
                color: #0f172a;
            }
            QCheckBox, QRadioButton {
                color: #0f172a;
            }
            QLineEdit, QComboBox, QPlainTextEdit {
                border: 1px solid #cfdbe8;
                border-radius: 8px;
                background: #ffffff;
                color: #0f172a;
                padding: 6px 8px;
                selection-background-color: #bfdbfe;
            }
            QComboBox QAbstractItemView {
                background: #ffffff;
                color: #0f172a;
                border: 1px solid #cfdbe8;
                selection-background-color: #dbeafe;
                selection-color: #0f172a;
            }
            QLineEdit:focus, QComboBox:focus, QPlainTextEdit:focus {
                border: 1px solid #60a5fa;
            }
            QPlainTextEdit#CommandPreview {
                background: #eef2ff;
                color: #1e293b;
            }
            QPlainTextEdit#LogOutput {
                background: #0f172a;
                color: #dbeafe;
                border: 1px solid #1e293b;
            }
            QLabel#CameraPreviewPlaceholder {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                            stop:0 #f8fafc, stop:1 #e2e8f0);
                color: #334155;
                border: 1px dashed #94a3b8;
                border-radius: 10px;
                padding: 10px;
                font-weight: 500;
            }
            QPushButton {
                border: 1px solid #cfdbe8;
                background: #ffffff;
                color: #0f172a;
                border-radius: 8px;
                padding: 7px 10px;
                font-weight: 500;
            }
            QPushButton:hover {
                background: #f3f7fc;
            }
            QPushButton#PrimaryButton {
                background: #1d4ed8;
                color: white;
                border-color: #1d4ed8;
            }
            QPushButton#PrimaryButton:hover {
                background: #1e40af;
                border-color: #1e40af;
            }
            QPushButton#DangerButton {
                background: #b91c1c;
                color: white;
                border-color: #b91c1c;
            }
            QPushButton#DangerButton:hover {
                background: #991b1b;
                border-color: #991b1b;
            }
            QPushButton#MenuPrimaryButton {
                background: #0f766e;
                color: white;
                border-color: #0f766e;
                padding: 10px 12px;
                font-weight: 600;
            }
            QPushButton#MenuPrimaryButton:hover {
                background: #0f5f59;
                border-color: #0f5f59;
            }
            QPushButton#MenuSecondaryButton {
                background: #eef2ff;
                color: #1e3a8a;
                border-color: #c7d2fe;
                padding: 9px 12px;
                font-weight: 600;
            }
            QPushButton#MenuSecondaryButton:hover {
                background: #e0e7ff;
                border-color: #a5b4fc;
            }
            QPushButton#MenuDangerButton {
                background: #fee2e2;
                color: #991b1b;
                border-color: #fecaca;
                padding: 9px 12px;
                font-weight: 600;
            }
            QPushButton#MenuDangerButton:hover {
                background: #fecaca;
                border-color: #fca5a5;
            }
            QPushButton#MenuDevButton {
                background: #ecfeff;
                color: #155e75;
                border-color: #a5f3fc;
                padding: 9px 12px;
                font-weight: 600;
            }
            QPushButton#MenuDevButton:hover {
                background: #cffafe;
                border-color: #67e8f9;
            }
            QScrollArea {
                background: transparent;
                border: none;
            }
            """
        )

    def _load_settings(self) -> LauncherSettings:
        if not self.settings_file.exists():
            return LauncherSettings()
        try:
            payload = json.loads(self.settings_file.read_text(encoding="utf-8"))
        except Exception:
            return LauncherSettings()
        return LauncherSettings.from_payload(payload)

    def _save_settings(self, quiet: bool = False) -> bool:
        try:
            payload = self._collect_settings_payload()
            self.settings_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as exc:
            if not quiet:
                QMessageBox.critical(self, "Einstellungen speichern", f"Einstellungen konnten nicht gespeichert werden:\n{exc}")
            return False
        if not quiet:
            self._set_status(f"Einstellungen gespeichert: {self.settings_file.name}")
        return True

    def _reload_saved_settings(self) -> None:
        self._apply_settings_to_widgets(self._load_settings())
        self._set_status("Gespeicherte Einstellungen geladen")

    def _reset_defaults(self) -> None:
        if not self._ask_yes_no("Standardwerte", "Launcher-Einstellungen auf Standardwerte zurücksetzen?"):
            return
        self._apply_settings_to_widgets(LauncherSettings())
        self._set_status("Standardwerte wiederhergestellt")

    def _ask_yes_no(self, title: str, text: str) -> bool:
        result = QMessageBox.question(
            self,
            title,
            text,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return result == QMessageBox.StandardButton.Yes

    def _apply_settings_to_widgets(self, settings: LauncherSettings) -> None:
        self._suppress_form_updates = True
        try:
            for key, widget in self._widgets.items():
                if not hasattr(settings, key):
                    continue
                value = getattr(settings, key)
                self._set_widget_value(widget, value)
            self._set_mode(settings.mode)
        finally:
            self._suppress_form_updates = False
        self._refresh_context()
        self._refresh_command_preview()

    def _set_widget_value(self, widget: QWidget, value: object) -> None:
        if isinstance(widget, QLineEdit):
            widget.setText("" if value is None else str(value))
            return
        if isinstance(widget, QComboBox):
            text = "" if value is None else str(value)
            idx = widget.findText(text)
            if idx >= 0:
                widget.setCurrentIndex(idx)
            elif widget.count() > 0:
                widget.setCurrentIndex(0)
            return
        if isinstance(widget, QCheckBox):
            widget.setChecked(bool(value))
            return
        raise TypeError(f"Nicht unterstützter Widget-Typ: {type(widget)!r}")

    def _collect_settings_payload(self) -> dict[str, object]:
        bool_keys = {
            "mill_flying",
            "mill_threefold_repetition",
            "mill_no_capture_draw",
            "mill_random_tiebreak",
            "mill_debug_vision",
            "mill_uarm_move_both_players",
        }
        int_keys = {
            "camera_index",
            "mill_max_plies",
            "mill_no_capture_draw_plies",
            "mill_ai_depth",
            "mill_seed",
            "mill_vision_attempts",
            "mill_robot_speed",
        }
        float_keys = {"mill_ai_temperature"}

        payload: dict[str, object] = {"mode": self._current_mode()}
        for key, widget in self._widgets.items():
            if key in bool_keys:
                payload[key] = self._widget_bool(widget)
            elif key in int_keys:
                text = self._widget_text(widget).strip()
                try:
                    payload[key] = int(text)
                except Exception:
                    payload[key] = text
            elif key in float_keys:
                text = self._widget_text(widget).strip()
                try:
                    payload[key] = float(text)
                except Exception:
                    payload[key] = text
            else:
                payload[key] = self._widget_text(widget).strip()
        return payload

    def _widget_text(self, widget: QWidget) -> str:
        if isinstance(widget, QLineEdit):
            return widget.text()
        if isinstance(widget, QComboBox):
            return widget.currentText()
        if isinstance(widget, QCheckBox):
            return "true" if widget.isChecked() else "false"
        raise TypeError(f"Nicht unterstützter Widget-Typ: {type(widget)!r}")

    def _widget_bool(self, widget: QWidget) -> bool:
        if not isinstance(widget, QCheckBox):
            raise TypeError(f"Bool-Feld erwartet QCheckBox, erhalten: {type(widget)!r}")
        return widget.isChecked()

    def _current_mode(self) -> str:
        for value, radio in self._mode_buttons.items():
            if radio.isChecked():
                return value
        return "vision-loop"

    def _set_mode(self, mode: object) -> None:
        mode_str = str(mode) if mode is not None else "vision-loop"
        if mode_str not in self._mode_buttons:
            mode_str = "vision-loop"
        self._mode_buttons[mode_str].setChecked(True)

    def _on_form_change(self, *_args: object) -> None:
        if self._suppress_form_updates:
            return
        self._refresh_context()
        self._refresh_command_preview()
        if self._camera_preview_enabled():
            self._start_camera_preview()

    def _refresh_context(self) -> None:
        if self._summary_label is None or self._hint_label is None:
            return

        mode = self._current_mode()
        if mode == "vision-loop":
            self._summary_label.setText(
                "Vision-Laufzeit ausgewählt. Gestartet wird der bestehende Vision-/Runtime-Loop mit dem angegebenen Kameraindex."
            )
        else:
            self._summary_label.setText(
                "Spielbare Mühle ausgewählt. Der Launcher übergibt Spiel-, Regel-, KI-, Vision- und Roboter-Optionen an die vorhandene CLI-Session."
            )

        hints: list[str] = []
        mill_mode = self._widgets["mill_mode"]
        human_input = self._widgets["mill_human_input"]
        ai_backend = self._widgets["mill_ai"]

        mill_mode_value = mill_mode.currentText() if isinstance(mill_mode, QComboBox) else ""
        human_input_value = human_input.currentText() if isinstance(human_input, QComboBox) else ""
        ai_backend_value = ai_backend.currentText() if isinstance(ai_backend, QComboBox) else ""

        if mode == "play-mill" and "human" in mill_mode_value and human_input_value == "manual":
            hints.append("Manuelle Züge können im Dev Mode über das Feld 'Prozess-Eingabe' gesendet werden.")
        if mode == "play-mill" and ai_backend_value == "neural":
            hints.append("Neural benötigt ML-Abhängigkeiten (z. B. torch) und ein lesbares Modell.")
        if mode == "play-mill" and human_input_value == "vision":
            hints.append("Vision-Eingabe benötigt Kalibrierungsdateien und eine verfügbare Kamera.")

        if not hints:
            hints.append("Einstellungen werden lokal gespeichert und beim nächsten Start wieder geladen.")
        self._hint_label.setText(" ".join(hints))

    def _refresh_command_preview(self) -> None:
        if self._command_preview is None:
            return
        try:
            preview = shlex.join(self._build_command())
        except ValueError as exc:
            preview = f"Ungültige Einstellungen: {exc}"
        self._command_preview.setPlainText(preview)

    def _parse_int(self, key: str, label: str, *, minimum: int | None = None) -> int:
        raw = self._widget_text(self._widgets[key]).strip()
        try:
            value = int(raw)
        except ValueError as exc:
            raise ValueError(f"{label} muss eine ganze Zahl sein") from exc
        if minimum is not None and value < minimum:
            raise ValueError(f"{label} muss >= {minimum} sein")
        return value

    def _parse_float(self, key: str, label: str) -> float:
        raw = self._widget_text(self._widgets[key]).strip()
        try:
            return float(raw)
        except ValueError as exc:
            raise ValueError(f"{label} muss eine Zahl sein") from exc

    def _parse_optional_xy(self, key: str, label: str) -> str | None:
        raw = self._widget_text(self._widgets[key]).strip()
        if not raw:
            return None
        parts = [part.strip() for part in raw.split(",")]
        if len(parts) != 2:
            raise ValueError(f"{label} muss im Format X,Y angegeben werden")
        try:
            float(parts[0])
            float(parts[1])
        except ValueError as exc:
            raise ValueError(f"{label} muss numerische X/Y-Werte enthalten") from exc
        return raw

    def _build_command(self) -> list[str]:
        mode = self._current_mode()
        if mode not in {"vision-loop", "play-mill"}:
            raise ValueError("Modus muss 'vision-loop' oder 'play-mill' sein")

        camera_index = self._parse_int("camera_index", "Kameraindex", minimum=0)
        cmd = [sys.executable, "-u", str(self.entry_script), "--mode", mode, "--camera-index", str(camera_index)]

        if mode == "vision-loop":
            return cmd

        mill_mode = self._widget_text(self._widgets["mill_mode"]).strip()
        human_color = self._widget_text(self._widgets["mill_human_color"]).strip()
        human_input = self._widget_text(self._widgets["mill_human_input"]).strip()
        ai_backend = self._widget_text(self._widgets["mill_ai"]).strip()
        ai_model = self._widget_text(self._widgets["mill_ai_model"]).strip()
        ai_device = self._widget_text(self._widgets["mill_ai_device"]).strip()
        robot_board_map = self._widget_text(self._widgets["mill_robot_board_map"]).strip()

        if mill_mode not in {"human-vs-human", "human-vs-ai", "ai-vs-ai"}:
            raise ValueError("Ungültiger Mühle-Spielmodus")
        if human_color not in {"W", "B"}:
            raise ValueError("Mensch-Farbe muss W oder B sein")
        if human_input not in {"manual", "vision"}:
            raise ValueError("Mensch-Eingabe muss 'manual' oder 'vision' sein")
        if ai_backend not in {"heuristic", "alphabeta", "neural"}:
            raise ValueError("KI-Backend muss heuristic, alphabeta oder neural sein")
        if robot_board_map not in {"default", "homography"}:
            raise ValueError("Brett-Mapping muss default oder homography sein")

        max_plies = self._parse_int("mill_max_plies", "Max. Halbzüge", minimum=0)
        no_capture_draw_plies = self._parse_int("mill_no_capture_draw_plies", "Remisgrenze (Halbzüge)", minimum=1)
        ai_depth = self._parse_int("mill_ai_depth", "AlphaBeta-Tiefe", minimum=1)
        ai_temperature = self._parse_float("mill_ai_temperature", "Temperatur")
        seed = self._parse_int("mill_seed", "Seed")
        vision_attempts = self._parse_int("mill_vision_attempts", "Scan-Versuche", minimum=1)
        robot_speed = self._parse_int("mill_robot_speed", "Robotergeschwindigkeit", minimum=1)

        white_reserve = self._parse_optional_xy("mill_white_reserve", "Weißer Vorrat")
        black_reserve = self._parse_optional_xy("mill_black_reserve", "Schwarzer Vorrat")
        capture_bin = self._parse_optional_xy("mill_capture_bin", "Ablage für Schläge")

        def add_bool(flag: str, value: bool) -> None:
            cmd.append(f"--{flag}" if value else f"--no-{flag}")

        cmd.extend(["--game-mode", mill_mode])
        cmd.extend(["--human-color", human_color])
        cmd.extend(["--human-input", human_input])
        cmd.extend(["--max-plies", str(max_plies)])
        add_bool("flying", self._widget_bool(self._widgets["mill_flying"]))
        add_bool("threefold-repetition", self._widget_bool(self._widgets["mill_threefold_repetition"]))
        add_bool("no-capture-draw", self._widget_bool(self._widgets["mill_no_capture_draw"]))
        cmd.extend(["--no-capture-draw-plies", str(no_capture_draw_plies)])

        cmd.extend(["--ai", ai_backend])
        cmd.extend(["--ai-depth", str(ai_depth)])
        if ai_model:
            cmd.extend(["--ai-model", ai_model])
        cmd.extend(["--ai-temperature", str(ai_temperature)])
        if ai_device:
            cmd.extend(["--ai-device", ai_device])
        add_bool("random-tiebreak", self._widget_bool(self._widgets["mill_random_tiebreak"]))
        cmd.extend(["--seed", str(seed)])

        cmd.extend(["--vision-attempts", str(vision_attempts)])
        add_bool("debug-vision", self._widget_bool(self._widgets["mill_debug_vision"]))

        uarm_port = self._widget_text(self._widgets["mill_uarm_port"]).strip()
        if uarm_port:
            cmd.extend(["--uarm-port", uarm_port])
        add_bool("uarm-move-both-players", self._widget_bool(self._widgets["mill_uarm_move_both_players"]))
        cmd.extend(["--robot-speed", str(robot_speed)])
        cmd.extend(["--robot-board-map", robot_board_map])

        if white_reserve:
            cmd.extend(["--white-reserve", white_reserve])
        if black_reserve:
            cmd.extend(["--black-reserve", black_reserve])
        if capture_bin:
            cmd.extend(["--capture-bin", capture_bin])

        return cmd

    def _start_process(self) -> None:
        if self._is_process_running():
            self._set_status("Es läuft bereits ein Prozess")
            return

        try:
            cmd = self._build_command()
        except ValueError as exc:
            QMessageBox.critical(self, "Starten nicht möglich", str(exc))
            self._set_status("Einstellungen prüfen")
            return

        self._save_settings(quiet=True)
        self._clear_stdin()
        self._append_log(f"\n[launcher] Starte Prozess:\n{shlex.join(cmd)}\n\n")

        if self._process is None:
            self._setup_process()
        assert self._process is not None

        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        env.insert("PYTHONUTF8", "1")
        self._process.setProcessEnvironment(env)
        self._process.setWorkingDirectory(str(self.project_root))
        self._process.setProgram(cmd[0])
        self._process.setArguments(cmd[1:])

        self._process.start()
        if not self._process.waitForStarted(1500):
            error = self._process.errorString() or "Unbekannter Startfehler"
            self._append_log(f"[launcher] Start fehlgeschlagen: {error}\n")
            QMessageBox.critical(self, "Start fehlgeschlagen", f"Prozess konnte nicht gestartet werden:\n{error}")
            self._set_status("Start fehlgeschlagen")
            self._sync_runtime_controls()
            return

        self._set_status(f"Läuft ({self._current_mode()})")
        self._sync_runtime_controls()

    def _stop_process(self) -> None:
        if not self._is_process_running() or self._process is None:
            self._set_status("Kein laufender Prozess")
            self._sync_runtime_controls()
            return
        self._append_log("[launcher] Prozess wird beendet ...\n")
        self._process.terminate()
        self._set_status("Wird beendet ...")
        QTimer.singleShot(1500, self._kill_if_still_running)

    def _kill_if_still_running(self) -> None:
        if not self._is_process_running() or self._process is None:
            return
        self._append_log("[launcher] Prozess reagiert nicht, erzwinge Beenden.\n")
        self._process.kill()

    def _is_process_running(self) -> bool:
        return self._process is not None and self._process.state() != QProcess.ProcessState.NotRunning

    def _read_process_output(self) -> None:
        if self._process is None:
            return
        qbytes = self._process.readAllStandardOutput()
        data = bytes(qbytes.data())
        if not data:
            return
        self._append_log(data.decode("utf-8", errors="replace"))

    def _on_process_finished(self, exit_code: int, _exit_status) -> None:
        self._read_process_output()
        self._append_log(f"\n[launcher] Prozess beendet mit Exit-Code {exit_code}\n")
        self._set_status(f"Beendet (Code {exit_code})")
        self._sync_runtime_controls()

    def _on_process_error(self, error) -> None:
        # Normalerweise folgt darauf ein finished-Signal. Wir loggen den Zustand trotzdem,
        # damit Startprobleme im UI sichtbar sind.
        if self._process is None:
            return
        self._append_log(f"[launcher] Prozessfehler: {error} | {self._process.errorString()}\n")
        self._sync_runtime_controls()

    def _on_process_state_changed(self, _state) -> None:
        if self._is_process_running():
            self._set_status(f"Läuft ({self._current_mode()})")
        self._sync_runtime_controls()

    def _send_process_input(self) -> None:
        if self._stdin_input is None:
            return
        text = self._stdin_input.text()
        self._send_text_to_process(text)
        self._clear_stdin()

    def _send_empty_line(self) -> None:
        self._send_text_to_process("")

    def _send_text_to_process(self, text: str) -> None:
        if not self._is_process_running() or self._process is None:
            self._set_status("Kein laufender Prozess für Eingabe")
            return
        payload = f"{text}\n".encode("utf-8")
        written = self._process.write(payload)
        if written < 0:
            self._append_log("[launcher] Eingabe konnte nicht gesendet werden.\n")
            return
        self._process.waitForBytesWritten(500)
        echoed = "<ENTER>" if text == "" else text
        self._append_log(f"[launcher -> stdin] {echoed}\n")

    def _clear_stdin(self) -> None:
        if self._stdin_input is not None:
            self._stdin_input.clear()

    def _clear_log(self) -> None:
        if self._log_output is not None:
            self._log_output.clear()

    def _append_log(self, text: str) -> None:
        if self._log_output is None:
            return
        cursor = self._log_output.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(text)
        self._log_output.setTextCursor(cursor)
        self._log_output.ensureCursorVisible()

    def _set_status(self, text: str) -> None:
        if self._status_label is not None:
            self._status_label.setText(text)

    def _sync_runtime_controls(self) -> None:
        running = self._is_process_running()
        if self._start_button is not None:
            self._start_button.setEnabled(not running)
        if self._stop_button is not None:
            self._stop_button.setEnabled(running)
        if self._send_button is not None:
            self._send_button.setEnabled(running)
        if self._stdin_input is not None:
            self._stdin_input.setEnabled(running)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 (Qt API)
        if self._is_process_running() and self._process is not None:
            if not self._ask_yes_no("Launcher schließen", "Es läuft noch ein Prozess. Prozess beenden und Launcher schließen?"):
                event.ignore()
                return
            self._append_log("[launcher] Launcher wird geschlossen, Prozess wird beendet ...\n")
            self._process.terminate()
            if not self._process.waitForFinished(2000):
                self._append_log("[launcher] Erzwinge Beenden beim Schließen ...\n")
                self._process.kill()
                self._process.waitForFinished(1000)

        self._stop_camera_preview(keep_label_text=True)
        self._save_settings(quiet=True)
        event.accept()
        super().closeEvent(event)


def launch_launcher(entry_script: Path) -> int:
    app = QApplication.instance()
    owns_app = app is None
    if app is None:
        app = QApplication(sys.argv)
        app.setStyle("Fusion")
    assert app is not None

    app.setApplicationName("Gaming Robot Arm")
    app.setOrganizationName("GamingRobotArm")

    window = LauncherWindow(entry_script=entry_script)
    window.show()

    if owns_app:
        return app.exec()
    return 0
