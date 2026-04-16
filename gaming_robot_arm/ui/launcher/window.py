"""PySide6-basierter Desktop-Launcher fuer die bestehenden CLI-Modi."""

from __future__ import annotations

import shlex
import sys
from pathlib import Path

from PySide6.QtCore import QProcess, QTimer, Qt
from PySide6.QtGui import QCloseEvent, QFont, QImage, QPixmap, QTextCursor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
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
    QScrollArea,
    QSplitter,
    QStackedWidget,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from .command_builder import build_command
from .preview import load_board_overlay_detector, load_figure_overlay_detector
from .process_runner import start_qprocess
from .settings import LauncherSettings, load_launcher_settings, save_launcher_settings

try:
    import cv2
except ModuleNotFoundError:  # pragma: no cover - optionaler Laufzeitpfad
    cv2 = None


class LauncherWindow(QMainWindow):
    def __init__(self, entry_script: Path) -> None:
        super().__init__()
        self.entry_script = Path(entry_script).resolve()
        self.project_root = self.entry_script.parent
        self.settings_file = self.project_root / ".gaming_robot_arm_launcher.json"

        self._suppress_form_updates = False
        self._widgets: dict[str, QWidget] = {}
        self._segment_buttons: dict[str, dict[str, QPushButton]] = {}
        self._player_role_buttons: dict[str, dict[str, QPushButton]] = {}
        self._launch_mode = "play-mill"
        self._process: QProcess | None = None

        self._status_label: QLabel | None = None
        self._command_preview: QPlainTextEdit | None = None
        self._log_output: QPlainTextEdit | None = None
        self._stdin_input: QLineEdit | None = None
        self._start_button: QPushButton | None = None
        self._stop_button: QPushButton | None = None
        self._quick_start_button: QPushButton | None = None
        self._runtime_back_button: QPushButton | None = None
        self._send_button: QPushButton | None = None
        self._left_pages: QStackedWidget | None = None
        self._settings_category_list: QListWidget | None = None
        self._settings_pages: QStackedWidget | None = None
        self._body_splitter: QSplitter | None = None
        self._left_panel: QFrame | None = None
        self._right_panel: QFrame | None = None
        self._status_row_frame: QFrame | None = None
        self._camera_box: QGroupBox | None = None
        self._command_box: QGroupBox | None = None
        self._input_box: QGroupBox | None = None
        self._log_box: QGroupBox | None = None
        self._camera_preview_label: QLabel | None = None
        self._camera_preview_overlay_combo: QComboBox | None = None
        self._camera_preview_timer: QTimer | None = None
        self._camera_capture = None
        self._camera_preview_index: int | None = None
        self._camera_preview_active = False
        self._runtime_output_only_active = False
        self._board_overlay_detector = None
        self._figure_overlay_detector = None
        self._camera_figure_board_coords_cache: dict[bool, dict[str, tuple[int, int]]] = {}
        self._camera_overlay_error_key: str | None = None
        self._figure_stabilizer = None
        self._figure_stabilizer_labels: list = []
        self._stop_requested = False
        self._stop_force_killed = False
        self._stop_request_id = 0

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
        self.setWindowTitle("Gaming Robot Arm")
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

        title = QLabel("Gaming Robot Arm")
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
        category_list.addItems(["Kamera", "Mühle", "KI", "uArm"])
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
        pages.addWidget(self._build_scrollable_page(self._build_camera_tab))
        pages.addWidget(self._build_scrollable_page(self._build_mill_tab))
        pages.addWidget(self._build_scrollable_page(self._build_ai_tab))
        pages.addWidget(self._build_scrollable_page(self._build_uarm_tab))
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
        start_vision_btn = QPushButton("Vision-Laufzeit starten")
        start_vision_btn.setObjectName("PrimaryButton")
        start_vision_btn.clicked.connect(self._start_vision_from_dev)
        go_launch_btn = QPushButton("Zu Spiel Starten")
        go_launch_btn.setObjectName("MenuSecondaryButton")
        go_launch_btn.clicked.connect(self._show_launch_screen)
        go_settings_btn = QPushButton("Zu Einstellungen")
        go_settings_btn.setObjectName("MenuSecondaryButton")
        go_settings_btn.clicked.connect(self._show_settings_screen)
        shortcuts_layout.addWidget(start_vision_btn)
        shortcuts_layout.addWidget(go_launch_btn)
        shortcuts_layout.addWidget(go_settings_btn)
        layout.addWidget(shortcuts)

    def _build_launch_content(self, layout: QVBoxLayout) -> None:
        self._register_hidden_combo("mill_mode", ["human-vs-human", "human-vs-ai", "ai-vs-ai"], default="human-vs-ai")
        self._register_hidden_combo("mill_human_input", ["manual", "vision", "voice"], default="manual")
        self._register_hidden_combo(
            "mill_uarm_controlled_players",
            ["white", "black", "both", "none", "legacy"],
            default="legacy",
        )
        self._register_hidden_combo("mill_human_color", ["W", "B"], default="W")
        self._register_hidden_check("mill_uarm_enable_ai_moves", checked=False)
        self._register_hidden_check("mill_uarm_move_both_players", checked=False)

        self._add_player_role_group(layout)
        self._add_segmented_group(
            layout,
            title="uArm-Support",
            key="mill_uarm_controlled_players",
            options=[
                ("uArm bewegt weiß", "white"),
                ("uArm bewegt schwarz", "black"),
                ("uArm bewegt beide", "both"),
            ],
        )
        self._add_segmented_group(
            layout,
            title="Eingabe der Spielzüge",
            key="mill_human_input",
            options=[
                ("Kamera", "vision"),
                ("Tastatur", "manual"),
                ("Sprache", "voice"),
            ],
        )

        start_row = QHBoxLayout()
        start_row.setSpacing(0)
        start_row.addStretch(1)
        self._start_button = QPushButton("Jetzt starten")
        self._start_button.setObjectName("PrimaryButton")
        self._start_button.setMinimumWidth(240)
        self._start_button.clicked.connect(self._start_play_mill_from_launch)
        start_row.addWidget(self._start_button)
        start_row.addStretch(1)
        layout.addLayout(start_row)

        layout.addStretch(1)
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

    def _add_player_role_group(self, layout: QVBoxLayout) -> None:
        box = self._group_box("Spielmodus")
        box_layout = QVBoxLayout(box)
        box_layout.setContentsMargins(12, 14, 12, 12)
        box_layout.setSpacing(8)

        self._player_role_buttons = {}
        for player, title in (("W", "Spieler weiß:"), ("B", "Spieler schwarz:")):
            row = QHBoxLayout()
            row.setSpacing(8)

            label = QLabel(title)
            label.setObjectName("MutedText")
            label.setMinimumWidth(110)
            row.addWidget(label)

            buttons: dict[str, QPushButton] = {}
            for button_text, role in (("Mensch", "human"), ("uArm", "uarm")):
                button = QPushButton(button_text)
                button.setObjectName("SegmentOption")
                button.setCheckable(True)
                button.clicked.connect(
                    lambda _checked=False, player_value=player, role_value=role: self._set_player_role(player_value, role_value)
                )
                row.addWidget(button)
                buttons[role] = button
            row.addStretch(1)
            box_layout.addLayout(row)
            self._player_role_buttons[player] = buttons

        layout.addWidget(box)
        self._sync_player_role_buttons()

    def _register_hidden_combo(self, key: str, values: list[str], *, default: str) -> QComboBox:
        existing = self._widgets.get(key)
        if isinstance(existing, QComboBox):
            return existing
        widget = QComboBox(self)
        widget.setEditable(False)
        for value in values:
            widget.addItem(value, value)
        widget.setVisible(False)
        idx = widget.findData(default)
        if idx < 0:
            idx = widget.findText(default)
        if idx >= 0:
            widget.setCurrentIndex(idx)
        widget.currentTextChanged.connect(self._on_form_change)
        self._widgets[key] = widget
        return widget

    def _register_hidden_check(self, key: str, *, checked: bool) -> QCheckBox:
        existing = self._widgets.get(key)
        if isinstance(existing, QCheckBox):
            return existing
        widget = QCheckBox(self)
        widget.setVisible(False)
        widget.setChecked(checked)
        widget.toggled.connect(self._on_form_change)
        self._widgets[key] = widget
        return widget

    def _add_segmented_group(
        self,
        layout: QVBoxLayout,
        *,
        title: str,
        key: str,
        options: list[tuple[str, str]],
        disabled_labels: list[str] | None = None,
        note_text: str | None = None,
    ) -> None:
        box = self._group_box(title)
        box_layout = QVBoxLayout(box)
        box_layout.setContentsMargins(12, 14, 12, 12)
        box_layout.setSpacing(8)

        row = QHBoxLayout()
        row.setSpacing(8)
        buttons: dict[str, QPushButton] = {}
        for label, value in options:
            button = QPushButton(label)
            button.setObjectName("SegmentOption")
            button.setCheckable(True)
            button.clicked.connect(lambda _checked=False, k=key, v=value: self._set_segment_value(k, v))
            row.addWidget(button)
            buttons[value] = button

        for disabled_label in disabled_labels or []:
            button = QPushButton(disabled_label)
            button.setObjectName("SegmentOption")
            button.setEnabled(False)
            button.setToolTip("Platzhalter: Funktion wird später ergänzt.")
            row.addWidget(button)

        box_layout.addLayout(row)
        if note_text:
            note = QLabel(note_text)
            note.setWordWrap(True)
            note.setObjectName("MutedText")
            box_layout.addWidget(note)
        layout.addWidget(box)

        self._segment_buttons[key] = buttons
        combo = self._widgets.get(key)
        if isinstance(combo, QComboBox):
            combo.currentTextChanged.connect(lambda _text, segment_key=key: self._sync_segment_buttons(segment_key))
        self._sync_segment_buttons(key)

    def _set_segment_value(self, key: str, value: str) -> None:
        widget = self._widgets.get(key)
        if not isinstance(widget, QComboBox):
            return
        idx = widget.findText(value)
        if idx < 0:
            return
        if widget.currentIndex() == idx:
            self._sync_segment_buttons(key)
            return
        widget.setCurrentIndex(idx)

    def _sync_segment_buttons(self, key: str) -> None:
        widget = self._widgets.get(key)
        buttons = self._segment_buttons.get(key)
        if not isinstance(widget, QComboBox) or buttons is None:
            return
        selected = widget.currentText().strip()
        for value, button in buttons.items():
            is_active = value == selected
            if button.isChecked() != is_active:
                button.setChecked(is_active)

    def _current_player_roles(self) -> dict[str, str]:
        mode_widget = self._widgets.get("mill_mode")
        human_color_widget = self._widgets.get("mill_human_color")

        mode_value = mode_widget.currentText().strip() if isinstance(mode_widget, QComboBox) else "human-vs-ai"
        human_color = human_color_widget.currentText().strip() if isinstance(human_color_widget, QComboBox) else "W"
        if human_color not in {"W", "B"}:
            human_color = "W"

        if mode_value == "human-vs-human":
            return {"W": "human", "B": "human"}
        if mode_value == "ai-vs-ai":
            return {"W": "uarm", "B": "uarm"}
        if mode_value == "human-vs-ai":
            if human_color == "W":
                return {"W": "human", "B": "uarm"}
            return {"W": "uarm", "B": "human"}
        return {"W": "human", "B": "human"}

    def _set_player_role(self, player: str, role: str) -> None:
        if player not in {"W", "B"} or role not in {"human", "uarm"}:
            return
        roles = self._current_player_roles()
        if roles.get(player) == role:
            return
        roles[player] = role

        white_role = roles.get("W", "human")
        black_role = roles.get("B", "human")
        if white_role == "human" and black_role == "human":
            target_mode = "human-vs-human"
            target_human_color = "W"
        elif white_role == "uarm" and black_role == "uarm":
            target_mode = "ai-vs-ai"
            target_human_color = "W"
        elif white_role == "human" and black_role == "uarm":
            target_mode = "human-vs-ai"
            target_human_color = "W"
        else:
            target_mode = "human-vs-ai"
            target_human_color = "B"

        self._set_combo_text_safely("mill_mode", target_mode)
        self._set_combo_text_safely("mill_human_color", target_human_color)
        self._on_form_change()

    def _sync_player_role_buttons(self) -> None:
        roles = self._current_player_roles()
        for player, buttons in self._player_role_buttons.items():
            selected_role = roles.get(player, "human")
            for role, button in buttons.items():
                is_active = role == selected_role
                if button.isChecked() != is_active:
                    button.setChecked(is_active)

    def _apply_uarm_support_constraints(self) -> None:
        roles = self._current_player_roles()
        white_is_uarm = roles.get("W") == "uarm"
        black_is_uarm = roles.get("B") == "uarm"

        if white_is_uarm and black_is_uarm:
            allowed_values = {"both"}
            fallback_value = "both"
        elif white_is_uarm:
            allowed_values = {"white", "both"}
            fallback_value = "white"
        elif black_is_uarm:
            allowed_values = {"black", "both"}
            fallback_value = "black"
        else:
            allowed_values = {"white", "black", "both"}
            fallback_value = "both"

        support_widget = self._widgets.get("mill_uarm_controlled_players")
        support_buttons = self._segment_buttons.get("mill_uarm_controlled_players")
        if not isinstance(support_widget, QComboBox) or support_buttons is None:
            return

        current_value = support_widget.currentText().strip().lower()
        if current_value not in allowed_values:
            self._set_combo_text_safely("mill_uarm_controlled_players", fallback_value)

        for value, button in support_buttons.items():
            allowed = value in allowed_values
            button.setEnabled(allowed)
            if allowed:
                if button.toolTip() == "Nicht verfügbar für aktuelle Rollenwahl.":
                    button.setToolTip("")
            else:
                button.setToolTip("Nicht verfügbar für aktuelle Rollenwahl.")
        self._sync_segment_buttons("mill_uarm_controlled_players")

    def _build_camera_tab(self, layout: QVBoxLayout) -> None:
        camera_box = self._group_box("Kamera")
        camera_form = self._new_form_layout(camera_box)
        self._add_line_edit(
            camera_form,
            "camera_index",
            "Kameraindex",
            note_text="Index der verwendeten Kamera. 0 ist normalerweise die Standardkamera.",
        )
        self._add_check(
            camera_form,
            "mill_record_game",
            "Spiel aufzeichnen (Video)",
            note_text="Speichert beim Start von 'Spielbare Mühle' ein MP4 der Partie.",
        )
        layout.addWidget(camera_box)

        vision_box = self._group_box("Vision-Brücke")
        vision_form = self._new_form_layout(vision_box)
        self._add_line_edit(
            vision_form,
            "mill_vision_attempts",
            "Scan-Versuche",
            note_text="Anzahl der Scan-Wiederholungen, wenn die Kamera einen Zustand nicht sicher erkennt.",
        )
        self._add_combo(
            vision_form,
            "mill_vision_trigger",
            "Vision-Trigger",
            [
                ("Automatisch nach Brettaenderung", "auto"),
                ("Manuell (Enter)", "manual"),
            ],
            note_text="Legt fest, ob Vision-Zuege automatisch nach stabiler Brettaenderung oder manuell per Enter-Scan erkannt werden.",
        )
        self._add_check(
            vision_form,
            "mill_vision_baseline_timeout_disabled",
            "Baseline-Timeout deaktivieren",
            note_text="Deaktiviert den Timeout beim Warten auf ein ruhiges Brett. Nützlich bei langsamen oder instabilen Kameras.",
        )
        self._add_check(
            vision_form,
            "mill_debug_vision",
            "Debug-Logging für Vision-Zuordnung",
            note_text="Schreibt zusätzliche Vision-Details in die Logs, um Erkennung und Zuordnung zu prüfen.",
        )
        self._add_check(
            vision_form,
            "mill_vision_preview",
            "Live-Vorschau mit Detector-Overlay",
            note_text="Öffnet beim Spielstart ein Fenster mit Live-Kamerabild, gelben Brettpunkten (A1–C8) und erkannten Figuren inkl. Zuordnung.",
        )
        layout.addWidget(vision_box)

    def _build_mill_tab(self, layout: QVBoxLayout) -> None:
        game_box = self._group_box("Spielablauf")
        game_form = self._new_form_layout(game_box)
        self._add_line_edit(
            game_form,
            "mill_max_plies",
            "Max. Halbzüge",
            note_text="Maximale Anzahl an Halbzügen. 0 bedeutet keine Begrenzung.",
        )
        layout.addWidget(game_box)

        rules_box = self._group_box("Regeln")
        rules_layout = QVBoxLayout(rules_box)
        rules_layout.setContentsMargins(12, 14, 12, 12)
        rules_layout.setSpacing(6)
        self._add_check(
            rules_layout,
            "mill_flying",
            "Flying-Regel aktivieren",
            note_text="Erlaubt einer Farbe mit drei Steinen das Springen auf jedes freie Feld.",
        )
        self._add_check(
            rules_layout,
            "mill_threefold_repetition",
            "Remis bei Dreifachwiederholung aktivieren",
            note_text="Beendet die Partie remis, wenn dieselbe Stellung dreimal erreicht wird.",
        )
        self._add_check(
            rules_layout,
            "mill_no_capture_draw",
            "Remis ohne Schlagserie aktivieren",
            note_text="Aktiviert ein Remis, wenn über viele Halbzüge kein Stein geschlagen wird.",
        )

        draw_form = QFormLayout()
        draw_form.setHorizontalSpacing(12)
        draw_form.setVerticalSpacing(8)
        draw_form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        draw_form.setFormAlignment(Qt.AlignmentFlag.AlignTop)
        draw_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        self._add_line_edit(
            draw_form,
            "mill_no_capture_draw_plies",
            "Remisgrenze (Halbzüge)",
            note_text="Grenzwert in Halbzügen für das Remis ohne Schlagserie.",
        )
        rules_layout.addLayout(draw_form)
        layout.addWidget(rules_box)

    def _build_ai_tab(self, layout: QVBoxLayout) -> None:
        ai_box = self._group_box("KI-Einstellungen")
        ai_form = self._new_form_layout(ai_box)
        self._add_combo(
            ai_form,
            "mill_ai",
            "Backend",
            ["heuristic", "alphabeta", "neural"],
            note_text="Wählt das KI-Backend für Computerzüge.",
        )
        self._add_line_edit(
            ai_form,
            "mill_ai_depth",
            "AlphaBeta-Tiefe",
            note_text="Suchtiefe für AlphaBeta. Höhere Werte sind stärker, aber langsamer.",
        )
        self._add_line_edit(
            ai_form,
            "mill_ai_model",
            "Modellpfad",
            note_text="Pfad zum Modell-Checkpoint für das Backend 'neural'.",
        )
        self._add_line_edit(
            ai_form,
            "mill_ai_temperature",
            "Temperatur",
            note_text="Steuert die Zufälligkeit bei der Zugauswahl. 0.0 ist deterministisch.",
        )
        self._add_line_edit(
            ai_form,
            "mill_ai_device",
            "Gerät",
            note_text="Ausführungsgerät für das neuronale Modell, z. B. auto, cpu oder cuda.",
        )
        self._add_line_edit(
            ai_form,
            "mill_seed",
            "Seed",
            note_text="Startwert für reproduzierbare Zufallsentscheidungen.",
        )
        layout.addWidget(ai_box)

        ai_flags = self._group_box("KI-Optionen")
        ai_flags_layout = QVBoxLayout(ai_flags)
        ai_flags_layout.setContentsMargins(12, 14, 12, 12)
        ai_flags_layout.setSpacing(6)
        self._add_check(
            ai_flags_layout,
            "mill_random_tiebreak",
            "Zufällige Tie-Breaks bei gleicher Bewertung",
            note_text="Löst gleich bewertete Züge zufällig statt stabil nach Reihenfolge auf.",
        )
        layout.addWidget(ai_flags)

    def _build_uarm_tab(self, layout: QVBoxLayout) -> None:
        robot_box = self._group_box("uArm")
        robot_form = self._new_form_layout(robot_box)
        self._add_line_edit(
            robot_form,
            "mill_uarm_port",
            "Serieller Port",
            note_text="Optionaler serieller Port für den uArm. Leer bedeutet Backend-Default bzw. Auto-Erkennung.",
        )
        self._add_line_edit(
            robot_form,
            "mill_robot_speed",
            "Robotergeschwindigkeit",
            note_text="Bewegungsgeschwindigkeit des uArm für Greif- und Ablagevorgänge.",
        )
        self._add_combo(
            robot_form,
            "mill_robot_board_map",
            "Brett-Mapping",
            ["default", "homography"],
            note_text="Wählt das Brett-Mapping zwischen festen Standardkoordinaten und Homographie-basiertem Mapping.",
        )
        layout.addWidget(robot_box)

    def _build_right_panel(self, layout: QVBoxLayout) -> None:
        status_frame = QFrame()
        status_row = QHBoxLayout(status_frame)
        status_row.setContentsMargins(0, 0, 0, 0)
        status_row.setSpacing(8)
        status_row.addWidget(self._section_label("Status"))
        self._status_label = QLabel("Bereit")
        self._status_label.setObjectName("StatusText")
        status_row.addWidget(self._status_label)
        status_row.addStretch(1)
        self._status_row_frame = status_frame
        layout.addWidget(status_frame)

        camera_box = self._group_box("Kameravorschau")
        self._camera_box = camera_box
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
        camera_placeholder.setMinimumHeight(320)
        camera_placeholder.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        camera_layout.addWidget(camera_placeholder)
        self._camera_preview_label = camera_placeholder
        layout.addWidget(camera_box, 2)

        command_box = self._group_box("Befehlsvorschau (Spielstart)")
        self._command_box = command_box
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
        self._input_box = input_box
        input_layout = QVBoxLayout(input_box)
        input_layout.setContentsMargins(12, 14, 12, 12)
        input_layout.setSpacing(8)

        input_note = QLabel(
            "Für Mühle-Eingaben und Vision-Fallbacks (Zugnummer, q oder manueller Enter-Scan). Die Eingabe wird an den laufenden Unterprozess gesendet."
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
        self._log_box = log_box
        log_layout = QVBoxLayout(log_box)
        log_layout.setContentsMargins(12, 14, 12, 12)
        log_layout.setSpacing(8)

        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)
        clear_btn = QPushButton("Log leeren")
        clear_btn.clicked.connect(self._clear_log)
        save_btn = QPushButton("Einstellungen speichern")
        save_btn.clicked.connect(self._save_settings)
        back_btn = QPushButton("Zurück")
        back_btn.clicked.connect(self._show_launch_screen)
        self._runtime_back_button = back_btn
        quick_stop = QPushButton("Stoppen")
        quick_stop.setObjectName("DangerButton")
        quick_stop.clicked.connect(self._stop_process)
        self._stop_button = quick_stop
        quick_start = QPushButton("Start")
        quick_start.setObjectName("PrimaryButton")
        quick_start.clicked.connect(self._start_process)
        self._quick_start_button = quick_start

        toolbar.addWidget(clear_btn)
        toolbar.addWidget(save_btn)
        toolbar.addWidget(back_btn)
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

    def _runtime_requires_stdin(self) -> bool:
        if self._current_mode() != "play-mill":
            return False
        input_widget = self._widgets.get("mill_human_input")
        mode_widget = self._widgets.get("mill_mode")
        if not isinstance(input_widget, QComboBox) or not isinstance(mode_widget, QComboBox):
            return False
        human_input = input_widget.currentText().strip()
        mill_mode = mode_widget.currentText().strip()
        has_human_player = "human" in mill_mode
        return has_human_player and human_input in {"manual", "vision"}

    def _set_runtime_output_only(self, active: bool) -> None:
        self._runtime_output_only_active = active
        if self._left_panel is not None:
            self._left_panel.setVisible(not active)

        if active:
            self._camera_preview_active = False
            self._stop_camera_preview(keep_label_text=True)
            if self._right_panel is not None:
                self._right_panel.setVisible(True)
            if self._body_splitter is not None:
                self._body_splitter.setSizes([1, 1340])

        show_detail_sections = not active
        show_input_section = show_detail_sections or self._runtime_requires_stdin()
        if self._status_row_frame is not None:
            self._status_row_frame.setVisible(show_detail_sections)
        if self._camera_box is not None:
            self._camera_box.setVisible(show_detail_sections)
        if self._command_box is not None:
            self._command_box.setVisible(show_detail_sections)
        if self._input_box is not None:
            self._input_box.setVisible(show_input_section)
        if self._runtime_back_button is not None:
            self._runtime_back_button.setVisible(active)

    def _set_dev_panel_visible(self, visible: bool, *, camera_preview: bool = False) -> None:
        if self._right_panel is None:
            return
        if not self._runtime_output_only_active and self._left_panel is not None:
            self._left_panel.setVisible(True)
        self._camera_preview_active = bool(visible and camera_preview)
        self._right_panel.setVisible(visible)
        if self._body_splitter is not None:
            if visible:
                self._body_splitter.setSizes([520, 820])
            else:
                self._body_splitter.setSizes([1, 0])
        if self._camera_preview_active:
            self._start_camera_preview()
        else:
            self._stop_camera_preview()

    def _show_home_screen(self) -> None:
        self._set_runtime_output_only(False)
        if self._left_pages is not None:
            self._left_pages.setCurrentIndex(0)
        self._set_dev_panel_visible(False)

    def _show_launch_screen(self) -> None:
        self._set_mode("play-mill")
        self._set_runtime_output_only(False)
        if self._left_pages is not None:
            self._left_pages.setCurrentIndex(1)
        self._set_dev_panel_visible(False)
        self._refresh_context()
        self._refresh_command_preview()

    def _show_settings_screen(self) -> None:
        self._set_runtime_output_only(False)
        if self._left_pages is not None:
            self._left_pages.setCurrentIndex(2)
        self._set_dev_panel_visible(False)
        if self._settings_category_list is not None and self._settings_pages is not None:
            row = max(0, self._settings_category_list.currentRow())
            self._settings_category_list.setCurrentRow(row)
            self._settings_pages.setCurrentIndex(row)

    def _show_dev_screen(self) -> None:
        self._set_mode("vision-loop")
        self._set_runtime_output_only(False)
        if self._left_pages is not None:
            self._left_pages.setCurrentIndex(3)
        self._set_dev_panel_visible(True, camera_preview=True)
        self._refresh_context()
        self._refresh_command_preview()

    def _start_play_mill_from_launch(self) -> None:
        self._set_mode("play-mill")
        self._refresh_context()
        self._refresh_command_preview()
        self._start_process()

    def _start_vision_from_dev(self) -> None:
        self._set_mode("vision-loop")
        self._refresh_context()
        self._refresh_command_preview()
        self._start_process()

    def _camera_preview_enabled(self) -> bool:
        return self._camera_preview_active and self._right_panel is not None and self._right_panel.isVisible()

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
        self._figure_stabilizer = None
        self._figure_stabilizer_labels = []
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
        detector = load_board_overlay_detector()
        self._board_overlay_detector = detector
        return detector

    def _load_figure_overlay_detector(self):
        detector = self._figure_overlay_detector
        if callable(detector):
            return detector
        detector = load_figure_overlay_detector()
        self._figure_overlay_detector = detector
        return detector

    def _camera_preview_board_coords(self, frame) -> dict[str, tuple[int, int]] | None:
        cached = self._camera_figure_board_coords_cache.get(True)
        if cached is not None:
            return cached
        try:
            from gaming_robot_arm.vision.mill_board_detector import detect_board_positions
            from gaming_robot_arm.games.mill.core.board import BOARD_LABELS
            positions, _ = detect_board_positions(frame, debug=False, return_bw=False)
            if len(positions) != len(BOARD_LABELS):
                return None
            labeled = {lbl: (int(x), int(y)) for lbl, (x, y) in zip(BOARD_LABELS, positions)}
            self._camera_figure_board_coords_cache[True] = labeled
            return labeled
        except Exception as exc:
            self._append_log(f"[launcher] Figure-Overlay: Live-Brett-Detektion fehlgeschlagen: {exc}\n")
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
            board_coords = self._camera_preview_board_coords(annotated_input)
            labels_order = sorted(board_coords.keys()) if board_coords else None
            result = detect_figures(
                annotated_input,
                board_coords=board_coords,
                labels_order=labels_order,
                draw_assignments=bool(board_coords),
                return_assignments=True,
            )
            if isinstance(result, tuple) and len(result) > 0:
                annotated = result[0]
            else:
                annotated = annotated_input
            raw_assignments = result[4] if isinstance(result, tuple) and len(result) > 4 else []

            if board_coords is not None and labels_order:
                if self._figure_stabilizer is None or self._figure_stabilizer_labels != labels_order:
                    from gaming_robot_arm.vision.figure_detector import AssignmentStabilizer
                    self._figure_stabilizer = AssignmentStabilizer(labels_order)
                    self._figure_stabilizer_labels = list(labels_order)
                self._figure_stabilizer.update(raw_assignments)
                stable_count = len(self._figure_stabilizer.stable_assignments())
                try:
                    cv2.putText(  # type: ignore[attr-defined]
                        annotated,
                        f"Stabile Zuordnungen: {stable_count}",
                        (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX,  # type: ignore[attr-defined]
                        0.7,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,  # type: ignore[attr-defined]
                    )
                except Exception:
                    pass

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

        # DSHOW-Backend zuerst (Windows) – bessere Codec/Auflösungs-Kompatibilität
        capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)  # type: ignore[operator]
        if capture is None or not capture.isOpened():
            try:
                if capture is not None:
                    capture.release()
            except Exception:
                pass
            capture = cv2.VideoCapture(index)  # type: ignore[operator]

        # Auflösung und Codec setzen – analog zu open_camera() in recording.py
        try:
            from gaming_robot_arm.config import FRAME_HEIGHT, FRAME_RATE, FRAME_WIDTH
            if FRAME_WIDTH is not None and FRAME_HEIGHT is not None:
                if int(FRAME_WIDTH) >= 1280 or int(FRAME_HEIGHT) >= 720:
                    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))  # type: ignore[attr-defined]
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(FRAME_WIDTH))  # type: ignore[attr-defined]
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(FRAME_HEIGHT))  # type: ignore[attr-defined]
            if FRAME_RATE is not None:
                capture.set(cv2.CAP_PROP_FPS, float(FRAME_RATE))  # type: ignore[attr-defined]
        except Exception:
            pass
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
        self._camera_overlay_error_key = None
        self._figure_stabilizer = None
        self._figure_stabilizer_labels = []
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
        self._figure_stabilizer = None
        self._figure_stabilizer_labels = []

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

    def _make_muted_note(self, text: str, *, left_margin: int = 0) -> QLabel:
        note = QLabel(text)
        note.setWordWrap(True)
        note.setObjectName("MutedText")
        if left_margin > 0:
            note.setIndent(left_margin)
        return note

    def _add_form_note(self, form: QFormLayout, text: str) -> None:
        form.addRow(self._make_muted_note(text))

    def _add_line_edit(self, form: QFormLayout, key: str, label: str, note_text: str | None = None) -> None:
        widget = QLineEdit()
        widget.textChanged.connect(self._on_form_change)
        self._widgets[key] = widget
        form.addRow(label, widget)
        if note_text:
            self._add_form_note(form, note_text)

    def _add_combo(
        self,
        form: QFormLayout,
        key: str,
        label: str,
        values: list[str | tuple[str, str]],
        note_text: str | None = None,
    ) -> None:
        widget = QComboBox()
        widget.setEditable(False)
        for entry in values:
            if isinstance(entry, tuple):
                label_text, stored_value = entry
            else:
                label_text = stored_value = entry
            widget.addItem(label_text, stored_value)
        widget.currentTextChanged.connect(self._on_form_change)
        self._widgets[key] = widget
        form.addRow(label, widget)
        if note_text:
            self._add_form_note(form, note_text)

    def _add_check(self, layout, key: str, label: str, note_text: str | None = None) -> None:
        widget = QCheckBox(label)
        widget.toggled.connect(self._on_form_change)
        self._widgets[key] = widget
        if isinstance(layout, QFormLayout):
            layout.addRow(widget)
            if note_text:
                self._add_form_note(layout, note_text)
            return
        layout.addWidget(widget)
        if note_text:
            layout.addWidget(self._make_muted_note(note_text, left_margin=24))

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
            QCheckBox {
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
            QPushButton#SegmentOption {
                background: #eef2ff;
                color: #1e3a8a;
                border-color: #c7d2fe;
                padding: 9px 12px;
                font-weight: 600;
            }
            QPushButton#SegmentOption:hover {
                background: #e0e7ff;
                border-color: #a5b4fc;
            }
            QPushButton#SegmentOption:checked {
                background: #1d4ed8;
                color: white;
                border-color: #1d4ed8;
            }
            QPushButton#SegmentOption:disabled {
                background: #f1f5f9;
                color: #94a3b8;
                border-color: #e2e8f0;
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
        return load_launcher_settings(self.settings_file)

    def _save_settings(self, quiet: bool = False) -> bool:
        try:
            save_launcher_settings(self.settings_file, self._collect_settings_payload())
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
            self._normalize_uarm_controlled_players_from_legacy()
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
            idx = widget.findData(text)
            if idx < 0:
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
            "mill_record_game",
            "mill_flying",
            "mill_threefold_repetition",
            "mill_no_capture_draw",
            "mill_random_tiebreak",
            "mill_debug_vision",
            "mill_vision_preview",
            "mill_uarm_enable_ai_moves",
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
            data = widget.currentData()
            return str(data) if data is not None else widget.currentText()
        if isinstance(widget, QCheckBox):
            return "true" if widget.isChecked() else "false"
        raise TypeError(f"Nicht unterstützter Widget-Typ: {type(widget)!r}")

    def _widget_bool(self, widget: QWidget) -> bool:
        if not isinstance(widget, QCheckBox):
            raise TypeError(f"Bool-Feld erwartet QCheckBox, erhalten: {type(widget)!r}")
        return widget.isChecked()

    def _current_mode(self) -> str:
        return self._launch_mode if self._launch_mode in {"vision-loop", "play-mill"} else "play-mill"

    def _set_mode(self, mode: object) -> None:
        mode_str = str(mode) if mode is not None else "play-mill"
        if mode_str not in {"vision-loop", "play-mill"}:
            mode_str = "play-mill"
        self._launch_mode = mode_str

    def _set_combo_text_safely(self, key: str, value: str) -> None:
        widget = self._widgets.get(key)
        if not isinstance(widget, QComboBox):
            return
        idx = widget.findData(value)
        if idx < 0:
            idx = widget.findText(value)
        if idx < 0 or widget.currentIndex() == idx:
            return
        old = self._suppress_form_updates
        self._suppress_form_updates = True
        try:
            widget.setCurrentIndex(idx)
        finally:
            self._suppress_form_updates = old

    def _set_check_safely(self, key: str, value: bool) -> None:
        widget = self._widgets.get(key)
        if not isinstance(widget, QCheckBox):
            return
        new_value = bool(value)
        if widget.isChecked() == new_value:
            return
        old = self._suppress_form_updates
        self._suppress_form_updates = True
        try:
            widget.setChecked(new_value)
        finally:
            self._suppress_form_updates = old

    def _normalize_uarm_controlled_players_from_legacy(self) -> None:
        widget = self._widgets.get("mill_uarm_controlled_players")
        if not isinstance(widget, QComboBox):
            return
        raw_value = widget.currentText().strip().lower()
        if raw_value in {"white", "black", "both"}:
            return

        inferred = "black"
        if raw_value == "none":
            inferred = "black"
        else:
            move_both_widget = self._widgets.get("mill_uarm_move_both_players")
            ai_moves_widget = self._widgets.get("mill_uarm_enable_ai_moves")
            mode_widget = self._widgets.get("mill_mode")
            human_color_widget = self._widgets.get("mill_human_color")

            move_both = move_both_widget.isChecked() if isinstance(move_both_widget, QCheckBox) else False
            ai_moves = ai_moves_widget.isChecked() if isinstance(ai_moves_widget, QCheckBox) else False
            mode_value = mode_widget.currentText().strip() if isinstance(mode_widget, QComboBox) else "human-vs-ai"
            human_color = human_color_widget.currentText().strip() if isinstance(human_color_widget, QComboBox) else "W"

            if move_both:
                inferred = "both"
            elif ai_moves:
                if mode_value == "human-vs-ai":
                    inferred = "black" if human_color == "W" else "white"
                elif mode_value == "ai-vs-ai":
                    inferred = "both"

        self._set_combo_text_safely("mill_uarm_controlled_players", inferred)

    def _sync_legacy_uarm_fields(self) -> None:
        controlled_widget = self._widgets.get("mill_uarm_controlled_players")
        if not isinstance(controlled_widget, QComboBox):
            return

        controlled_value = controlled_widget.currentText().strip().lower()
        if controlled_value not in {"white", "black", "both"}:
            controlled_value = "black"

        self._set_check_safely("mill_uarm_enable_ai_moves", controlled_value in {"white", "black", "both"})
        self._set_check_safely("mill_uarm_move_both_players", controlled_value == "both")

    def _on_form_change(self, *_args: object) -> None:
        if self._suppress_form_updates:
            return
        self._refresh_context()
        self._refresh_command_preview()
        if self._runtime_output_only_active:
            self._set_runtime_output_only(True)
        if self._camera_preview_enabled():
            self._start_camera_preview()

    def _refresh_context(self) -> None:
        self._normalize_uarm_controlled_players_from_legacy()
        self._sync_player_role_buttons()
        self._apply_uarm_support_constraints()
        self._sync_legacy_uarm_fields()

    def _refresh_command_preview(self) -> None:
        if self._command_preview is None:
            return
        try:
            preview = shlex.join(self._build_command())
        except ValueError as exc:
            preview = f"Ungültige Einstellungen: {exc}"
        self._command_preview.setPlainText(preview)

    def _build_command(self) -> list[str]:
        payload = self._collect_settings_payload()
        payload["mode"] = self._current_mode()
        settings = LauncherSettings.from_payload(payload)
        return build_command(settings, python_executable=sys.executable, entry_script=self.entry_script)

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
        self._stop_requested = False
        self._stop_force_killed = False
        self._stop_request_id += 1
        self._append_log(f"\n[launcher] Starte Prozess:\n{shlex.join(cmd)}\n\n")

        if self._process is None:
            self._setup_process()
        assert self._process is not None

        error = start_qprocess(self._process, cmd=cmd, project_root=self.project_root)
        if error is not None:
            self._append_log(f"[launcher] Start fehlgeschlagen: {error}\n")
            QMessageBox.critical(self, "Start fehlgeschlagen", f"Prozess konnte nicht gestartet werden:\n{error}")
            self._set_status("Start fehlgeschlagen")
            self._sync_runtime_controls()
            return

        self._set_runtime_output_only(True)
        self._set_status(f"Läuft ({self._current_mode()})")
        self._sync_runtime_controls()

    def _stop_process(self) -> None:
        if not self._is_process_running() or self._process is None:
            self._set_status("Kein laufender Prozess")
            self._sync_runtime_controls()
            return
        self._stop_requested = True
        self._stop_force_killed = False
        self._stop_request_id += 1
        stop_id = self._stop_request_id
        self._append_log("[launcher] Prozess wird beendet ...\n")
        self._process.closeWriteChannel()
        self._set_status("Wird beendet ...")
        QTimer.singleShot(450, lambda: self._terminate_if_still_running(stop_id))
        QTimer.singleShot(2200, lambda: self._kill_if_still_running(stop_id))

    def _terminate_if_still_running(self, stop_id: int) -> None:
        if stop_id != self._stop_request_id or not self._stop_requested:
            return
        if not self._is_process_running() or self._process is None:
            return
        self._process.terminate()

    def _kill_if_still_running(self, stop_id: int) -> None:
        if stop_id != self._stop_request_id or not self._stop_requested:
            return
        if not self._is_process_running() or self._process is None:
            return
        self._append_log("[launcher] Prozess reagiert nicht, erzwinge Beenden.\n")
        self._stop_force_killed = True
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
        if self._stop_requested:
            if self._stop_force_killed:
                self._append_log("\n[launcher] Prozess gestoppt (erzwungen).\n")
                self._set_status("Gestoppt (erzwungen)")
            else:
                self._append_log("\n[launcher] Prozess gestoppt.\n")
                self._set_status("Gestoppt")
        else:
            self._append_log(f"\n[launcher] Prozess beendet mit Exit-Code {exit_code}\n")
            self._set_status(f"Beendet (Code {exit_code})")
        self._stop_requested = False
        self._stop_force_killed = False
        self._sync_runtime_controls()

    def _on_process_error(self, error) -> None:
        # Normalerweise folgt darauf ein finished-Signal. Wir loggen den Zustand trotzdem,
        # damit Startprobleme im UI sichtbar sind.
        if self._process is None:
            return
        if self._stop_requested and error == QProcess.ProcessError.Crashed:
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
        if self._quick_start_button is not None:
            self._quick_start_button.setEnabled(not running)
        if self._stop_button is not None:
            self._stop_button.setEnabled(running)
        if self._runtime_back_button is not None:
            self._runtime_back_button.setEnabled(not running)
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
            self._stop_requested = True
            self._stop_force_killed = False
            self._stop_request_id += 1
            self._process.closeWriteChannel()
            self._process.terminate()
            if not self._process.waitForFinished(2000):
                self._append_log("[launcher] Erzwinge Beenden beim Schließen ...\n")
                self._stop_force_killed = True
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
