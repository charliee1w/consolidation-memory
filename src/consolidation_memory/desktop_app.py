"""Native desktop UI for consolidation-memory with system tray support.

Requires: pip install consolidation-memory[desktop]
"""

from __future__ import annotations

import sys
from typing import Any, Callable

from consolidation_memory.desktop_backend import DesktopBackend

_PYSIDE_AVAILABLE = False

try:
    from PySide6.QtCore import Qt, QThread, Signal
    from PySide6.QtGui import QAction, QColor, QFont, QIcon, QPainter, QPixmap
    from PySide6.QtWidgets import (
        QApplication,
        QComboBox,
        QFormLayout,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QStatusBar,
        QSystemTrayIcon,
        QTabWidget,
        QTableWidget,
        QTableWidgetItem,
        QTextBrowser,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )

    _PYSIDE_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised when desktop extra missing
    pass

APP_STYLESHEET = """
QMainWindow, QWidget {
    background-color: #0f1419;
    color: #e8eef5;
    font-family: "Segoe UI", system-ui, sans-serif;
    font-size: 13px;
}
QTabWidget::pane {
    border: 1px solid #2f3d4f;
    border-radius: 8px;
    background: #1a222d;
    top: -1px;
}
QTabBar::tab {
    background: #1a222d;
    color: #8fa3b8;
    border: 1px solid #2f3d4f;
    border-bottom: none;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    padding: 8px 16px;
    margin-right: 2px;
}
QTabBar::tab:selected {
    background: #232d3b;
    color: #e8eef5;
}
QLineEdit, QTextEdit, QComboBox, QTextBrowser {
    background: #232d3b;
    color: #e8eef5;
    border: 1px solid #2f3d4f;
    border-radius: 8px;
    padding: 8px;
    selection-background-color: #3d9cf5;
}
QPushButton {
    background: #2a6fad;
    color: #e8eef5;
    border: none;
    border-radius: 8px;
    padding: 8px 14px;
    font-weight: 600;
}
QPushButton:hover { background: #3d9cf5; }
QPushButton:disabled { background: #2f3d4f; color: #8fa3b8; }
QPushButton#secondary {
    background: #232d3b;
    border: 1px solid #2f3d4f;
    font-weight: 500;
}
QTableWidget {
    background: #1a222d;
    alternate-background-color: #232d3b;
    gridline-color: #2f3d4f;
    border: 1px solid #2f3d4f;
    border-radius: 8px;
}
QHeaderView::section {
    background: #232d3b;
    color: #8fa3b8;
    border: none;
    border-bottom: 1px solid #2f3d4f;
    padding: 6px;
}
QStatusBar {
    background: #1a222d;
    color: #8fa3b8;
    border-top: 1px solid #2f3d4f;
}
QGroupBox {
    border: 1px solid #2f3d4f;
    border-radius: 8px;
    margin-top: 10px;
    padding-top: 12px;
    color: #8fa3b8;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
}
"""


def build_app_icon() -> Any:
    """Build a multi-resolution app icon without external asset files."""
    if not _PYSIDE_AVAILABLE:
        raise ImportError("PySide6 is required for the desktop app icon")

    icon = QIcon()
    for size in (16, 24, 32, 48, 64, 128, 256):
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        margin = max(1, size // 10)
        painter.setBrush(QColor("#3d9cf5"))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(margin, margin, size - margin * 2, size - margin * 2, size // 5, size // 5)

        painter.setPen(QColor("#0f1419"))
        font = QFont("Segoe UI", max(8, size // 3))
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "M")
        painter.end()
        icon.addPixmap(pixmap)
    return icon


if _PYSIDE_AVAILABLE:

    class _Worker(QThread):
        finished_ok = Signal(object)
        failed = Signal(str)

        def __init__(self, fn: Callable[..., object], *args: object, **kwargs: object) -> None:
            super().__init__()
            self._fn = fn
            self._args = args
            self._kwargs = kwargs

        def run(self) -> None:
            try:
                result = self._fn(*self._args, **self._kwargs)
            except Exception as exc:  # noqa: BLE001 - surface to UI
                self.failed.emit(str(exc))
                return
            self.finished_ok.emit(result)


    class MainWindow(QMainWindow):
        def __init__(self, backend: DesktopBackend, *, tray_enabled: bool = True) -> None:
            super().__init__()
            self._backend = backend
            self._tray_enabled = tray_enabled
            self._worker: _Worker | None = None
            self._icon = build_app_icon()

            self.setWindowTitle("consolidation-memory")
            self.setWindowIcon(self._icon)
            self.resize(920, 680)
            self.setStyleSheet(APP_STYLESHEET)

            central = QWidget()
            self.setCentralWidget(central)
            root = QVBoxLayout(central)
            root.setContentsMargins(16, 16, 16, 12)
            root.setSpacing(12)

            header = QHBoxLayout()
            title_block = QVBoxLayout()
            title = QLabel("consolidation-memory")
            title.setStyleSheet("font-size: 20px; font-weight: 650; color: #e8eef5;")
            subtitle = QLabel("Remember fixes and notes. Search them later in plain language.")
            subtitle.setStyleSheet("color: #8fa3b8;")
            title_block.addWidget(title)
            title_block.addWidget(subtitle)
            header.addLayout(title_block, stretch=1)

            self._health_label = QLabel("Loading…")
            self._health_label.setStyleSheet(
                "background: #1a222d; border: 1px solid #2f3d4f; border-radius: 999px; padding: 8px 12px;"
            )
            header.addWidget(self._health_label, alignment=Qt.AlignmentFlag.AlignTop)
            root.addLayout(header)

            self._tabs = QTabWidget()
            self._tabs.addTab(self._build_ask_tab(), "Ask")
            self._tabs.addTab(self._build_remember_tab(), "Remember")
            self._tabs.addTab(self._build_browse_tab(), "Browse")
            self._tabs.addTab(self._build_health_tab(), "Health")
            self._tabs.addTab(self._build_hygiene_tab(), "Hygiene")
            root.addWidget(self._tabs, stretch=1)

            actions = QHBoxLayout()
            self._consolidate_btn = QPushButton("Run consolidation")
            self._consolidate_btn.clicked.connect(self._on_consolidate)
            actions.addWidget(self._consolidate_btn)
            actions.addStretch(1)
            refresh_btn = QPushButton("Refresh")
            refresh_btn.setObjectName("secondary")
            refresh_btn.clicked.connect(self.refresh_overview)
            actions.addWidget(refresh_btn)
            root.addLayout(actions)

            self._status = QStatusBar()
            self.setStatusBar(self._status)

            if tray_enabled:
                self._tray = QSystemTrayIcon(self._icon, self)
                self._tray.setToolTip("consolidation-memory")
                tray_menu = self._tray.contextMenu()
                if tray_menu is None:
                    from PySide6.QtWidgets import QMenu

                    tray_menu = QMenu(self)
                    self._tray.setContextMenu(tray_menu)
                open_action = QAction("Open", self)
                open_action.triggered.connect(self.show_and_raise)
                consolidate_action = QAction("Run consolidation", self)
                consolidate_action.triggered.connect(self._on_consolidate)
                quit_action = QAction("Quit", self)
                quit_action.triggered.connect(self._quit_app)
                tray_menu.addAction(open_action)
                tray_menu.addAction(consolidate_action)
                tray_menu.addSeparator()
                tray_menu.addAction(quit_action)
                self._tray.activated.connect(self._on_tray_activated)
                self._tray.show()
            else:
                self._tray = None

            self.refresh_overview()
            self._load_episodes()

        def _build_ask_tab(self) -> QWidget:
            page = QWidget()
            layout = QVBoxLayout(page)
            row = QHBoxLayout()
            self._ask_input = QLineEdit()
            self._ask_input.setPlaceholderText("What do you want to recall?")
            self._ask_input.returnPressed.connect(self._on_ask)
            ask_btn = QPushButton("Search")
            ask_btn.clicked.connect(self._on_ask)
            row.addWidget(self._ask_input, stretch=1)
            row.addWidget(ask_btn)
            layout.addLayout(row)
            self._ask_results = QTextBrowser()
            self._ask_results.setOpenExternalLinks(False)
            layout.addWidget(self._ask_results, stretch=1)
            return page

        def _build_remember_tab(self) -> QWidget:
            page = QWidget()
            layout = QVBoxLayout(page)
            form = QFormLayout()
            self._remember_kind = QComboBox()
            self._remember_kind.addItems(["note", "fact", "fix", "preference"])
            self._remember_tags = QLineEdit()
            self._remember_tags.setPlaceholderText("optional, comma-separated")
            form.addRow("Kind", self._remember_kind)
            form.addRow("Tags", self._remember_tags)
            layout.addLayout(form)
            self._remember_input = QTextEdit()
            self._remember_input.setPlaceholderText("What should be remembered?")
            layout.addWidget(self._remember_input, stretch=1)
            save_btn = QPushButton("Save to memory")
            save_btn.clicked.connect(self._on_remember)
            layout.addWidget(save_btn, alignment=Qt.AlignmentFlag.AlignRight)
            return page

        def _build_browse_tab(self) -> QWidget:
            page = QWidget()
            layout = QVBoxLayout(page)
            self._episodes_table = QTableWidget(0, 5)
            self._episodes_table.setHorizontalHeaderLabels(
                ["Content", "Type", "Tags", "Created", "Status"],
            )
            self._episodes_table.horizontalHeader().setSectionResizeMode(
                0, QHeaderView.ResizeMode.Stretch,
            )
            self._episodes_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
            self._episodes_table.setAlternatingRowColors(True)
            layout.addWidget(self._episodes_table, stretch=1)
            row = QHBoxLayout()
            forget_btn = QPushButton("Forget selected")
            forget_btn.setObjectName("secondary")
            forget_btn.clicked.connect(self._on_forget_selected)
            row.addWidget(forget_btn)
            row.addStretch(1)
            reload_btn = QPushButton("Reload")
            reload_btn.setObjectName("secondary")
            reload_btn.clicked.connect(self._load_episodes)
            row.addWidget(reload_btn)
            layout.addLayout(row)
            return page

        def _build_health_tab(self) -> QWidget:
            page = QWidget()
            layout = QVBoxLayout(page)
            row = QHBoxLayout()
            refresh_btn = QPushButton("Refresh status")
            refresh_btn.clicked.connect(self._load_health)
            row.addWidget(refresh_btn)
            row.addStretch(1)
            layout.addLayout(row)
            self._health_details = QTextBrowser()
            layout.addWidget(self._health_details, stretch=1)
            return page

        def _build_hygiene_tab(self) -> QWidget:
            page = QWidget()
            layout = QVBoxLayout(page)
            actions = QHBoxLayout()
            scan_btn = QPushButton("Scan corpus")
            scan_btn.clicked.connect(self._on_hygiene_scan)
            preview_btn = QPushButton("Preview cleanup")
            preview_btn.setObjectName("secondary")
            preview_btn.clicked.connect(self._on_hygiene_preview)
            apply_btn = QPushButton("Apply cleanup")
            apply_btn.clicked.connect(self._on_hygiene_apply)
            actions.addWidget(scan_btn)
            actions.addWidget(preview_btn)
            actions.addWidget(apply_btn)
            actions.addStretch(1)
            layout.addLayout(actions)
            from PySide6.QtWidgets import QCheckBox

            self._hygiene_expire_orphans = QCheckBox("Expire orphaned claims")
            layout.addWidget(self._hygiene_expire_orphans)
            self._hygiene_results = QTextBrowser()
            layout.addWidget(self._hygiene_results, stretch=1)
            return page

        def show_and_raise(self) -> None:
            self.show()
            self.raise_()
            self.activateWindow()

        def refresh_overview(self) -> None:
            overview = self._backend.overview()
            health = str(overview.get("health") or "ok")
            note = str(overview.get("health_note") or "")
            project = str(overview.get("project") or "")
            stats = overview.get("stats")
            total = 0
            if isinstance(stats, dict):
                total = int(stats.get("total_episodes") or 0)
            dot = "#3ecf8e" if health == "ok" else "#f5b83d"
            self._health_label.setText(f"● {note}  ·  {project}  ·  {total} episodes")
            self._health_label.setStyleSheet(
                "background: #1a222d; border: 1px solid #2f3d4f; border-radius: 999px; "
                f"padding: 8px 12px; color: {dot};"
            )
            self._status.showMessage(note)

        def _load_episodes(self) -> None:
            episodes = self._backend.recent_episodes(limit=80)
            self._episodes_table.setRowCount(len(episodes))
            status_labels = {0: "pending", 1: "consolidated", 2: "pruned"}
            for row, episode in enumerate(episodes):
                preview = str(episode.get("content_preview") or "")
                content_type = str(episode.get("content_type") or "")
                tags = episode.get("tags") or []
                tags_text = ", ".join(str(tag) for tag in tags) if isinstance(tags, list) else ""
                created = str(episode.get("created_at") or "")[:19].replace("T", " ")
                status = status_labels.get(int(episode.get("consolidated") or 0), "pending")
                values = [preview, content_type, tags_text, created, status]
                for col, value in enumerate(values):
                    item = QTableWidgetItem(value)
                    if col == 0:
                        item.setData(Qt.ItemDataRole.UserRole, episode.get("id"))
                    self._episodes_table.setItem(row, col, item)

        def _run_async(
            self,
            fn: Callable[..., object],
            *,
            busy_message: str,
            on_success: Callable[[object], None],
            fn_args: tuple[object, ...] = (),
            fn_kwargs: dict[str, object] | None = None,
        ) -> None:
            if self._worker is not None and self._worker.isRunning():
                self._status.showMessage("Already working…")
                return
            self._consolidate_btn.setEnabled(False)
            self._status.showMessage(busy_message)
            worker = _Worker(fn, *fn_args, **(fn_kwargs or {}))
            self._worker = worker

            def _done(result: object) -> None:
                self._consolidate_btn.setEnabled(True)
                on_success(result)

            def _fail(message: str) -> None:
                self._consolidate_btn.setEnabled(True)
                self._status.showMessage(f"Error: {message}")
                QMessageBox.warning(self, "consolidation-memory", message)

            worker.finished_ok.connect(_done)
            worker.failed.connect(_fail)
            worker.finished.connect(worker.deleteLater)
            worker.start()

        def _on_ask(self) -> None:
            query = self._ask_input.text().strip()
            if not query:
                return
            self._run_async(
                self._backend.ask,
                busy_message="Searching memory…",
                on_success=self._show_ask_results,
                fn_kwargs={"query": query},
            )

        def _show_ask_results(self, result: object) -> None:
            if not isinstance(result, dict):
                self._ask_results.setPlainText("Unexpected response.")
                return
            lines: list[str] = []
            warnings = result.get("warnings") or []
            if warnings:
                lines.append("<p style='color:#f5b83d'><b>Warnings</b></p><ul>")
                for warning in warnings:
                    lines.append(f"<li>{warning}</li>")
                lines.append("</ul>")

            def _section(title: str, rows: object, formatter: Callable[[dict], str]) -> None:
                if not isinstance(rows, list) or not rows:
                    return
                lines.append(f"<h3 style='color:#3d9cf5'>{title}</h3><ul>")
                for row in rows:
                    if isinstance(row, dict):
                        lines.append(f"<li>{formatter(row)}</li>")
                lines.append("</ul>")

            _section(
                "Episodes",
                result.get("episodes"),
                lambda row: (
                    f"<b>{row.get('kind')}</b> — {row.get('preview')} "
                    f"<span style='color:#8fa3b8'>({row.get('score')})</span>"
                ),
            )
            _section(
                "Knowledge",
                result.get("knowledge"),
                lambda row: f"<b>{row.get('title')}</b> — {row.get('summary')}",
            )
            _section(
                "Records",
                result.get("records"),
                lambda row: f"<b>{row.get('type')}</b> — {row.get('text')}",
            )
            _section(
                "Claims",
                result.get("claims"),
                lambda row: (
                    f"<b>{row.get('type')}</b> [{row.get('status')}] — {row.get('text')}"
                ),
            )
            if len(lines) == 0:
                lines.append("<p style='color:#8fa3b8'>No matches found.</p>")
            self._ask_results.setHtml("".join(lines))
            self._status.showMessage("Search complete.")

        def _on_remember(self) -> None:
            content = self._remember_input.toPlainText().strip()
            if not content:
                return
            tags_raw = self._remember_tags.text().strip()
            tags = [tag.strip() for tag in tags_raw.split(",") if tag.strip()] if tags_raw else None

            def _done(result: object) -> None:
                self._remember_input.clear()
                self._status.showMessage("Saved to memory.")
                self.refresh_overview()
                self._load_episodes()
                if isinstance(result, dict) and result.get("episode_id"):
                    self._tabs.setCurrentIndex(2)

            self._run_async(
                self._backend.remember,
                busy_message="Saving…",
                on_success=_done,
                fn_kwargs={
                    "content": content,
                    "kind": self._remember_kind.currentText(),
                    "tags": tags,
                },
            )

        def _on_consolidate(self) -> None:
            def _done(result: object) -> None:
                if isinstance(result, dict) and result.get("status") == "already_running":
                    self._status.showMessage("Consolidation already running.")
                    return
                self._status.showMessage("Consolidation finished.")
                self.refresh_overview()
                self._load_episodes()

            self._run_async(
                self._backend.consolidate,
                busy_message="Running consolidation…",
                on_success=_done,
            )

        def _load_health(self) -> None:
            def _done(result: object) -> None:
                if not isinstance(result, dict):
                    self._health_details.setPlainText("Unexpected status response.")
                    return
                import json

                self._health_details.setPlainText(json.dumps(result, indent=2, default=str))
                self._status.showMessage("Health status refreshed.")

            self._run_async(
                self._backend.status,
                busy_message="Loading health status…",
                on_success=_done,
            )

        def _format_hygiene_report(self, result: object) -> str:
            if not isinstance(result, dict):
                return "Unexpected hygiene response."
            import json

            return json.dumps(result, indent=2, default=str)

        def _on_hygiene_scan(self) -> None:
            def _done(result: object) -> None:
                self._hygiene_results.setPlainText(self._format_hygiene_report(result))
                self._status.showMessage("Hygiene scan complete.")

            self._run_async(
                self._backend.hygiene_scan,
                busy_message="Scanning corpus…",
                on_success=_done,
            )

        def _on_hygiene_preview(self) -> None:
            self._run_hygiene_apply(dry_run=True)

        def _on_hygiene_apply(self) -> None:
            self._run_hygiene_apply(dry_run=False)

        def _run_hygiene_apply(self, *, dry_run: bool) -> None:
            expire_orphans = self._hygiene_expire_orphans.isChecked()

            def _done(result: object) -> None:
                self._hygiene_results.setPlainText(self._format_hygiene_report(result))
                label = "Preview complete." if dry_run else "Hygiene apply complete."
                self._status.showMessage(label)
                if not dry_run:
                    self.refresh_overview()
                    self._load_episodes()

            self._run_async(
                self._backend.hygiene_apply,
                busy_message="Previewing cleanup…" if dry_run else "Applying cleanup…",
                on_success=_done,
                fn_kwargs={
                    "use_recommended": True,
                    "expire_orphans": expire_orphans,
                    "dry_run": dry_run,
                },
            )

        def _on_forget_selected(self) -> None:
            row = self._episodes_table.currentRow()
            if row < 0:
                return
            item = self._episodes_table.item(row, 0)
            if item is None:
                return
            episode_id = item.data(Qt.ItemDataRole.UserRole)
            if not episode_id:
                return

            def _done(_result: object) -> None:
                self._status.showMessage("Episode forgotten.")
                self.refresh_overview()
                self._load_episodes()

            self._run_async(
                self._backend.forget,
                busy_message="Forgetting episode…",
                on_success=_done,
                fn_kwargs={"episode_id": str(episode_id)},
            )

        def _on_tray_activated(self, reason: Any) -> None:
            if reason == QSystemTrayIcon.ActivationReason.DoubleClick:
                self.show_and_raise()

        def closeEvent(self, event: Any) -> None:
            if self._tray_enabled and self._tray is not None and self._tray.isVisible():
                event.ignore()
                self.hide()
                self._tray.showMessage(
                    "consolidation-memory",
                    "Still running in the taskbar tray. Open from the icon.",
                    QSystemTrayIcon.MessageIcon.Information,
                    3000,
                )
                return
            super().closeEvent(event)

        def _quit_app(self) -> None:
            if self._tray is not None:
                self._tray.hide()
            QApplication.instance().quit()


def run_desktop_app(*, tray: bool = True) -> None:
    """Launch the native desktop application."""
    if not _PYSIDE_AVAILABLE:
        raise ImportError(
            "Desktop app requires PySide6. Install with: "
            "pip install consolidation-memory[desktop]"
        )

    app = QApplication(sys.argv)
    app.setApplicationName("consolidation-memory")
    app.setOrganizationName("consolidation-memory")
    app.setWindowIcon(build_app_icon())
    app.setQuitOnLastWindowClosed(not tray)

    window = MainWindow(DesktopBackend(), tray_enabled=tray)
    window.show_and_raise()
    sys.exit(app.exec())