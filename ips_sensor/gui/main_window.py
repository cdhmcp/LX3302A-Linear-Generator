"""Main desktop window for configuring and generating sensor footprints."""

from __future__ import annotations

from dataclasses import asdict
import math
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QObject, QRunnable, QThreadPool, QTimer, Qt, Signal, Slot
from PySide6.QtGui import QDoubleValidator, QIntValidator
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStatusBar,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..diagnostics import Diagnostic, DiagnosticSeverity, ValidationReport
from ..engine import AnalysisResult, GENERATORS, GenerationBlocked, LinearSensorDefinition
from ..models import GenerationRequest, LinearSensorConfig, OutputConfig, default_request
from ..parameters import PARAMETER_GROUPS, PARAMETERS, PARAMETERS_BY_KEY, ParameterKind, ParameterSpec
from ..project import ProjectLoadError, load_project, save_project
from .preview import FootprintPreview


MM_PER_MIL = 0.0254


class NoWheelComboBox(QComboBox):
    """A selector that changes only through a deliberate click or keyboard action."""

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt callback name
        event.ignore()


class UnitNumberEdit(QLineEdit):
    """Keyboard-only numeric editor that stores values canonically in millimeters."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        validator = QDoubleValidator(-1_000_000_000.0, 1_000_000_000.0, 8, self)
        validator.setNotation(QDoubleValidator.Notation.StandardNotation)
        self.setValidator(validator)
        self.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._unit = "mm"
        self.editingFinished.connect(self._normalize)

    @property
    def display_unit(self) -> str:
        return self._unit

    def set_mm_value(self, value: float) -> None:
        factor = MM_PER_MIL if self._unit == "mil" else 1.0
        self.setText(self._format(value / factor))

    def value_mm(self) -> float:
        try:
            value = float(self.text().strip())
        except ValueError:
            return float("nan")
        factor = MM_PER_MIL if self._unit == "mil" else 1.0
        return value * factor

    def set_display_unit(self, unit: str) -> None:
        if unit == self._unit:
            return
        value_mm = self.value_mm()
        self._unit = unit
        if math.isfinite(value_mm):
            self.set_mm_value(value_mm)

    def _normalize(self) -> None:
        value_mm = self.value_mm()
        if math.isfinite(value_mm):
            self.set_mm_value(value_mm)

    @staticmethod
    def _format(value: float) -> str:
        return f"{value:.8f}".rstrip("0").rstrip(".") or "0"


class IntegerEntry(QLineEdit):
    """Keyboard-only integer editor for non-selector integer settings."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setValidator(QIntValidator(-1_000_000, 1_000_000, self))
        self.setAlignment(Qt.AlignmentFlag.AlignRight)


class _WorkerSignals(QObject):
    completed = Signal(int, object, object)


class _AnalysisWorker(QRunnable):
    """Run the potentially slow legacy geometry checks outside the GUI thread."""

    def __init__(
        self,
        revision: int,
        definition: LinearSensorDefinition,
        request: GenerationRequest,
    ) -> None:
        super().__init__()
        self.revision = revision
        self.definition = definition
        self.request = request
        self.signals = _WorkerSignals()

    @Slot()
    def run(self) -> None:
        try:
            result: AnalysisResult | None = self.definition.analyze(self.request)
            error: Exception | None = None
        except Exception as caught:  # defensive boundary for a desktop tool
            result = None
            error = caught
        self.signals.completed.emit(self.revision, result, error)


class SensorMainWindow(QMainWindow):
    """Interactive editor with help, diagnostics, preview, and safe export."""

    def __init__(self, *, start_validation: bool = True) -> None:
        super().__init__()
        self.setWindowTitle("Inductive Sensor Footprint Generator")
        self.resize(1480, 920)

        self.definition = GENERATORS["linear-lx3302a"]
        assert isinstance(self.definition, LinearSensorDefinition)
        self.request = default_request()
        self._controls: dict[str, QWidget] = {}
        self._unit_controls: dict[str, NoWheelComboBox] = {}
        self._control_labels: dict[str, QLabel] = {}
        self._advanced_rows: list[tuple[QWidget, QWidget, QGroupBox]] = []
        self._groups: dict[str, QGroupBox] = {}
        self._latest_analysis: AnalysisResult | None = None
        self._analysis_revision = -1
        self._revision = 0
        self._worker_active = False
        self._pending_validation = False
        self._generate_after_validation = False
        self._thread_pool = QThreadPool(self)
        self._thread_pool.setMaxThreadCount(1)

        self._validation_timer = QTimer(self)
        self._validation_timer.setSingleShot(True)
        self._validation_timer.timeout.connect(self._start_validation)

        self._build_ui()
        self._populate_controls()
        if start_validation:
            self._schedule_validation(delay_ms=0)

    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.addWidget(self._build_editor())
        splitter.addWidget(self._build_results())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([470, 1010])
        self.setCentralWidget(splitter)
        self.setStatusBar(QStatusBar(self))
        self.statusBar().showMessage("Ready")

    def _build_editor(self) -> QWidget:
        outer = QWidget(self)
        outer_layout = QVBoxLayout(outer)
        outer_layout.setContentsMargins(8, 8, 8, 8)

        form_header = QHBoxLayout()
        form_header.addWidget(QLabel("Configuration"))
        form_header.addStretch()
        self._advanced_toggle = QCheckBox("Show advanced")
        self._advanced_toggle.toggled.connect(self._toggle_advanced)
        form_header.addWidget(self._advanced_toggle)
        outer_layout.addLayout(form_header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_body = QWidget()
        self._form_layout = QVBoxLayout(scroll_body)
        self._form_layout.setContentsMargins(4, 4, 4, 4)
        self._form_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        for group_name in PARAMETER_GROUPS:
            box = QGroupBox(group_name)
            form = QFormLayout(box)
            form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
            self._groups[group_name] = box
            self._form_layout.addWidget(box)
        for spec in PARAMETERS:
            self._add_parameter(spec)
        scroll.setWidget(scroll_body)
        outer_layout.addWidget(scroll, 1)

        button_grid = QHBoxLayout()
        validate_button = QPushButton("Validate")
        validate_button.clicked.connect(self._validate_now)
        button_grid.addWidget(validate_button)
        generate_button = QPushButton("Generate")
        generate_button.setDefault(True)
        generate_button.clicked.connect(self._generate)
        button_grid.addWidget(generate_button)
        outer_layout.addLayout(button_grid)

        project_buttons = QHBoxLayout()
        load_button = QPushButton("Load project")
        load_button.clicked.connect(self._load_project)
        project_buttons.addWidget(load_button)
        save_button = QPushButton("Save project")
        save_button.clicked.connect(self._save_project)
        project_buttons.addWidget(save_button)
        outer_layout.addLayout(project_buttons)
        reset_button = QPushButton("Reset to defaults")
        reset_button.clicked.connect(self._reset_defaults)
        outer_layout.addWidget(reset_button)
        return outer

    def _build_results(self) -> QWidget:
        panel = QWidget(self)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        preview_header = QHBoxLayout()
        preview_header.addWidget(QLabel("2D footprint preview"))
        preview_header.addStretch()
        self._layer_button = QToolButton()
        self._layer_button.setText("Layers")
        self._layer_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self._layer_button.setMenu(QMenu(self._layer_button))
        self._layer_button.setEnabled(False)
        preview_header.addWidget(self._layer_button)
        layout.addLayout(preview_header)
        self.preview = FootprintPreview(self)
        self.preview.setFrameShape(QFrame.Shape.StyledPanel)
        layout.addWidget(self.preview, 3)

        layout.addWidget(QLabel("Diagnostics"))
        self.diagnostics = QTreeWidget(self)
        self.diagnostics.setHeaderLabels(["Severity", "Setting", "Message"])
        self.diagnostics.setRootIsDecorated(False)
        self.diagnostics.setAlternatingRowColors(True)
        self.diagnostics.itemActivated.connect(self._focus_diagnostic)
        self.diagnostics.itemClicked.connect(self._focus_diagnostic)
        layout.addWidget(self.diagnostics, 1)
        return panel

    def _add_parameter(self, spec: ParameterSpec) -> None:
        group = self._groups[spec.group]
        form = group.layout()
        assert isinstance(form, QFormLayout)
        label = QLabel(spec.label)
        label.setToolTip(spec.help_text)
        control, wrapper = self._create_control(spec)
        help_button = QToolButton()
        help_button.setText("?")
        help_button.setToolTip(f"Help for {spec.label}")
        help_button.setAccessibleName(f"Help for {spec.label}")
        help_button.clicked.connect(lambda _checked=False, item=spec: self._show_help(item))
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.addWidget(wrapper, 1)
        row_layout.addWidget(help_button)
        form.addRow(label, row)
        self._controls[spec.key] = control
        self._control_labels[spec.key] = label
        if not spec.basic:
            self._advanced_rows.append((label, row, group))
            label.setVisible(False)
            row.setVisible(False)

    def _create_control(self, spec: ParameterSpec) -> tuple[QWidget, QWidget]:
        changed: Callable[..., None] = self._on_field_edited
        if spec.kind is ParameterKind.FLOAT:
            if spec.unit:
                control = UnitNumberEdit()
                control.editingFinished.connect(changed)
                unit_selector = NoWheelComboBox()
                unit_selector.addItems(("mm", "mil"))
                # Leave enough room for "mil" plus the combo arrow at common
                # Windows scaling factors; the old 58 px width clipped "mm".
                unit_selector.setFixedWidth(76)
                unit_selector.setToolTip("Select millimeters or mils for this value")
                unit_selector.currentTextChanged.connect(control.set_display_unit)
                unit_selector.currentTextChanged.connect(changed)
                wrapper = QWidget()
                wrapper_layout = QHBoxLayout(wrapper)
                wrapper_layout.setContentsMargins(0, 0, 0, 0)
                wrapper_layout.addWidget(control, 1)
                wrapper_layout.addWidget(unit_selector)
                self._unit_controls[spec.key] = unit_selector
                return control, wrapper
            control = QLineEdit()
            validator = QDoubleValidator(-1_000_000_000.0, 1_000_000_000.0, 8, control)
            validator.setNotation(QDoubleValidator.Notation.StandardNotation)
            control.setValidator(validator)
            control.setAlignment(Qt.AlignmentFlag.AlignRight)
            control.editingFinished.connect(changed)
            return control, control
        if spec.kind is ParameterKind.INTEGER:
            if spec.key in {"number_of_primary_turns", "number_of_secondary_turns"}:
                control = NoWheelComboBox()
                control.addItems(("1", "2", "3", "4", "5"))
                control.currentTextChanged.connect(changed)
                return control, control
            control = IntegerEntry()
            control.editingFinished.connect(changed)
            return control, control
        if spec.kind is ParameterKind.BOOLEAN:
            control = QCheckBox()
            control.toggled.connect(changed)
            return control, control
        if spec.kind is ParameterKind.CHOICE:
            control = NoWheelComboBox()
            control.addItems(spec.choices)
            control.currentTextChanged.connect(changed)
            return control, control
        control = QLineEdit()
        control.editingFinished.connect(changed)
        if spec.kind is ParameterKind.DIRECTORY:
            browse = QPushButton("Browse…")
            browse.clicked.connect(lambda _checked=False, key=spec.key: self._browse_directory(key))
            wrapper = QWidget()
            wrapper_layout = QHBoxLayout(wrapper)
            wrapper_layout.setContentsMargins(0, 0, 0, 0)
            wrapper_layout.addWidget(control, 1)
            wrapper_layout.addWidget(browse)
            return control, wrapper
        return control, control

    def _populate_controls(self) -> None:
        values = {**asdict(self.request.config), "output_dir": self.request.output.output_dir}
        for spec in PARAMETERS:
            control = self._controls[spec.key]
            value = values[spec.key]
            control.blockSignals(True)
            if isinstance(control, UnitNumberEdit):
                control.set_mm_value(float(value))
            elif isinstance(control, IntegerEntry):
                control.setText(str(int(value)))
            elif isinstance(control, QCheckBox):
                control.setChecked(bool(value))
            elif isinstance(control, QComboBox):
                control.setCurrentText(str(value))
            elif isinstance(control, QLineEdit):
                control.setText(str(value))
            control.blockSignals(False)

    def _request_from_controls(self) -> GenerationRequest:
        config_values: dict[str, object] = {}
        output_values: dict[str, object] = {}
        for spec in PARAMETERS:
            control = self._controls[spec.key]
            if isinstance(control, UnitNumberEdit):
                value: object = control.value_mm()
            elif isinstance(control, IntegerEntry):
                try:
                    value = int(control.text().strip())
                except ValueError:
                    value = control.text().strip()
            elif isinstance(control, QCheckBox):
                value = control.isChecked()
            elif isinstance(control, QComboBox):
                value = (
                    int(control.currentText())
                    if spec.kind is ParameterKind.INTEGER
                    else control.currentText()
                )
            elif spec.kind is ParameterKind.FLOAT:
                assert isinstance(control, QLineEdit)
                try:
                    value = float(control.text().strip())
                except ValueError:
                    value = control.text().strip()
            else:
                assert isinstance(control, QLineEdit)
                value = control.text().strip()
            if spec.key == "output_dir":
                output_values[spec.key] = value
            else:
                config_values[spec.key] = value
        return GenerationRequest(
            config=LinearSensorConfig.from_mapping(config_values),
            output=OutputConfig.from_mapping(output_values),
            sensor_type=self.request.sensor_type,
            exporter_id=self.request.exporter_id,
        )

    def _on_field_edited(self, *_args) -> None:
        self._revision += 1
        self._schedule_validation()

    def _schedule_validation(self, *, delay_ms: int = 650) -> None:
        self._validation_timer.start(delay_ms)

    def _start_validation(self) -> None:
        if self._worker_active:
            self._pending_validation = True
            return
        request = self._request_from_controls()
        self._worker_active = True
        revision = self._revision
        self.statusBar().showMessage("Checking geometry and clearance rules…")
        worker = _AnalysisWorker(revision, self.definition, request)
        worker.signals.completed.connect(self._validation_finished)
        self._thread_pool.start(worker)

    def _validation_finished(
        self,
        revision: int,
        result: AnalysisResult | None,
        error: Exception | None,
    ) -> None:
        self._worker_active = False
        if revision == self._revision:
            if error is not None:
                report = ValidationReport((
                    Diagnostic(
                        "UNEXPECTED_FAILURE",
                        DiagnosticSeverity.BLOCKING,
                        str(error) or type(error).__name__,
                    ),
                ))
                result = AnalysisResult(report, None)
            assert result is not None
            self._latest_analysis = result
            self._analysis_revision = revision
            self._show_analysis(result)
            if self._generate_after_validation:
                self._generate_after_validation = False
                self._complete_generation(result)

        if self._pending_validation or revision != self._revision:
            self._pending_validation = False
            self._schedule_validation(delay_ms=0)

    def _show_analysis(self, analysis: AnalysisResult) -> None:
        self.diagnostics.clear()
        for item in analysis.report.diagnostics:
            setting = PARAMETERS_BY_KEY[item.field].label if item.field in PARAMETERS_BY_KEY else "General"
            row = QTreeWidgetItem([item.severity.value.title(), setting, item.message])
            row.setData(0, Qt.ItemDataRole.UserRole, item.field)
            self.diagnostics.addTopLevelItem(row)
        self.diagnostics.resizeColumnToContents(0)
        self.diagnostics.resizeColumnToContents(1)
        if analysis.layout is not None:
            self.preview.set_layout(analysis.layout)
            self._rebuild_layer_menu(analysis.layout)
        else:
            self.preview.set_layout(None)
            self._layer_button.setEnabled(False)
        report = analysis.report
        self.statusBar().showMessage(
            f"Validation complete: {len(report.blocking)} blocking, {len(report.errors)} errors, {len(report.warnings)} warnings."
        )

    def _rebuild_layer_menu(self, layout) -> None:
        menu = self._layer_button.menu()
        assert menu is not None
        menu.clear()
        for layer in layout.layers:
            action = menu.addAction(layer)
            action.setCheckable(True)
            action.setChecked(True)
            action.toggled.connect(
                lambda visible, layer_name=layer: self.preview.set_layer_visible(layer_name, visible)
            )
        self._layer_button.setEnabled(bool(layout.layers))

    def _validate_now(self) -> None:
        self._revision += 1
        self._schedule_validation(delay_ms=0)

    def _generate(self) -> None:
        if self._worker_active or self._analysis_revision != self._revision:
            self._generate_after_validation = True
            self._schedule_validation(delay_ms=0)
            self.statusBar().showMessage("Validating before generation…")
            return
        assert self._latest_analysis is not None
        self._complete_generation(self._latest_analysis)

    def _complete_generation(self, analysis: AnalysisResult) -> None:
        if analysis.layout is None or analysis.report.blocking:
            QMessageBox.critical(self, "Generation blocked", self._format_report(analysis.report))
            return
        if analysis.report.errors:
            warning = QMessageBox(self)
            warning.setIcon(QMessageBox.Icon.Warning)
            warning.setWindowTitle("Geometry issues detected")
            warning.setText("The footprint has validation errors. Generate it anyway?")
            warning.setInformativeText(self._format_report(analysis.report))
            return_button = warning.addButton("Return to settings", QMessageBox.ButtonRole.RejectRole)
            force_button = warning.addButton("Generate anyway", QMessageBox.ButtonRole.AcceptRole)
            warning.setDefaultButton(return_button)
            warning.exec()
            if warning.clickedButton() is not force_button:
                return
            self._write_footprint(analysis, force=True)
            return
        self._write_footprint(analysis, force=False)

    def _write_footprint(self, analysis: AnalysisResult, *, force: bool) -> None:
        request = self._request_from_controls()
        try:
            output_path = self.definition.generate(request, force=force, analysis=analysis)
        except GenerationBlocked as error:
            QMessageBox.critical(self, "Generation blocked", self._format_report(error.report))
            return
        except OSError as error:
            QMessageBox.critical(self, "Could not write footprint", str(error))
            return
        suffix = " with validation errors" if force else ""
        self.statusBar().showMessage(f"Wrote {output_path}{suffix}.")
        QMessageBox.information(self, "Footprint generated", f"Wrote:\n{output_path}{suffix}.")

    def _save_project(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save sensor project", "sensor.ips-sensor.json", "Sensor projects (*.ips-sensor.json)"
        )
        if not path:
            return
        if not path.endswith(".ips-sensor.json"):
            path += ".ips-sensor.json"
        try:
            save_project(path, self._request_from_controls())
        except OSError as error:
            QMessageBox.critical(self, "Could not save project", str(error))
            return
        self.statusBar().showMessage(f"Saved project {path}")

    def _load_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load sensor project", "", "Sensor projects (*.ips-sensor.json);;JSON files (*.json)"
        )
        if not path:
            return
        try:
            self.request = load_project(path)
        except ProjectLoadError as error:
            QMessageBox.critical(self, "Could not load project", str(error))
            return
        if self.request.sensor_type != self.definition.id:
            QMessageBox.critical(self, "Unsupported sensor", f"This application supports {self.definition.display_name}.")
            return
        self._populate_controls()
        self._revision += 1
        self._schedule_validation(delay_ms=0)
        self.statusBar().showMessage(f"Loaded project {path}")

    def _reset_defaults(self) -> None:
        self.request = default_request()
        self._populate_controls()
        self._revision += 1
        self._schedule_validation(delay_ms=0)

    def _browse_directory(self, key: str) -> None:
        control = self._controls[key]
        assert isinstance(control, QLineEdit)
        start_dir = control.text() or str(Path.cwd())
        directory = QFileDialog.getExistingDirectory(self, "Select output folder", start_dir)
        if directory:
            control.setText(directory)
            self._on_field_edited()

    def _show_help(self, spec: ParameterSpec) -> None:
        details = spec.help_text
        if spec.unit:
            details += f"\n\nUnits: {spec.unit}."
        if spec.choices:
            details += f"\n\nAllowed values: {', '.join(spec.choices)}."
        if spec.minimum is not None or spec.maximum is not None:
            lower = (
                f"{'>' if spec.exclusive_minimum else '≥'} {spec.minimum:g}"
                if spec.minimum is not None else None
            )
            upper = f"≤ {spec.maximum:g}" if spec.maximum is not None else None
            details += f"\n\nAccepted range: {' and '.join(item for item in (lower, upper) if item)}."
        QMessageBox.information(self, spec.label, details)

    def _toggle_advanced(self, visible: bool) -> None:
        for label, row, _group in self._advanced_rows:
            label.setVisible(visible)
            row.setVisible(visible)

    def _focus_diagnostic(self, item: QTreeWidgetItem, _column: int = 0) -> None:
        field = item.data(0, Qt.ItemDataRole.UserRole)
        if field not in self._controls:
            return
        spec = PARAMETERS_BY_KEY[field]
        if not spec.basic and not self._advanced_toggle.isChecked():
            self._advanced_toggle.setChecked(True)
        control = self._controls[field]
        control.setFocus()

    @staticmethod
    def _format_report(report: ValidationReport) -> str:
        if not report.diagnostics:
            return "No diagnostics."
        return "\n".join(
            f"• {item.severity.value.title()}: {item.message}" for item in report.diagnostics
        )
