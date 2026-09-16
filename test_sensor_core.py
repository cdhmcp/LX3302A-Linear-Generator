"""Coverage for the typed core, project files, exporter, and desktop form."""

from __future__ import annotations

from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ips_sensor.diagnostics import Diagnostic, DiagnosticSeverity, ValidationReport
from ips_sensor.engine import AnalysisResult, GenerationBlocked, LinearSensorDefinition
from ips_sensor.exporters import KiCad9Exporter
from ips_sensor.layout import CopperLine, FootprintLayout, ThroughHolePad
from ips_sensor.models import GenerationRequest, OutputConfig, default_request
from ips_sensor.parameters import PARAMETERS, PARAMETERS_BY_KEY, validate_static_config
from ips_sensor.project import load_project, save_project


class CoreValidationTests(unittest.TestCase):
    def test_parameter_catalog_exposes_every_typed_setting_and_output_directory(self):
        typed_keys = set(asdict(default_request().config))
        catalog_keys = {spec.key for spec in PARAMETERS}
        self.assertEqual(catalog_keys, typed_keys | {"output_dir"})
        self.assertTrue(all(spec.help_text for spec in PARAMETERS))

    def test_static_validation_aggregates_independent_blocking_problems(self):
        request = default_request()
        invalid = replace(
            request.config,
            target_y_mm=1.0,
            secondary_y_reduction_mm=2.0,
            via_diameter_mm=0.1,
            generate_osc1=False,
            generate_osc2=True,
        )
        diagnostics = validate_static_config(invalid, request.output)
        self.assertEqual(
            {item.code for item in diagnostics},
            {"OSC2_REQUIRES_OSC1", "VIA_DIAMETER_TOO_SMALL", "SECONDARY_WIDTH_NONPOSITIVE"},
        )
        self.assertTrue(all(item.severity is DiagnosticSeverity.BLOCKING for item in diagnostics))

    def test_force_policy_requires_an_explicit_override(self):
        layout = FootprintLayout(
            name="TEST",
            reference="REF**",
            items=(CopperLine("F.Cu", (0.0, 0.0), (1.0, 1.0), 0.2),),
            primary_length_mm=1.0,
            primary_width_mm=1.0,
            secondary_length_mm=1.0,
            secondary_width_mm=1.0,
        )
        report = ValidationReport((
            Diagnostic("CLEARANCE", DiagnosticSeverity.ERROR, "Clearance is below the requested limit."),
        ))
        analysis = AnalysisResult(report, layout)
        with tempfile.TemporaryDirectory() as temp_dir:
            request = replace(default_request(), output=OutputConfig(output_dir=temp_dir))
            definition = LinearSensorDefinition()
            with self.assertRaises(GenerationBlocked):
                definition.generate(request, analysis=analysis)
            written = definition.generate(request, force=True, analysis=analysis)
            self.assertTrue(written.exists())

    def test_kicad_exporter_preserves_legacy_default_output(self):
        """The facade must not perturb the established KiCad 9 footprint."""
        import linear_sensor_generator as legacy

        request = default_request()
        definition = LinearSensorDefinition()
        layout = definition.build_layout(request.config, request.output, strict=False)
        legacy_cfg = legacy.build_config(
            request.config.as_legacy_overrides(
                request.output, enforce_geometry_validation=False,
            )
        )
        self.assertEqual(KiCad9Exporter().render(layout), legacy.render_footprint(legacy_cfg))

    def test_primary_corridor_metadata_is_advanced_and_allows_automatic_zero(self):
        spec = PARAMETERS_BY_KEY["osc1_vin_exit_offset_mm"]
        self.assertEqual(spec.label, "Primary corridor top inset")
        self.assertFalse(spec.basic)
        self.assertEqual(spec.minimum, 0)
        self.assertFalse(spec.exclusive_minimum)
        self.assertIn("automatic", spec.help_text.lower())
        self.assertEqual(default_request().config.osc1_vin_exit_offset_mm, 0.0)

    def test_too_small_manual_corridor_inset_is_an_overrideable_setting_diagnostic(self):
        request = default_request()
        request = replace(
            request,
            config=replace(request.config, osc1_vin_exit_offset_mm=1.0),
        )
        analysis = LinearSensorDefinition().analyze(request)
        self.assertIsNotNone(analysis.layout)
        diagnostic = next(
            item
            for item in analysis.report.diagnostics
            if item.field == "osc1_vin_exit_offset_mm"
        )
        self.assertIs(diagnostic.severity, DiagnosticSeverity.ERROR)
        self.assertIn("minimum primary corridor top-edge inset", diagnostic.message)


class ProjectFileTests(unittest.TestCase):
    def test_project_round_trip_preserves_request(self):
        request = replace(
            default_request(),
            config=replace(default_request().config, footprint_name="MY_SENSOR"),
            output=OutputConfig(output_dir="output.pretty"),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sensor.ips-sensor.json"
            save_project(path, request)
            self.assertEqual(load_project(path), request)

    def test_v0_project_migrates_output_directory_out_of_config(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "legacy.ips-sensor.json"
            path.write_text(json.dumps({"config": {"output_dir": "legacy.pretty"}}), encoding="utf-8")
            request = load_project(path)
            self.assertEqual(request.output.output_dir, "legacy.pretty")
            self.assertEqual(request.config.osc1_vin_exit_offset_mm, 0.0)

    def test_v1_project_resets_inert_corridor_offset_to_automatic(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "v1.ips-sensor.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "config": {"osc1_vin_exit_offset_mm": 1.2},
                        "output": {},
                    }
                ),
                encoding="utf-8",
            )
            request = load_project(path)
            self.assertEqual(request.config.osc1_vin_exit_offset_mm, 0.0)

    def test_current_project_preserves_explicit_manual_corridor_inset(self):
        request = replace(
            default_request(),
            config=replace(default_request().config, osc1_vin_exit_offset_mm=2.0),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "v3.ips-sensor.json"
            save_project(path, request)
            self.assertEqual(load_project(path).config.osc1_vin_exit_offset_mm, 2.0)

    def test_v2_project_discards_retired_receiver_routing_controls(self):
        retired = {
            "secondary_jump_runup_via_multiplier": 3.0,
            "secondary_jump_detour_via_multiplier": 0.35,
            "cl1_transition_column_fraction": 0.03,
            "cl1_primary_end_min_clearance_mm": 1.0,
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "v2.ips-sensor.json"
            path.write_text(
                json.dumps({"schema_version": 2, "config": retired, "output": {}}),
                encoding="utf-8",
            )
            request = load_project(path)
            self.assertFalse(set(retired) & set(asdict(request.config)))


try:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication, QAbstractSpinBox, QGraphicsSimpleTextItem
    from ips_sensor.gui.main_window import (
        NoWheelComboBox,
        SensorMainWindow,
        UnitNumberEdit,
        ValidationState,
    )
    from ips_sensor.gui.preview import FootprintPreview
except ImportError:  # pragma: no cover - developer machines can run core-only tests
    QApplication = None
    SensorMainWindow = None


@unittest.skipUnless(QApplication is not None, "PySide6 is not installed")
class GuiFormTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    @staticmethod
    def _analysis_result(
        severity: DiagnosticSeverity = DiagnosticSeverity.WARNING,
    ) -> AnalysisResult:
        layout = FootprintLayout(
            name="PREVIEW",
            reference="REF**",
            items=(
                CopperLine("F.Cu", (0.0, 0.0), (5.0, 1.0), 0.2),
                ThroughHolePad("VIN", (0.0, 0.0), 0.5, 0.25),
            ),
            primary_length_mm=5.0,
            primary_width_mm=1.0,
            secondary_length_mm=5.0,
            secondary_width_mm=1.0,
        )
        return AnalysisResult(
            ValidationReport((
                Diagnostic("CHECK", severity, "Review this setting.", "trace_width_mm"),
            )),
            layout,
        )

    @staticmethod
    def _set_current(window: SensorMainWindow, analysis: AnalysisResult) -> None:
        window._validation_finished(window._revision, analysis, None)

    def test_form_has_all_metadata_backed_controls_and_advanced_toggle(self):
        window = SensorMainWindow()
        try:
            window.show()
            self.app.processEvents()
            self.assertEqual(set(window._controls), {spec.key for spec in PARAMETERS})
            advanced = next(spec for spec in PARAMETERS if not spec.basic)
            self.assertFalse(window._controls[advanced.key].isVisible())
            window._advanced_toggle.setChecked(True)
            self.app.processEvents()
            self.assertTrue(window._controls[advanced.key].isVisible())
            self.assertTrue(window._controls[advanced.key].toolTip() in {"", advanced.help_text})
        finally:
            window.close()

    def test_results_start_not_validated_without_starting_background_analysis(self):
        window = SensorMainWindow()
        try:
            self.assertIs(window._validation_state, ValidationState.NOT_VALIDATED)
            self.assertFalse(window._worker_active)
            self.assertTrue(window._validate_button.isEnabled())
            self.assertFalse(window._generate_button.isEnabled())
            self.assertIn("Not validated", window._result_status.text())
            self.assertIn("Not validated", window.preview.overlay_message or "")
            self.assertIn("Not validated", window.diagnostics.overlay_message or "")
        finally:
            window.close()

    def test_semantic_edits_stale_results_but_unit_switching_does_not(self):
        window = SensorMainWindow()
        try:
            analysis = self._analysis_result()
            self._set_current(window, analysis)
            revision = window._revision
            unit_selector = window._unit_controls["trace_width_mm"]
            unit_selector.setCurrentText("mil")
            self.assertIs(window._validation_state, ValidationState.CURRENT)
            self.assertEqual(window._revision, revision)

            trace_width = window._controls["trace_width_mm"]
            trace_width.setText("10")
            trace_width.textEdited.emit("10")
            self.assertIs(window._validation_state, ValidationState.STALE)
            self.assertEqual(window._revision, revision + 1)
            self.assertIs(window.preview.layout, analysis.layout)
            self.assertIn("out of date", (window.preview.overlay_message or "").lower())
            self.assertIn("Out of date", window._preview_heading.text())
            self.assertIn("Out of date", window._diagnostics_heading.text())
            self.assertFalse(window.diagnostics.isEnabled())
            self.assertFalse(window._layer_button.isEnabled())
            self.assertFalse(window._generate_button.isEnabled())
        finally:
            window.close()

    def test_validation_is_explicit_and_discards_a_result_for_edited_settings(self):
        class CapturePool:
            def __init__(self) -> None:
                self.workers = []

            def start(self, worker) -> None:
                self.workers.append(worker)

        window = SensorMainWindow()
        try:
            pool = CapturePool()
            window._thread_pool = pool
            window._validate_now()
            self.assertIs(window._validation_state, ValidationState.VALIDATING)
            self.assertEqual(len(pool.workers), 1)
            self.assertFalse(window._validate_button.isEnabled())
            window._validate_now()
            self.assertEqual(len(pool.workers), 1)

            validation_revision = window._revision
            window._on_field_edited()
            self.assertIs(window._validation_state, ValidationState.STALE)
            window._validation_finished(validation_revision, self._analysis_result(), None)
            self.assertIs(window._validation_state, ValidationState.STALE)
            self.assertIsNone(window._latest_analysis)
            self.assertTrue(window._validate_button.isEnabled())
            self.assertFalse(window._generate_button.isEnabled())
        finally:
            window.close()

    def test_load_and_reset_mark_existing_results_stale_without_starting_validation(self):
        window = SensorMainWindow()
        try:
            analysis = self._analysis_result()
            self._set_current(window, analysis)
            window._reset_defaults()
            self.assertIs(window._validation_state, ValidationState.STALE)
            self.assertFalse(window._worker_active)
            self.assertIs(window.preview.layout, analysis.layout)
            self.assertFalse(window._generate_button.isEnabled())

            self._set_current(window, analysis)
            with tempfile.TemporaryDirectory() as temp_dir:
                path = Path(temp_dir) / "loaded.ips-sensor.json"
                save_project(path, default_request())
                with patch(
                    "ips_sensor.gui.main_window.QFileDialog.getOpenFileName",
                    return_value=(str(path), "Sensor projects (*.ips-sensor.json)"),
                ):
                    window._load_project()
            self.assertIs(window._validation_state, ValidationState.STALE)
            self.assertFalse(window._worker_active)
            self.assertIs(window.preview.layout, analysis.layout)
            self.assertFalse(window._generate_button.isEnabled())
        finally:
            window.close()

    def test_current_errors_remain_generateable_but_blocking_results_do_not(self):
        window = SensorMainWindow()
        try:
            self._set_current(window, self._analysis_result(DiagnosticSeverity.ERROR))
            self.assertIs(window._validation_state, ValidationState.CURRENT)
            self.assertTrue(window._generate_button.isEnabled())

            window._on_field_edited()
            self._set_current(window, self._analysis_result(DiagnosticSeverity.BLOCKING))
            self.assertTrue(window.diagnostics.isEnabled())
            self.assertFalse(window._generate_button.isEnabled())
            self.assertIn("blocking", window._generate_button.toolTip().lower())
        finally:
            window.close()

    def test_turn_selectors_and_mm_mil_inputs_are_keyboard_only(self):
        window = SensorMainWindow()
        try:
            for key in ("number_of_primary_turns", "number_of_secondary_turns"):
                selector = window._controls[key]
                self.assertIsInstance(selector, NoWheelComboBox)
                self.assertEqual([selector.itemText(index) for index in range(selector.count())], ["1", "2", "3", "4", "5"])

            millimeter_keys = {spec.key for spec in PARAMETERS if spec.unit == "mm"}
            self.assertEqual(set(window._unit_controls), millimeter_keys)
            self.assertTrue(all(selector.width() >= 76 for selector in window._unit_controls.values()))
            self.assertFalse(window.findChildren(QAbstractSpinBox))

            trace_width = window._controls["trace_width_mm"]
            unit_selector = window._unit_controls["trace_width_mm"]
            self.assertIsInstance(trace_width, UnitNumberEdit)
            trace_width.set_mm_value(25.4)
            unit_selector.setCurrentText("mil")
            self.assertEqual(trace_width.text(), "1000")
            trace_width.setText("500")
            request = window._request_from_controls()
            self.assertAlmostEqual(request.config.trace_width_mm, 12.7)
            self.assertIsInstance(request.config.number_of_primary_turns, int)
            self.assertIsInstance(request.config.number_of_secondary_turns, int)
        finally:
            window.close()

    def test_primary_corridor_control_is_advanced_and_converts_mils(self):
        window = SensorMainWindow()
        try:
            window.show()
            self.app.processEvents()
            key = "osc1_vin_exit_offset_mm"
            control = window._controls[key]
            unit_selector = window._unit_controls[key]
            self.assertFalse(control.isVisible())
            window._advanced_toggle.setChecked(True)
            self.app.processEvents()
            self.assertTrue(control.isVisible())
            self.assertIsInstance(control, UnitNumberEdit)
            self.assertEqual(control.value_mm(), 0.0)
            unit_selector.setCurrentText("mil")
            control.setText("100")
            self.assertAlmostEqual(
                window._request_from_controls().config.osc1_vin_exit_offset_mm,
                2.54,
            )
        finally:
            window.close()

    def test_preview_and_diagnostics_consume_neutral_layout(self):
        window = SensorMainWindow()
        try:
            layout = FootprintLayout(
                name="PREVIEW",
                reference="REF**",
                items=(
                    CopperLine("F.Cu", (0.0, 0.0), (5.0, 1.0), 0.2),
                    ThroughHolePad("VIN", (0.0, 0.0), 0.5, 0.25),
                ),
                primary_length_mm=5.0,
                primary_width_mm=1.0,
                secondary_length_mm=5.0,
                secondary_width_mm=1.0,
            )
            report = ValidationReport((
                Diagnostic("CHECK", DiagnosticSeverity.WARNING, "Review this setting.", "trace_width_mm"),
            ))
            analysis = AnalysisResult(report, layout)
            window._latest_analysis = analysis
            window._show_analysis(analysis)
            window._validation_state = ValidationState.CURRENT
            window._update_result_presentation()
            self.assertIs(window.preview.layout, layout)
            self.assertEqual(window.diagnostics.topLevelItemCount(), 1)
            self.assertTrue(window._layer_button.isEnabled())
            self.assertFalse(
                any(isinstance(item, QGraphicsSimpleTextItem) for item in window.preview.scene().items())
            )
            layer_action = window._layer_button.menu().actions()[0]
            layer_action.setChecked(False)
        finally:
            window.close()

    def test_preview_preserves_kicad_y_orientation(self):
        self.assertEqual(FootprintPreview._point((12.5, 3.25)), (12.5, 3.25))
