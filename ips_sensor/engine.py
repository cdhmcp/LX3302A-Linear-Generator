"""Sensor-definition facade over the existing, verified routing engine."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import linear_sensor_generator as legacy

from .diagnostics import Diagnostic, DiagnosticSeverity, ValidationReport
from .exporters import EXPORTERS, Exporter
from .layout import CopperArc, CopperLine, FootprintLayout, ThroughHolePad
from .models import GenerationRequest, LinearSensorConfig, OutputConfig
from .parameters import PARAMETERS, validate_static_config


@dataclass(frozen=True)
class AnalysisResult:
    """Layout plus all diagnostics that apply to one generation request."""

    report: ValidationReport
    layout: FootprintLayout | None


class GenerationBlocked(RuntimeError):
    """Raised when a request cannot be written under the selected policy."""

    def __init__(self, report: ValidationReport) -> None:
        super().__init__("Generation is blocked by validation diagnostics.")
        self.report = report


class SensorDefinition:
    """Extension point implemented by each future sensor shape."""

    id: str
    display_name: str

    @property
    def parameter_specs(self):  # pragma: no cover - interface definition
        raise NotImplementedError

    def analyze(self, request: GenerationRequest) -> AnalysisResult:  # pragma: no cover
        raise NotImplementedError


class LinearSensorDefinition(SensorDefinition):
    """LX3302A linear-sensor adapter with strict and relaxed validation modes."""

    id = "linear-lx3302a"
    display_name = "LX3302A linear sensor"

    @property
    def parameter_specs(self):
        return PARAMETERS

    def analyze(self, request: GenerationRequest) -> AnalysisResult:
        if request.sensor_type != self.id:
            report = ValidationReport((
                Diagnostic(
                    "UNSUPPORTED_SENSOR", DiagnosticSeverity.BLOCKING,
                    f"{request.sensor_type!r} is not handled by {self.display_name}.",
                    "sensor_type",
                ),
            ))
            return AnalysisResult(report, None)

        diagnostics = list(validate_static_config(request.config, request.output))
        if any(item.severity is DiagnosticSeverity.BLOCKING for item in diagnostics):
            return AnalysisResult(ValidationReport(tuple(diagnostics)), None)

        # A relaxed build proves that the current request can still be made
        # visible/exported.  It intentionally bypasses only the legacy deep
        # clearance checks; fundamental configuration checks remain active.
        try:
            layout = self.build_layout(request.config, request.output, strict=False)
        except (TypeError, ValueError, KeyError) as error:
            diagnostics.append(_blocking_build_diagnostic(error))
            return AnalysisResult(ValidationReport(tuple(diagnostics)), None)

        # The strict build runs the existing, well-tested geometry checks.
        # It may be expensive, which is why the GUI invokes this method in a
        # worker thread.  A strict failure remains overrideable when relaxed
        # layout construction succeeded above.
        try:
            self.build_layout(request.config, request.output, strict=True)
        except (TypeError, ValueError, KeyError) as error:
            diagnostics.append(_geometry_diagnostic(error))

        return AnalysisResult(ValidationReport(tuple(diagnostics)), layout)

    def build_layout(
        self,
        config: LinearSensorConfig,
        output: OutputConfig,
        *,
        strict: bool,
    ) -> FootprintLayout:
        """Create neutral layout primitives while preserving legacy routing."""
        cfg = legacy.build_config(
            config.as_legacy_overrides(
                output, enforce_geometry_validation=strict,
            )
        )
        primary = legacy.build_primary_geometry(cfg)
        cl2 = legacy.build_cl2_geometry(cfg, primary)
        cl1 = legacy.build_cl1_geometry(cfg, primary, cl2)
        items: list[CopperLine | CopperArc | ThroughHolePad] = []

        def pad(name: str, point: tuple[float, float]) -> None:
            items.append(
                ThroughHolePad(name, point, cfg["via_diameter_mm"], cfg["via_hole_size_mm"])
            )

        if cfg["generate_osc1"]:
            osc1 = primary.coils[0]
            pad(cfg["osc1_output_pad_name"], primary.pads["OSC1_TERMINAL_OUTPUT_VIA"])
            items.extend(
                CopperLine(osc1.layer, start, end, cfg["trace_width_mm"])
                for start, end in osc1.body_segments
            )
            pad(cfg["primary_input_pad_name"], primary.pads["VIN_SHARED_VIA"])
            items.extend(
                CopperLine(osc1.escape_layer, start, end, cfg["trace_width_mm"])
                for start, end in osc1.escape_segments
            )
            pad(cfg["primary_input_pad_name"], primary.pads["VIN_TERMINAL_VIA"])

        if cfg["generate_osc2"]:
            osc2 = next(coil for coil in primary.coils if coil.name == "OSC2")
            pad(cfg["osc2_output_pad_name"], primary.pads["OSC2_TERMINAL_OUTPUT_VIA"])
            items.extend(
                CopperLine(osc2.layer, start, end, cfg["trace_width_mm"])
                for start, end in osc2.body_segments
            )

        if cl2 is not None:
            items.extend(
                CopperLine(cl2.target_layer, start, end, cfg["trace_width_mm"])
                for start, end in cl2.target_segments
            )
            items.extend(
                CopperLine(cl2.inner_layer, start, end, cfg["trace_width_mm"])
                for start, end in cl2.inner_segments
            )
            for label in cl2.via_labels:
                pad_name = {
                    "TERMINAL_OUTPUT_VIA": cfg["cl2_output_pad_name"],
                    "TERMINAL_RETURN_VIA": cfg["cl2_return_pad_name"],
                }.get(label, f"CL2_{label}")
                pad(pad_name, cl2.points[label])

        if cl1 is not None:
            items.extend(
                CopperLine(cl1.target_layer, start, end, cfg["trace_width_mm"])
                for start, end in cl1.target_segments
            )
            items.extend(
                CopperArc(cl1.target_layer, start, mid, end, cfg["trace_width_mm"])
                for start, mid, end in cl1.target_arcs
            )
            items.extend(
                CopperLine(cl1.inner_layer, start, end, cfg["trace_width_mm"])
                for start, end in cl1.inner_segments
            )
            items.extend(
                CopperArc(cl1.inner_layer, start, mid, end, cfg["trace_width_mm"])
                for start, mid, end in cl1.inner_arcs
            )
            items.extend(
                CopperLine(cl1.crossover_layer, start, end, cfg["trace_width_mm"])
                for start, end in cl1.crossover_segments
            )
            for label in cl1.via_labels:
                pad_name = {
                    "TERMINAL_OUTPUT_VIA": cfg["cl1_output_pad_name"],
                    "TERMINAL_RETURN_VIA": cfg["cl1_return_pad_name"],
                }.get(label, f"CL1_{label}")
                pad(pad_name, cl1.points[label])

        dimensions = primary.dimensions
        return FootprintLayout(
            name=cfg["footprint_name"],
            reference=cfg["reference_text"],
            items=tuple(items),
            primary_length_mm=dimensions.primary_length_mm,
            primary_width_mm=dimensions.primary_width_mm,
            secondary_length_mm=dimensions.secondary_length_mm,
            secondary_width_mm=dimensions.secondary_width_mm,
        )

    def generate(
        self,
        request: GenerationRequest,
        *,
        force: bool = False,
        analysis: AnalysisResult | None = None,
    ) -> Path:
        """Write the selected exporter output after enforcing generation policy."""
        analysis = analysis or self.analyze(request)
        if analysis.layout is None or (
            not force and not analysis.report.can_generate_normally
        ) or (force and not analysis.report.can_force_generate):
            raise GenerationBlocked(analysis.report)
        try:
            exporter = EXPORTERS[request.exporter_id]
        except KeyError as error:
            report = analysis.report.with_diagnostic(
                Diagnostic(
                    "UNSUPPORTED_EXPORTER", DiagnosticSeverity.BLOCKING,
                    f"No exporter is registered for {request.exporter_id!r}.", "exporter_id",
                )
            )
            raise GenerationBlocked(report) from error
        return exporter.write(analysis.layout, request.output)


def _blocking_build_diagnostic(error: Exception) -> Diagnostic:
    message = str(error) or type(error).__name__
    return Diagnostic(
        "LAYOUT_BUILD_FAILED", DiagnosticSeverity.BLOCKING, message,
        _field_from_legacy_message(message),
        "Correct the highlighted setting before generating a footprint.",
    )


def _geometry_diagnostic(error: Exception) -> Diagnostic:
    message = str(error) or type(error).__name__
    return Diagnostic(
        "GEOMETRY_VALIDATION_FAILED", DiagnosticSeverity.ERROR, message,
        _field_from_legacy_message(message),
        "Adjust the setting or choose Generate anyway after reviewing the warning.",
    )


def _field_from_legacy_message(message: str) -> str | None:
    lowered = message.lower()
    hints = (
        ("primary_end_extension", "primary_end_extension_mm"),
        ("secondary_y_reduction", "secondary_y_reduction_mm"),
        ("via_diameter", "via_diameter_mm"),
        ("via_hole", "via_hole_size_mm"),
        ("number_of_primary_turns", "number_of_primary_turns"),
        ("number_of_secondary_turns", "number_of_secondary_turns"),
        ("secondary stroke", "stroke_range_mm"),
        ("secondary width", "target_y_mm"),
        ("primary width", "primary_y_margin_mm"),
        ("transition columns", "cl1_transition_column_fraction"),
        ("osc2", "generate_osc2"),
        ("fanout", "fanout_side"),
    )
    return next((field for needle, field in hints if needle in lowered), None)


GENERATORS: dict[str, SensorDefinition] = {LinearSensorDefinition.id: LinearSensorDefinition()}
