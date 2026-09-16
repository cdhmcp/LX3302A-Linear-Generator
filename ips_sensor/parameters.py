"""The single source of truth for editable sensor settings and help text."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any

from .diagnostics import Diagnostic, DiagnosticSeverity
from .models import LinearSensorConfig, OutputConfig


class ParameterKind(str, Enum):
    FLOAT = "float"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    CHOICE = "choice"
    TEXT = "text"
    DIRECTORY = "directory"


@dataclass(frozen=True)
class ParameterSpec:
    key: str
    label: str
    group: str
    kind: ParameterKind
    help_text: str
    basic: bool = True
    unit: str | None = None
    minimum: float | None = None
    maximum: float | None = None
    exclusive_minimum: bool = False
    choices: tuple[str, ...] = ()


def _spec(
    key: str,
    label: str,
    group: str,
    kind: ParameterKind,
    help_text: str,
    **kwargs: Any,
) -> ParameterSpec:
    return ParameterSpec(key, label, group, kind, help_text, **kwargs)


# Keep this catalog deliberately explicit.  A new field must receive a clear
# label and help description before it can appear in the user interface.
PARAMETERS: tuple[ParameterSpec, ...] = (
    _spec("target_x_mm", "Target width", "Sensor", ParameterKind.FLOAT,
          "Width of the moving conductive target. It is used to report the usable measurement range.",
          unit="mm", minimum=0, exclusive_minimum=True),
    _spec("target_y_mm", "Target height", "Sensor", ParameterKind.FLOAT,
          "Height of the moving conductive target. The receiver-wave height is derived from this value.",
          unit="mm", minimum=0, exclusive_minimum=True),
    _spec("stroke_range_mm", "Total stroke range", "Sensor", ParameterKind.FLOAT,
          "Total mechanical travel span represented by the receiver coils. It normally includes target width for useful coupling.",
          unit="mm", minimum=0, exclusive_minimum=True),
    _spec("target_side", "Target side", "Sensor", ParameterKind.CHOICE,
          "Board side closest to the moving target. This determines the assigned copper layers.",
          choices=("top", "bottom")),

    _spec("primary_end_extension_mm", "Primary end extension", "Primary coils", ParameterKind.FLOAT,
          "Extra primary-coil length beyond the secondary stroke on each end. Enter 0 to select the calculated minimum safe extension.",
          unit="mm", minimum=0),
    _spec("primary_y_margin_mm", "Primary vertical margin", "Primary coils", ParameterKind.FLOAT,
          "How far the primary envelope extends beyond receiver windings vertically.",
          unit="mm", minimum=0, exclusive_minimum=True),
    _spec("number_of_primary_turns", "Primary turns", "Primary coils", ParameterKind.INTEGER,
          "Number of turns in each oscillator primary winding. The GUI supports one through five turns.",
          minimum=1, maximum=5),
    _spec("generate_osc1", "Generate OSC1", "Primary coils", ParameterKind.BOOLEAN,
          "Include the first oscillator coil and shared VIN transition.", basic=False),
    _spec("generate_osc2", "Generate OSC2", "Primary coils", ParameterKind.BOOLEAN,
          "Include the second oscillator coil. OSC2 requires OSC1 because both use its VIN transition.", basic=False),

    _spec("number_of_secondary_turns", "Secondary turns", "Secondary coils", ParameterKind.INTEGER,
          "Number of turns in each CL1 and CL2 receiver winding. The currently supported range is one through five.",
          minimum=1, maximum=5),
    _spec("secondary_y_reduction_mm", "Secondary height reduction", "Secondary coils", ParameterKind.FLOAT,
          "Amount subtracted from target height to set the receiver-wave amplitude. A slightly smaller secondary is normally preferred.",
          unit="mm", minimum=0),
    _spec("generate_cl1", "Generate CL1", "Secondary coils", ParameterKind.BOOLEAN,
          "Include the phase-shifted CL1 receiver coil.", basic=False),
    _spec("generate_cl2", "Generate CL2", "Secondary coils", ParameterKind.BOOLEAN,
          "Include the CL2 receiver coil.", basic=False),
    _spec("secondary_curve_samples_per_cycle", "Curve samples per cycle", "Secondary coils", ParameterKind.INTEGER,
          "Polyline resolution used when approximating receiver sinusoidal curves. Higher values smooth the exported footprint but increase validation time.",
          basic=False, minimum=16),
    _spec("trace_width_mm", "Trace width", "Fabrication", ParameterKind.FLOAT,
          "Copper trace width used for every coil segment.", unit="mm", minimum=0, exclusive_minimum=True),
    _spec("trace_spacing_mm", "Trace spacing", "Fabrication", ParameterKind.FLOAT,
          "Minimum edge-to-edge copper spacing assumed by the layout generator.", unit="mm", minimum=0, exclusive_minimum=True),
    _spec("via_hole_size_mm", "Via drill", "Fabrication", ParameterKind.FLOAT,
          "Finished drill diameter for through-hole transition vias and terminals.", unit="mm", minimum=0, exclusive_minimum=True),
    _spec("via_diameter_mm", "Via pad diameter", "Fabrication", ParameterKind.FLOAT,
          "Outer copper diameter for through-hole vias and terminals; it must be at least as large as the drill.",
          unit="mm", minimum=0, exclusive_minimum=True),

    _spec("fanout_side", "Fanout side", "Fanout", ParameterKind.CHOICE,
          "Side of the sensor where external terminals and escape routing are placed.", choices=("left", "right")),
    _spec("terminal_escape_length_mm", "Terminal escape length", "Fanout", ParameterKind.FLOAT,
          "Distance from the primary envelope to the shared terminal column.", unit="mm", minimum=0, exclusive_minimum=True),
    _spec("osc1_vin_exit_offset_mm", "Primary corridor top inset", "Fanout", ParameterKind.FLOAT,
          "Vertical placement of the shared VIN/OSC corridor. 0 selects the automatic, clearance-safe position. "
          "A positive value measures downward from the physical outer copper edge of the primary coil's top rail; "
          "larger values move the corridor lower. Manual positions are checked for clearance to primary copper and "
          "related transition vias.", unit="mm", minimum=0, basic=False),

    _spec("footprint_name", "Footprint name", "Naming", ParameterKind.TEXT,
          "Name written into the exported footprint filename and KiCad footprint header."),
    _spec("reference_text", "Reference text", "Naming", ParameterKind.TEXT,
          "Reference designator text stored in the footprint."),
    _spec("primary_input_pad_name", "Primary input pad", "Naming", ParameterKind.TEXT,
          "Pad name assigned to the shared oscillator input."),
    _spec("osc1_output_pad_name", "OSC1 output pad", "Naming", ParameterKind.TEXT,
          "Pad name assigned to the OSC1 output terminal."),
    _spec("osc2_output_pad_name", "OSC2 output pad", "Naming", ParameterKind.TEXT,
          "Pad name assigned to the OSC2 output terminal."),
    _spec("cl2_output_pad_name", "CL2 output pad", "Naming", ParameterKind.TEXT,
          "Pad name assigned to the CL2 output terminal."),
    _spec("cl2_return_pad_name", "CL2 return pad", "Naming", ParameterKind.TEXT,
          "Pad name assigned to the CL2 return terminal."),
    _spec("cl1_output_pad_name", "CL1 output pad", "Naming", ParameterKind.TEXT,
          "Pad name assigned to the CL1 output terminal."),
    _spec("cl1_return_pad_name", "CL1 return pad", "Naming", ParameterKind.TEXT,
          "Pad name assigned to the CL1 return terminal."),

    _spec("output_dir", "Output folder", "Output", ParameterKind.DIRECTORY,
          "Folder where the generated footprint will be written. Relative paths are resolved from the application working directory."),
)

PARAMETERS_BY_KEY = {item.key: item for item in PARAMETERS}
PARAMETER_GROUPS = (
    "Sensor",
    "Primary coils",
    "Secondary coils",
    "Fabrication",
    "Fanout",
    "Naming",
    "Output",
)


def validate_static_config(
    config: LinearSensorConfig,
    output: OutputConfig,
) -> tuple[Diagnostic, ...]:
    """Aggregate input-level problems without attempting geometry routing."""
    diagnostics: list[Diagnostic] = []
    values = {**config.__dict__, "output_dir": output.output_dir}
    for spec in PARAMETERS:
        value = values[spec.key]
        if spec.kind is ParameterKind.BOOLEAN:
            if not isinstance(value, bool):
                diagnostics.append(_blocking_type(spec, "a true/false value"))
            continue
        if spec.kind is ParameterKind.INTEGER:
            if isinstance(value, bool) or not isinstance(value, int):
                diagnostics.append(_blocking_type(spec, "an integer"))
                continue
        elif spec.kind is ParameterKind.FLOAT:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                diagnostics.append(_blocking_type(spec, "a number"))
                continue
            if not math.isfinite(value):
                diagnostics.append(
                    Diagnostic(
                        "NONFINITE_NUMBER", DiagnosticSeverity.BLOCKING,
                        f"{spec.label} must be a finite number.", spec.key,
                    )
                )
                continue
        elif spec.kind is ParameterKind.CHOICE:
            if value not in spec.choices:
                diagnostics.append(
                    Diagnostic(
                        "INVALID_CHOICE", DiagnosticSeverity.BLOCKING,
                        f"{spec.label} must be one of: {', '.join(spec.choices)}.", spec.key,
                    )
                )
            continue
        elif spec.kind in (ParameterKind.TEXT, ParameterKind.DIRECTORY):
            if not isinstance(value, str) or not value.strip():
                diagnostics.append(
                    Diagnostic(
                        "REQUIRED_TEXT", DiagnosticSeverity.BLOCKING,
                        f"{spec.label} cannot be empty.", spec.key,
                    )
                )
            continue

        if spec.minimum is not None:
            invalid = value <= spec.minimum if spec.exclusive_minimum else value < spec.minimum
            if invalid:
                comparison = ">" if spec.exclusive_minimum else ">="
                diagnostics.append(
                    Diagnostic(
                        "OUT_OF_RANGE", DiagnosticSeverity.BLOCKING,
                        f"{spec.label} must be {comparison} {spec.minimum:g}{' ' + spec.unit if spec.unit else ''}.",
                        spec.key,
                    )
                )
        if spec.maximum is not None and value > spec.maximum:
            diagnostics.append(
                Diagnostic(
                    "OUT_OF_RANGE", DiagnosticSeverity.BLOCKING,
                    f"{spec.label} must be <= {spec.maximum:g}{' ' + spec.unit if spec.unit else ''}.",
                    spec.key,
                )
            )

    if config.generate_osc2 and not config.generate_osc1:
        diagnostics.append(
            Diagnostic(
                "OSC2_REQUIRES_OSC1", DiagnosticSeverity.BLOCKING,
                "OSC2 requires OSC1 because both coils use OSC1's shared VIN transition via.",
                "generate_osc2", "Enable OSC1 or disable OSC2.",
            )
        )
    if config.via_diameter_mm < config.via_hole_size_mm:
        diagnostics.append(
            Diagnostic(
                "VIA_DIAMETER_TOO_SMALL", DiagnosticSeverity.BLOCKING,
                "Via pad diameter must be at least as large as via drill diameter.",
                "via_diameter_mm", "Increase the pad diameter or reduce the drill.",
            )
        )
    if config.target_y_mm > 0 and config.secondary_y_reduction_mm >= config.target_y_mm:
        diagnostics.append(
            Diagnostic(
                "SECONDARY_WIDTH_NONPOSITIVE", DiagnosticSeverity.BLOCKING,
                "Secondary height reduction must leave a positive secondary width.",
                "secondary_y_reduction_mm", "Reduce the height reduction or increase target height.",
            )
        )
    return tuple(diagnostics)


def _blocking_type(spec: ParameterSpec, expected: str) -> Diagnostic:
    return Diagnostic(
        "INVALID_TYPE", DiagnosticSeverity.BLOCKING,
        f"{spec.label} must be {expected}.", spec.key,
    )
