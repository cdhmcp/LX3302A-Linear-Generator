"""Typed configuration models for inductive-sensor footprint generation.

The legacy generator still accepts a dictionary so existing automation keeps
working.  New callers should use these immutable models instead.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping


@dataclass(frozen=True)
class LinearSensorConfig:
    """All user-configurable settings for the LX3302A linear sensor."""

    target_x_mm: float = 20.0
    target_y_mm: float = 11.0
    stroke_range_mm: float = 70.0
    target_side: str = "top"

    primary_end_extension_mm: float = 0.0
    primary_y_margin_mm: float = 0.075
    number_of_primary_turns: int = 3

    number_of_secondary_turns: int = 4
    secondary_y_reduction_mm: float = 1.5

    trace_width_mm: float = 8 * 0.0254
    trace_spacing_mm: float = 9 * 0.0254
    via_hole_size_mm: float = 8 * 0.0254
    via_diameter_mm: float = 16 * 0.0254

    fanout_side: str = "left"
    terminal_escape_length_mm: float = 10.0
    # 0 preserves the automatically calculated, clearance-safe corridor position.
    osc1_vin_exit_offset_mm: float = 0.0

    footprint_name: str = "LX3302A_LINEAR_SENSOR_COILS"
    reference_text: str = "REF**"
    primary_input_pad_name: str = "VIN"
    osc1_output_pad_name: str = "OSC1"
    osc2_output_pad_name: str = "OSC2"
    cl2_output_pad_name: str = "CL2"
    cl2_return_pad_name: str = "CL2-GND"
    cl1_output_pad_name: str = "CL1"
    cl1_return_pad_name: str = "CL1-GND"

    generate_osc1: bool = True
    generate_osc2: bool = True
    generate_cl2: bool = True
    generate_cl1: bool = True

    secondary_curve_samples_per_cycle: int = 256

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "LinearSensorConfig":
        """Create a config from known fields, retaining defaults for omissions."""
        allowed = {item.name for item in fields(cls)}
        return cls(**{name: value for name, value in values.items() if name in allowed})

    def as_legacy_overrides(
        self,
        output: "OutputConfig",
        *,
        enforce_geometry_validation: bool,
    ) -> dict[str, Any]:
        """Translate this typed model for the compatibility generator."""
        values = asdict(self)
        values["output_dir"] = output.output_dir
        # This is intentionally internal.  The GUI offers an explicit forced
        # generation workflow instead of exposing this implementation flag.
        values["allow_invalid_geometry"] = not enforce_geometry_validation
        return values


@dataclass(frozen=True)
class OutputConfig:
    """Destination settings independent of sensor geometry."""

    output_dir: str = "InductiveSensors.pretty"

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "OutputConfig":
        allowed = {item.name for item in fields(cls)}
        return cls(**{name: value for name, value in values.items() if name in allowed})


@dataclass(frozen=True)
class GenerationRequest:
    """A fully reproducible request to create one sensor footprint."""

    config: LinearSensorConfig
    output: OutputConfig = OutputConfig()
    sensor_type: str = "linear-lx3302a"
    exporter_id: str = "kicad9"


def default_request() -> GenerationRequest:
    """Return the safe, editable default request used by the application."""
    return GenerationRequest(config=LinearSensorConfig())
