#!/usr/bin/env python3
"""KiCad footprint generator for LX3302A linear primary coil layouts.

This first milestone emits the two primary oscillator windings only.  Receiver
dimensions remain in the configuration because the primary envelope is sized
from the future CL1/CL2 sensing region.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path


MIL_TO_MM = 0.0254

Point = tuple[float, float]
Segment = tuple[Point, Point]


# =============================================================================
# Main Properties
# =============================================================================
# Start here for normal sensor-size and placement changes.
MAIN_PROPERTIES = {
    # Moving target and measurement-region inputs
    "target_x_mm": 21.0,
    "target_y_mm": 7.0,
    "measurement_range_mm": 50.0,
    "limit_before_mm": 10.0,
    "limit_after_mm": 10.0,
    "secondary_period_multiplier": 2.0,
    "secondary_y_reduction_mm": 1.5,

    # Primary oscillator sizing
    "primary_end_extension_mm": 3.0,
    "primary_y_margin_mm": 3.0,
    "number_of_primary_turns": 2,

    # Placement and output
    "target_side": "top",
    "fanout_side": "left",
    "output_dir": "InductiveSensors.pretty",
    "footprint_name": "LX3302A_LINEAR_PRIMARY_COILS",
    "reference_text": "REF**",
    "primary_input_pad_name": "VIN",
    "osc1_output_pad_name": "OSC1",
    "osc2_output_pad_name": "OSC2",
    "generate_osc1": True,
    "generate_osc2": True,
}


# =============================================================================
# Fine Tuning Properties
# =============================================================================
# These values tune fabrication constraints and the fanout outside the coil.
FINE_TUNING_PROPERTIES = {
    "trace_width_mm": 6 * MIL_TO_MM,
    "trace_spacing_mm": 7 * MIL_TO_MM,
    "via_hole_size_mm": 10 * MIL_TO_MM,
    "via_diameter_mm": 20 * MIL_TO_MM,
    "fanout_escape_length_mm": 2.0,
    "fanout_pad_run_mm": 1.0,
    "fanout_pad_pitch_mm": 1.0,
}


@dataclass(frozen=True)
class SensorDimensions:
    """Calculated sensing and primary envelope dimensions in millimeters."""

    secondary_period_mm: float
    secondary_length_mm: float
    secondary_width_mm: float
    primary_length_mm: float
    primary_width_mm: float


@dataclass(frozen=True)
class PrimaryCoil:
    """One oscillator path and its assigned PCB layer."""

    name: str
    layer: str
    body_segments: tuple[Segment, ...]
    fanout_segments: tuple[Segment, ...]


@dataclass(frozen=True)
class PrimaryGeometry:
    """All primary geometry and the pads that connect it."""

    dimensions: SensorDimensions
    pads: dict[str, Point]
    coils: tuple[PrimaryCoil, PrimaryCoil]


def build_config(overrides: dict | None = None) -> dict:
    """Combine user-editable settings and optional programmatic overrides."""
    cfg = {
        **MAIN_PROPERTIES,
        **FINE_TUNING_PROPERTIES,
    }
    if overrides:
        cfg.update(overrides)
    return cfg


def calculate_dimensions(cfg: dict) -> SensorDimensions:
    """Calculate receiver reference bounds and the primary outer centerline."""
    secondary_period = cfg["target_x_mm"] * cfg["secondary_period_multiplier"]
    secondary_length = (
        cfg["measurement_range_mm"]
        + cfg["limit_before_mm"]
        + cfg["limit_after_mm"]
        + cfg["target_x_mm"]
    )
    secondary_width = cfg["target_y_mm"] - cfg["secondary_y_reduction_mm"]
    primary_length = secondary_length + (2.0 * cfg["primary_end_extension_mm"])
    primary_width = secondary_width + (2.0 * cfg["primary_y_margin_mm"])
    return SensorDimensions(
        secondary_period_mm=secondary_period,
        secondary_length_mm=secondary_length,
        secondary_width_mm=secondary_width,
        primary_length_mm=primary_length,
        primary_width_mm=primary_width,
    )


def primary_layers(cfg: dict) -> tuple[str, str]:
    """Return ``(OSC1, OSC2)`` layers opposite the configured target side."""
    if cfg["target_side"] == "top":
        return "B.Cu", "In2.Cu"
    if cfg["target_side"] == "bottom":
        return "F.Cu", "In1.Cu"
    raise ValueError("target_side must be 'top' or 'bottom'.")


def fanout_direction(cfg: dict) -> float:
    """Return -1 for a left breakout or +1 for a right breakout."""
    if cfg["fanout_side"] == "left":
        return -1.0
    if cfg["fanout_side"] == "right":
        return 1.0
    raise ValueError("fanout_side must be 'left' or 'right'.")


def trace_pitch(cfg: dict) -> float:
    return cfg["trace_width_mm"] + cfg["trace_spacing_mm"]


def distance(point_a: Point, point_b: Point) -> float:
    return math.hypot(point_b[0] - point_a[0], point_b[1] - point_a[1])


def point_to_segment_distance(point: Point, segment: Segment) -> float:
    """Return minimum distance between a point and a finite line segment."""
    start, end = segment
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length_squared = (dx * dx) + (dy * dy)
    if length_squared == 0.0:
        return distance(point, start)
    projection = (
        ((point[0] - start[0]) * dx) + ((point[1] - start[1]) * dy)
    ) / length_squared
    projection = min(1.0, max(0.0, projection))
    closest = (start[0] + (projection * dx), start[1] + (projection * dy))
    return distance(point, closest)


def validate_config(cfg: dict, dimensions: SensorDimensions | None = None) -> None:
    """Reject impossible envelope, fabrication, or breakout inputs."""
    positive_values = (
        "target_x_mm",
        "target_y_mm",
        "measurement_range_mm",
        "secondary_period_multiplier",
        "primary_end_extension_mm",
        "primary_y_margin_mm",
        "trace_width_mm",
        "trace_spacing_mm",
        "via_hole_size_mm",
        "via_diameter_mm",
        "fanout_escape_length_mm",
        "fanout_pad_run_mm",
        "fanout_pad_pitch_mm",
    )
    for name in positive_values:
        if cfg[name] <= 0:
            raise ValueError(f"{name} must be > 0.")
    for name in ("limit_before_mm", "limit_after_mm", "secondary_y_reduction_mm"):
        if cfg[name] < 0:
            raise ValueError(f"{name} must be >= 0.")

    if not isinstance(cfg["number_of_primary_turns"], int) or cfg["number_of_primary_turns"] < 1:
        raise ValueError("number_of_primary_turns must be a positive integer.")
    if not isinstance(cfg["generate_osc1"], bool) or not isinstance(cfg["generate_osc2"], bool):
        raise ValueError("generate_osc1 and generate_osc2 must be booleans.")

    primary_layers(cfg)
    fanout_direction(cfg)
    dimensions = dimensions or calculate_dimensions(cfg)

    if dimensions.secondary_width_mm <= 0:
        raise ValueError("secondary_y_reduction_mm must leave a positive secondary width.")
    if cfg["via_diameter_mm"] < cfg["via_hole_size_mm"]:
        raise ValueError("via_diameter_mm must be at least as large as via_hole_size_mm.")

    pitch = trace_pitch(cfg)
    inset = (cfg["number_of_primary_turns"] - 1) * pitch
    inner_width = dimensions.primary_width_mm - (2.0 * inset)
    inner_length = dimensions.primary_length_mm - (2.0 * inset)
    if inner_width < pitch:
        raise ValueError("Primary width is insufficient for requested turns and trace spacing.")
    if inner_length < pitch:
        raise ValueError("Primary length is insufficient for requested turns and trace spacing.")
    if cfg["fanout_pad_pitch_mm"] < cfg["via_diameter_mm"] + cfg["trace_spacing_mm"]:
        raise ValueError("fanout_pad_pitch_mm does not maintain via-to-via clearance.")


def build_rectangular_spiral(
    cfg: dict,
    dimensions: SensorDimensions,
) -> tuple[tuple[Segment, ...], Point, Point]:
    """Build an open, sharp-corner rectangular spiral with endpoints at the fanout end."""
    side = fanout_direction(cfg)
    pitch = trace_pitch(cfg)
    half_length = dimensions.primary_length_mm / 2.0
    half_width = dimensions.primary_width_mm / 2.0
    segments: list[Segment] = []
    first_endpoint: Point | None = None
    final_endpoint: Point | None = None

    for turn in range(cfg["number_of_primary_turns"]):
        x_radius = half_length - (turn * pitch)
        y_radius = half_width - (turn * pitch)
        near_x = side * x_radius
        far_x = -near_x
        traverse_top_first = turn % 2 == 0

        if traverse_top_first:
            points = (
                (near_x, -y_radius),
                (far_x, -y_radius),
                (far_x, y_radius),
                (near_x, y_radius),
            )
        else:
            points = (
                (near_x, y_radius),
                (far_x, y_radius),
                (far_x, -y_radius),
                (near_x, -y_radius),
            )

        if first_endpoint is None:
            first_endpoint = points[0]
        for start, end in zip(points, points[1:]):
            segments.append((start, end))

        final_endpoint = points[-1]
        if turn == cfg["number_of_primary_turns"] - 1:
            continue

        next_near_x = side * (x_radius - pitch)
        next_y_radius = y_radius - pitch
        next_y = next_y_radius if traverse_top_first else -next_y_radius
        corner_step = (near_x, next_y)
        next_start = (next_near_x, next_y)
        segments.append((final_endpoint, corner_step))
        segments.append((corner_step, next_start))

    assert first_endpoint is not None
    assert final_endpoint is not None
    return tuple(segments), first_endpoint, final_endpoint


def calculate_fanout_pads(cfg: dict, dimensions: SensorDimensions) -> dict[str, Point]:
    """Place ordered through-via connections above and outside the active loop."""
    side = fanout_direction(cfg)
    pad_x = side * (
        (dimensions.primary_length_mm / 2.0)
        + cfg["fanout_escape_length_mm"]
        + cfg["fanout_pad_run_mm"]
    )
    outer_top_y = -(dimensions.primary_width_mm / 2.0)
    pitch = cfg["fanout_pad_pitch_mm"]
    return {
        cfg["osc2_output_pad_name"]: (pad_x, outer_top_y - (2.0 * pitch)),
        cfg["primary_input_pad_name"]: (pad_x, outer_top_y - pitch),
        cfg["osc1_output_pad_name"]: (pad_x, outer_top_y),
    }


def route_endpoint_to_pad(
    cfg: dict,
    dimensions: SensorDimensions,
    endpoint: Point,
    pad: Point,
) -> tuple[Segment, Segment]:
    """Route through a pad lane outside the loop before reaching a plated via."""
    side = fanout_direction(cfg)
    routing_x = side * (
        (dimensions.primary_length_mm / 2.0) + cfg["fanout_escape_length_mm"]
    )
    lane_point = (routing_x, pad[1])
    return (endpoint, lane_point), (lane_point, pad)


def validate_breakout_clearance(
    cfg: dict,
    pads: dict[str, Point],
    body_segments: tuple[Segment, ...],
    osc1_vin_route: tuple[Segment, ...],
    osc1_output_route: tuple[Segment, ...],
    osc2_vin_route: tuple[Segment, ...],
    osc2_output_route: tuple[Segment, ...],
) -> None:
    """Ensure plated fanout vias do not contact unintended copper paths."""
    pad_names = tuple(pads)
    minimum_pad_distance = cfg["via_diameter_mm"] + cfg["trace_spacing_mm"]
    for index, name in enumerate(pad_names):
        for other_name in pad_names[index + 1:]:
            if distance(pads[name], pads[other_name]) < minimum_pad_distance:
                raise ValueError("Fanout pad spacing causes plated via clearance violations.")

    minimum_trace_distance = (
        (cfg["via_diameter_mm"] / 2.0)
        + (cfg["trace_width_mm"] / 2.0)
        + cfg["trace_spacing_mm"]
    )
    vin = cfg["primary_input_pad_name"]
    osc1 = cfg["osc1_output_pad_name"]
    osc2 = cfg["osc2_output_pad_name"]
    osc1_fanout = osc1_vin_route + osc1_output_route
    osc2_fanout = osc2_vin_route + osc2_output_route
    prohibited = {
        vin: body_segments + osc1_output_route + osc2_output_route,
        osc1: body_segments + osc1_vin_route + osc2_fanout,
        osc2: body_segments + osc1_fanout + osc2_vin_route,
    }
    for name, segments in prohibited.items():
        for segment in segments:
            if point_to_segment_distance(pads[name], segment) < minimum_trace_distance:
                raise ValueError(f"{name} breakout via violates primary trace clearance.")


def build_primary_geometry(cfg: dict | None = None) -> PrimaryGeometry:
    """Return calculated OSC1/OSC2 paths and their fanout pads."""
    cfg = build_config() if cfg is None else cfg
    dimensions = calculate_dimensions(cfg)
    validate_config(cfg, dimensions)
    body_segments, first_endpoint, final_endpoint = build_rectangular_spiral(cfg, dimensions)
    pads = calculate_fanout_pads(cfg, dimensions)
    osc1_layer, osc2_layer = primary_layers(cfg)

    vin = pads[cfg["primary_input_pad_name"]]
    osc1_output = pads[cfg["osc1_output_pad_name"]]
    osc2_output = pads[cfg["osc2_output_pad_name"]]
    osc1_vin_route = route_endpoint_to_pad(cfg, dimensions, first_endpoint, vin)
    osc1_output_route = route_endpoint_to_pad(cfg, dimensions, final_endpoint, osc1_output)
    osc1_fanout = osc1_vin_route + osc1_output_route
    # Connect VIN to the opposite spiral end to reverse OSC2 winding polarity.
    osc2_vin_route = route_endpoint_to_pad(cfg, dimensions, final_endpoint, vin)
    osc2_output_route = route_endpoint_to_pad(cfg, dimensions, first_endpoint, osc2_output)
    osc2_fanout = osc2_vin_route + osc2_output_route
    validate_breakout_clearance(
        cfg,
        pads,
        body_segments,
        osc1_vin_route,
        osc1_output_route,
        osc2_vin_route,
        osc2_output_route,
    )

    return PrimaryGeometry(
        dimensions=dimensions,
        pads=pads,
        coils=(
            PrimaryCoil("OSC1", osc1_layer, body_segments, osc1_fanout),
            PrimaryCoil("OSC2", osc2_layer, tuple(reversed(body_segments)), osc2_fanout),
        ),
    )


def fp_line(start: Point, end: Point, width: float, layer: str) -> str:
    return f'''  (fp_line (start {start[0]:.6f} {start[1]:.6f}) (end {end[0]:.6f} {end[1]:.6f})
    (stroke (width {width:.6f}) (type solid)) (layer "{layer}"))\n'''


def pad_thru_hole(name: str, point: Point, diameter: float, drill: float) -> str:
    return f'''  (pad "{name}" thru_hole circle (at {point[0]:.6f} {point[1]:.6f}) (size {diameter:.6f} {diameter:.6f}) (drill {drill:.6f})
    (layers "*.Cu" "*.Mask"))\n'''


def kicad_header(name: str) -> str:
    return f'''(footprint "{name}"
  (version 20240201)
  (generator "linear_sensor_generator")
  (layer "F.Cu")
  (attr smd)
'''


def fp_text(reference: str, value: str) -> str:
    return f'''  (fp_text reference "{reference}" (at 0 0) (layer "F.SilkS") hide
    (effects (font (size 1 1) (thickness 0.15))))
  (fp_text value "{value}" (at 0 0) (layer "F.Fab") hide
    (effects (font (size 1 1) (thickness 0.15))))
'''


def render_footprint(cfg: dict | None = None) -> str:
    """Render the configured primary-only KiCad footprint text."""
    cfg = build_config() if cfg is None else cfg
    geometry = build_primary_geometry(cfg)
    sections = [
        kicad_header(cfg["footprint_name"]),
        fp_text(cfg["reference_text"], cfg["footprint_name"]),
    ]

    if cfg["generate_osc1"] or cfg["generate_osc2"]:
        sections.append(
            pad_thru_hole(
                cfg["primary_input_pad_name"],
                geometry.pads[cfg["primary_input_pad_name"]],
                cfg["via_diameter_mm"],
                cfg["via_hole_size_mm"],
            )
        )

    enabled = {
        "OSC1": cfg["generate_osc1"],
        "OSC2": cfg["generate_osc2"],
    }
    output_pad_names = {
        "OSC1": cfg["osc1_output_pad_name"],
        "OSC2": cfg["osc2_output_pad_name"],
    }
    for coil in geometry.coils:
        if not enabled[coil.name]:
            continue
        for segment in coil.body_segments + coil.fanout_segments:
            sections.append(fp_line(segment[0], segment[1], cfg["trace_width_mm"], coil.layer))
        pad_name = output_pad_names[coil.name]
        sections.append(
            pad_thru_hole(
                pad_name,
                geometry.pads[pad_name],
                cfg["via_diameter_mm"],
                cfg["via_hole_size_mm"],
            )
        )

    sections.append(")\n")
    return "".join(sections)


def write_linear_sensor_footprint(cfg: dict | None = None) -> Path:
    """Write the configured KiCad footprint file and return its output path."""
    cfg = build_config() if cfg is None else cfg
    output_dir = Path.cwd() / cfg["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'{cfg["footprint_name"]}.kicad_mod'
    output_path.write_text(render_footprint(cfg), encoding="ascii")
    return output_path


def main() -> None:
    cfg = build_config()
    geometry = build_primary_geometry(cfg)
    output_path = write_linear_sensor_footprint(cfg)
    dims = geometry.dimensions
    print(f"Wrote {output_path}")
    print(
        "Primary outer centerline envelope: "
        f"{dims.primary_length_mm:.3f} mm x {dims.primary_width_mm:.3f} mm"
    )


if __name__ == "__main__":
    main()
