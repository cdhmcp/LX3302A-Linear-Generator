#!/usr/bin/env python3
"""KiCad footprint generator for LX3302A linear primary coil layouts.

OSC1 and OSC2 are built from their annotated primary-coil point maps. Receiver
dimensions remain in the configuration because the primary envelope is sized
from the future CL1/CL2 sensing region.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path


MIL_TO_MM = 0.0254
GEOMETRY_TOLERANCE_MM = 1e-9

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
    "number_of_primary_turns": 3,

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
    "terminal_escape_length_mm": 5.0,
    "osc1_vin_exit_offset_mm": 1.2,
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
    escape_layer: str
    points: dict[str, Point]
    body_segments: tuple[Segment, ...]
    escape_segments: tuple[Segment, ...]


@dataclass(frozen=True)
class PrimaryGeometry:
    """All primary geometry and the pads that connect it."""

    dimensions: SensorDimensions
    pads: dict[str, Point]
    coils: tuple[PrimaryCoil, ...]


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


def target_facing_layer(cfg: dict) -> str:
    """Return the external copper layer nearest the moving target."""
    if cfg["target_side"] == "top":
        return "F.Cu"
    if cfg["target_side"] == "bottom":
        return "B.Cu"
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


def parallel_45_center_shift(cfg: dict) -> float:
    """Offset parallel 45 degree transitions so their perpendicular pitch is legal."""
    return trace_pitch(cfg) * (math.sqrt(2.0) - 1.0)


def parallel_45_junction_separation(cfg: dict) -> float:
    """Separate same-column ends of adjacent 45 degree transitions."""
    return trace_pitch(cfg) * math.sqrt(2.0)


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
        "terminal_escape_length_mm",
        "osc1_vin_exit_offset_mm",
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
    if cfg["generate_osc2"] and not cfg["generate_osc1"]:
        raise ValueError("OSC2 requires OSC1 because it shares OSC1's VIN transition via.")

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
def osc1_via_trace_clearance(cfg: dict) -> float:
    """Return center-to-center clearance from the U via to an adjacent trace."""
    return (
        (cfg["via_diameter_mm"] / 2.0)
        + (cfg["trace_width_mm"] / 2.0)
        + cfg["trace_spacing_mm"]
    )


def osc1_turn_labels(turn_index: int) -> tuple[str, str, str, str, str, str]:
    """Return point-map labels for a turn, extending names beyond the reference."""
    point_map_labels = (
        ("C", "D", "E", "F", "G", "H"),
        ("I", "J", "K", "L", "M", "N"),
        ("O", "P", "Q", "R", "S", "T"),
    )
    if turn_index < len(point_map_labels):
        return point_map_labels[turn_index]
    turn_number = turn_index + 1
    return (
        f"TURN{turn_number}_START",
        f"TURN{turn_number}_BOTTOM_NEAR",
        f"TURN{turn_number}_BOTTOM_FAR",
        f"TURN{turn_number}_TOP_FAR",
        f"TURN{turn_number}_TOP_NEAR",
        f"TURN{turn_number}_END",
    )


def build_osc1_point_map(cfg: dict, dimensions: SensorDimensions) -> dict[str, Point]:
    """Construct OSC1 points using the annotated A-through-V path pattern."""
    side = fanout_direction(cfg)
    pitch = trace_pitch(cfg)
    diagonal_junction_separation = parallel_45_junction_separation(cfg)
    half_length = dimensions.primary_length_mm / 2.0
    half_width = dimensions.primary_width_mm / 2.0
    outer_near_x = side * half_length
    transition_half_height = pitch / 2.0
    entry_x = outer_near_x + (side * transition_half_height)
    terminal_x = outer_near_x + (side * cfg["terminal_escape_length_mm"])
    points: dict[str, Point] = {
        "A": (terminal_x, 0.0),
        "B": (entry_x, 0.0),
    }

    start_y = transition_half_height
    for turn in range(cfg["number_of_primary_turns"]):
        x_near = side * (half_length - (turn * pitch))
        x_far = -x_near
        y_top = -(half_width - (turn * pitch))
        y_bottom = half_width - (turn * pitch)
        turn_labels = osc1_turn_labels(turn)
        start, bottom_near, bottom_far, top_far, top_near, end = turn_labels
        points[start] = (x_near, start_y)
        points[bottom_near] = (x_near, y_bottom)
        points[bottom_far] = (x_far, y_bottom)
        points[top_far] = (x_far, y_top)
        points[top_near] = (x_near, y_top)
        points[end] = (x_near, start_y - diagonal_junction_separation)
        start_y = points[end][1] + pitch

    inner_near_x = side * (half_length - ((cfg["number_of_primary_turns"] - 1) * pitch))
    via_transition = osc1_via_trace_clearance(cfg)
    last_end = osc1_turn_labels(cfg["number_of_primary_turns"] - 1)[5]
    via_y = -cfg["osc1_vin_exit_offset_mm"]
    # The requested exit Y controls U; T is shifted upward so T-U remains a
    # 45 degree descent into the via while honoring via-to-trace clearance.
    points[last_end] = (inner_near_x, via_y - via_transition)
    via_x = inner_near_x - (side * via_transition)
    points["U"] = (via_x, via_y)
    minimum_pad_distance = cfg["via_diameter_mm"] + cfg["trace_spacing_mm"]
    terminal_separation_y = abs(via_y - points["A"][1])
    additional_fanout = 0.0
    if terminal_separation_y < minimum_pad_distance:
        additional_fanout = math.sqrt(
            (minimum_pad_distance * minimum_pad_distance)
            - (terminal_separation_y * terminal_separation_y)
        )
    points["V"] = (terminal_x + (side * additional_fanout), via_y)
    return points


def build_osc1_segments(
    cfg: dict,
    points: dict[str, Point],
) -> tuple[tuple[Segment, ...], tuple[Segment, ...]]:
    """Return OSC1 bottom-layer winding and target-facing escape segments."""
    point_sequence = ["A", "B"]
    for turn in range(cfg["number_of_primary_turns"]):
        point_sequence.extend(osc1_turn_labels(turn))
    point_sequence.append("U")
    body = tuple(
        (points[start], points[end])
        for start, end in zip(point_sequence, point_sequence[1:])
    )
    return body, ((points["U"], points["V"]),)


def osc2_turn_labels(turn_index: int) -> tuple[str, str, str, str]:
    """Return OSC2 labels for one overlaid perimeter in point-map order."""
    point_map_labels = (
        ("G", "OUTER_TOP_FAR", "OUTER_BOTTOM_FAR", "J"),
        ("M", "MIDDLE_TOP_FAR", "MIDDLE_BOTTOM_FAR", "P"),
        ("S", "INNER_TOP_FAR", "INNER_BOTTOM_FAR", "V"),
    )
    if turn_index < len(point_map_labels):
        return point_map_labels[turn_index]
    turn_number = turn_index + 1
    return (
        f"OSC2_TURN{turn_number}_START",
        f"OSC2_TURN{turn_number}_TOP_FAR",
        f"OSC2_TURN{turn_number}_BOTTOM_FAR",
        f"OSC2_TURN{turn_number}_END",
    )


def osc2_transition_labels(turn_index: int) -> tuple[str, str]:
    """Return the labels between an overlaid OSC2 perimeter and the next one."""
    point_map_labels = (("K", "L"), ("Q", "R"))
    if turn_index < len(point_map_labels):
        return point_map_labels[turn_index]
    turn_number = turn_index + 1
    return (
        f"OSC2_AFTER_TURN{turn_number}_NEAR",
        f"OSC2_AFTER_TURN{turn_number}_INNER",
    )


def build_osc2_point_map(
    cfg: dict,
    osc1_points: dict[str, Point],
) -> dict[str, Point]:
    """Construct OSC2 from its mapped entry, overlaid turns, and shared VIN via."""
    side = fanout_direction(cfg)
    pitch = trace_pitch(cfg)
    junction_separation = parallel_45_junction_separation(cfg)
    via_clearance = osc1_via_trace_clearance(cfg)
    pad_clearance = cfg["via_diameter_mm"] + cfg["trace_spacing_mm"]
    turn_count = cfg["number_of_primary_turns"]
    outer_x = osc1_points[osc1_turn_labels(0)[1]][0]

    # Join the midpoint entry after clearing OSC1's through-via at A.
    b_x = osc1_points["A"][0] - (side * via_clearance * math.sqrt(2.0))
    points: dict[str, Point] = {
        "A": (b_x + (side * pad_clearance), pad_clearance),
        "B": (b_x, 0.0),
        "C": (outer_x + (side * 2.0 * pitch), 0.0),
        "X": osc1_points["U"],
    }

    # Overlay each OSC1 rectangular perimeter in the opposite traversal order.
    near_x: list[float] = []
    for turn in range(turn_count):
        osc1_labels = osc1_turn_labels(turn)
        osc2_labels = osc2_turn_labels(turn)
        source_labels = (
            osc1_labels[4],
            osc1_labels[3],
            osc1_labels[2],
            osc1_labels[1],
        )
        for osc2_label, osc1_label in zip(osc2_labels, source_labels):
            points[osc2_label] = osc1_points[osc1_label]
        near_x.append(osc1_points[osc1_labels[1]][0])

    # Work backward from shared VIN so the independent 45 degree transitions
    # retain legal pitch as more turns are added.
    points["W"] = (near_x[-1], points["X"][1] + via_clearance)
    transition_tail_y = points["W"][1]
    for turn in reversed(range(turn_count - 1)):
        near_label, inner_label = osc2_transition_labels(turn)
        points[inner_label] = (near_x[turn + 1], transition_tail_y - junction_separation)
        points[near_label] = (near_x[turn], points[inner_label][1] + pitch)
        transition_tail_y = points[near_label][1]

    points["F"] = (outer_x, transition_tail_y - junction_separation)
    points["E"] = (outer_x + (side * pitch), points["F"][1] + pitch)
    points["D"] = (points["E"][0], -pitch)
    return points


def build_osc2_segments(cfg: dict, points: dict[str, Point]) -> tuple[Segment, ...]:
    """Return OSC2 path in alphabetical point-map order, with hidden far corners."""
    point_sequence = ["A", "B", "C", "D", "E", "F"]
    for turn in range(cfg["number_of_primary_turns"]):
        point_sequence.extend(osc2_turn_labels(turn))
        if turn < cfg["number_of_primary_turns"] - 1:
            point_sequence.extend(osc2_transition_labels(turn))
    point_sequence.extend(("W", "X"))
    return tuple(
        (points[start], points[end])
        for start, end in zip(point_sequence, point_sequence[1:])
    )


def validate_osc1_clearance(
    cfg: dict,
    points: dict[str, Point],
    body_segments: tuple[Segment, ...],
    escape_segments: tuple[Segment, ...],
) -> None:
    """Ensure the OSC1 via transition and terminal vias are manufacturable."""
    minimum_pad_distance = cfg["via_diameter_mm"] + cfg["trace_spacing_mm"]
    if cfg["osc1_vin_exit_offset_mm"] < osc1_via_trace_clearance(cfg):
        raise ValueError(
            "osc1_vin_exit_offset_mm is too small for horizontal A-B and U-V "
            "via-to-trace clearance."
        )
    for start, end in (("A", "V"), ("A", "U"), ("U", "V")):
        if distance(points[start], points[end]) < minimum_pad_distance:
            raise ValueError(f"OSC1 vias {start} and {end} violate plated via clearance.")

    minimum_trace_distance = osc1_via_trace_clearance(cfg)
    connected_body = {
        "A": body_segments[:1],
        "U": body_segments[-1:],
        "V": (),
    }
    for pad_name in ("A", "U", "V"):
        for segment in body_segments:
            if segment in connected_body[pad_name]:
                continue
            if point_to_segment_distance(points[pad_name], segment) < minimum_trace_distance:
                raise ValueError(f"OSC1 {pad_name} via violates clearance to winding copper.")
    if point_to_segment_distance(points["A"], escape_segments[0]) < minimum_trace_distance:
        raise ValueError("OSC1 A via violates clearance to the top-layer VIN escape.")


def validate_osc2_clearance(
    cfg: dict,
    osc1_points: dict[str, Point],
    osc1_body_segments: tuple[Segment, ...],
    osc1_escape_segments: tuple[Segment, ...],
    osc2_points: dict[str, Point],
    osc2_body_segments: tuple[Segment, ...],
) -> None:
    """Check OSC2 fanout against existing through-vias and shared VIN routing."""
    if osc2_points["X"] != osc1_points["U"]:
        raise ValueError("OSC2 must terminate at OSC1's shared VIN via U.")

    minimum_pad_distance = cfg["via_diameter_mm"] + cfg["trace_spacing_mm"]
    for osc1_pad in ("A", "V"):
        if distance(osc2_points["A"], osc1_points[osc1_pad]) < minimum_pad_distance:
            raise ValueError(f"OSC2 A via violates clearance to OSC1 {osc1_pad} via.")

    minimum_trace_distance = osc1_via_trace_clearance(cfg)
    for segment in osc2_body_segments:
        if (
            point_to_segment_distance(osc1_points["A"], segment) + GEOMETRY_TOLERANCE_MM
            < minimum_trace_distance
        ):
            raise ValueError("OSC2 entry copper violates clearance to OSC1 A via.")
        if (
            point_to_segment_distance(osc1_points["V"], segment) + GEOMETRY_TOLERANCE_MM
            < minimum_trace_distance
        ):
            raise ValueError("OSC2 copper violates clearance to OSC1 V via.")
    for segment in osc1_body_segments + osc1_escape_segments:
        if point_to_segment_distance(osc2_points["A"], segment) < minimum_trace_distance:
            raise ValueError("OSC2 A via violates clearance to OSC1 copper.")


def build_primary_geometry(cfg: dict | None = None) -> PrimaryGeometry:
    """Return point-driven primary geometry and the shared VIN escape."""
    cfg = build_config() if cfg is None else cfg
    dimensions = calculate_dimensions(cfg)
    validate_config(cfg, dimensions)
    osc1_points = build_osc1_point_map(cfg, dimensions)
    osc1_body_segments, osc1_escape_segments = build_osc1_segments(cfg, osc1_points)
    validate_osc1_clearance(cfg, osc1_points, osc1_body_segments, osc1_escape_segments)
    osc1_layer, osc2_layer = primary_layers(cfg)
    coils: list[PrimaryCoil] = [
        PrimaryCoil(
            "OSC1",
            osc1_layer,
            target_facing_layer(cfg),
            osc1_points,
            osc1_body_segments,
            osc1_escape_segments,
        )
    ]
    pads = {"OSC1_A": osc1_points["A"], "VIN_U": osc1_points["U"], "VIN_V": osc1_points["V"]}

    if cfg["generate_osc2"]:
        osc2_points = build_osc2_point_map(cfg, osc1_points)
        osc2_body_segments = build_osc2_segments(cfg, osc2_points)
        validate_osc2_clearance(
            cfg,
            osc1_points,
            osc1_body_segments,
            osc1_escape_segments,
            osc2_points,
            osc2_body_segments,
        )
        coils.append(
            PrimaryCoil("OSC2", osc2_layer, osc2_layer, osc2_points, osc2_body_segments, ())
        )
        pads["OSC2_A"] = osc2_points["A"]

    return PrimaryGeometry(
        dimensions=dimensions,
        pads=pads,
        coils=tuple(coils),
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

    if cfg["generate_osc1"]:
        coil = geometry.coils[0]
        sections.append(
            pad_thru_hole(
                cfg["osc1_output_pad_name"],
                geometry.pads["OSC1_A"],
                cfg["via_diameter_mm"],
                cfg["via_hole_size_mm"],
            )
        )
        for segment in coil.body_segments:
            sections.append(fp_line(segment[0], segment[1], cfg["trace_width_mm"], coil.layer))
        sections.append(
            pad_thru_hole(
                cfg["primary_input_pad_name"],
                geometry.pads["VIN_U"],
                cfg["via_diameter_mm"],
                cfg["via_hole_size_mm"],
            )
        )
        for segment in coil.escape_segments:
            sections.append(
                fp_line(segment[0], segment[1], cfg["trace_width_mm"], coil.escape_layer)
            )
        sections.append(
            pad_thru_hole(
                cfg["primary_input_pad_name"],
                geometry.pads["VIN_V"],
                cfg["via_diameter_mm"],
                cfg["via_hole_size_mm"],
            )
        )

    if cfg["generate_osc2"]:
        coil = next(coil for coil in geometry.coils if coil.name == "OSC2")
        sections.append(
            pad_thru_hole(
                cfg["osc2_output_pad_name"],
                geometry.pads["OSC2_A"],
                cfg["via_diameter_mm"],
                cfg["via_hole_size_mm"],
            )
        )
        for segment in coil.body_segments:
            sections.append(fp_line(segment[0], segment[1], cfg["trace_width_mm"], coil.layer))

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
