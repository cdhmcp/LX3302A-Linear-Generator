#!/usr/bin/env python3
"""KiCad footprint generator for LX3302A linear sensor coil layouts.

OSC1 and OSC2 are built from their annotated primary-coil point maps. Receiver
coil CL2 is built from its annotated two-turn sinusoidal point map.
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
    "limit_before_mm": 0.5,
    "limit_after_mm": 0.5,
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
    "cl2_output_pad_name": "CL2",
    "cl2_return_pad_name": "CL2-GND",
    "generate_osc1": True,
    "generate_osc2": True,
    "generate_cl2": True,

    # CL2 is mapped for two turns in the current reference drawing.
    "number_of_secondary_turns": 2,
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
    "secondary_curve_samples_per_cycle": 128,
    "secondary_jump_runup_via_multiplier": 3.0,
    "secondary_jump_detour_via_multiplier": 0.35,
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


@dataclass(frozen=True)
class SecondaryCoil:
    """One receiver winding routed across its target-facing and inner layers."""

    name: str
    target_layer: str
    inner_layer: str
    stroke_length_mm: float
    points: dict[str, Point]
    target_segments: tuple[Segment, ...]
    inner_segments: tuple[Segment, ...]
    via_labels: tuple[str, ...]


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


def receiver_layers(cfg: dict) -> tuple[str, str]:
    """Return ``(target-facing, inner)`` copper layers used by secondary coils."""
    if cfg["target_side"] == "top":
        return "F.Cu", "In1.Cu"
    if cfg["target_side"] == "bottom":
        return "B.Cu", "In2.Cu"
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


def segment_to_segment_distance(first: Segment, second: Segment) -> float:
    """Return the minimum distance between two non-crossing copper segments."""
    return min(
        point_to_segment_distance(first[0], second),
        point_to_segment_distance(first[1], second),
        point_to_segment_distance(second[0], first),
        point_to_segment_distance(second[1], first),
    )


def path_to_path_distance(first: tuple[Segment, ...], second: tuple[Segment, ...]) -> float:
    """Return the closest segment distance between two sampled paths."""
    return min(
        segment_to_segment_distance(first_segment, second_segment)
        for first_segment in first
        for second_segment in second
    )


def cl2_stroke_length(cfg: dict) -> float:
    """Return the active CL2 waveform span from the mapped stroke definition."""
    return cfg["measurement_range_mm"] + cfg["target_x_mm"]


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
        "secondary_jump_runup_via_multiplier",
        "secondary_jump_detour_via_multiplier",
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
    if not isinstance(cfg["generate_cl2"], bool):
        raise ValueError("generate_cl2 must be a boolean.")
    if cfg["generate_osc2"] and not cfg["generate_osc1"]:
        raise ValueError("OSC2 requires OSC1 because it shares OSC1's VIN transition via.")
    if (
        not isinstance(cfg["number_of_secondary_turns"], int)
        or cfg["number_of_secondary_turns"] != 2
    ):
        raise ValueError("CL2 currently supports exactly two secondary turns.")
    if (
        not isinstance(cfg["secondary_curve_samples_per_cycle"], int)
        or cfg["secondary_curve_samples_per_cycle"] < 16
    ):
        raise ValueError("secondary_curve_samples_per_cycle must be an integer >= 16.")

    primary_layers(cfg)
    receiver_layers(cfg)
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


def secondary_wave_value_and_slope(
    cfg: dict,
    dimensions: SensorDimensions,
    x: float,
    phase_sign: float,
) -> tuple[float, float]:
    """Return the shared CL2 sinusoid centerline and its slope at ``x``."""
    span = cl2_stroke_length(cfg)
    amplitude = (dimensions.secondary_width_mm / 2.0) - (trace_pitch(cfg) / 2.0)
    angle = (2.0 * math.pi * (x + (span / 2.0))) / span
    return (
        phase_sign * amplitude * math.sin(angle),
        phase_sign * amplitude * (2.0 * math.pi / span) * math.cos(angle),
    )


def secondary_rail_point(
    cfg: dict,
    dimensions: SensorDimensions,
    station_x: float,
    phase_sign: float,
    rail_offset: float,
) -> Point:
    """Offset one CL2 waveform rail perpendicular to the common sine centerline."""
    y, slope = secondary_wave_value_and_slope(cfg, dimensions, station_x, phase_sign)
    normal_scale = math.hypot(slope, 1.0)
    return (
        station_x - ((slope / normal_scale) * rail_offset),
        y + (rail_offset / normal_scale),
    )


def secondary_curve_segments(
    cfg: dict,
    dimensions: SensorDimensions,
    start: Point,
    end: Point,
    phase_sign: float,
    rail_offset: float,
    reference_start: Point | None = None,
    reference_end: Point | None = None,
    station_start_x: float | None = None,
    station_end_x: float | None = None,
) -> tuple[Segment, ...]:
    """Sample part of a full-span sine rail while connecting mapped transition points."""
    stroke_length = cl2_stroke_length(cfg)
    effective_phase = phase_sign * -fanout_direction(cfg)
    reference_start = start if reference_start is None else reference_start
    reference_end = end if reference_end is None else reference_end
    station_start_x = start[0] if station_start_x is None else station_start_x
    station_end_x = end[0] if station_end_x is None else station_end_x
    sample_count = max(
        2,
        round(
            cfg["secondary_curve_samples_per_cycle"]
            * abs(station_end_x - station_start_x)
            / stroke_length
        ),
    )
    raw_start = secondary_rail_point(
        cfg, dimensions, reference_start[0], effective_phase, rail_offset
    )
    raw_end = secondary_rail_point(
        cfg, dimensions, reference_end[0], effective_phase, rail_offset
    )
    points: list[Point] = []
    for index in range(sample_count + 1):
        fraction = index / sample_count
        station_x = station_start_x + ((station_end_x - station_start_x) * fraction)
        reference_fraction = (station_x - reference_start[0]) / (
            reference_end[0] - reference_start[0]
        )
        raw_point = secondary_rail_point(
            cfg, dimensions, station_x, effective_phase, rail_offset
        )
        points.append(
            (
                raw_point[0]
                + ((reference_start[0] - raw_start[0]) * (1.0 - reference_fraction))
                + ((reference_end[0] - raw_end[0]) * reference_fraction),
                raw_point[1]
                + ((reference_start[1] - raw_start[1]) * (1.0 - reference_fraction))
                + ((reference_end[1] - raw_end[1]) * reference_fraction),
            )
        )
    return tuple(zip(points, points[1:]))


def secondary_corrected_rail_point(
    cfg: dict,
    dimensions: SensorDimensions,
    station_x: float,
    phase_sign: float,
    rail_offset: float,
    reference_start: Point,
    reference_end: Point,
) -> Point:
    """Return a point on a full corrected rail before fanout-side mirroring."""
    raw_start = secondary_rail_point(
        cfg, dimensions, reference_start[0], phase_sign, rail_offset
    )
    raw_end = secondary_rail_point(
        cfg, dimensions, reference_end[0], phase_sign, rail_offset
    )
    raw_point = secondary_rail_point(cfg, dimensions, station_x, phase_sign, rail_offset)
    fraction = (station_x - reference_start[0]) / (reference_end[0] - reference_start[0])
    return (
        raw_point[0]
        + ((reference_start[0] - raw_start[0]) * (1.0 - fraction))
        + ((reference_end[0] - raw_end[0]) * fraction),
        raw_point[1]
        + ((reference_start[1] - raw_start[1]) * (1.0 - fraction))
        + ((reference_end[1] - raw_end[1]) * fraction),
    )


def build_cl2_point_map(cfg: dict, dimensions: SensorDimensions) -> dict[str, Point]:
    """Construct the annotated two-turn CL2 point map in a left-entry frame."""
    half_span = cl2_stroke_length(cfg) / 2.0
    quarter_span = half_span / 2.0
    amplitude = dimensions.secondary_width_mm / 2.0
    pitch = trace_pitch(cfg)
    via_pair_spacing = cfg["via_diameter_mm"] + cfg["trace_spacing_mm"]
    runup = cfg["secondary_jump_runup_via_multiplier"] * cfg["via_diameter_mm"]
    detour = cfg["secondary_jump_detour_via_multiplier"] * cfg["via_diameter_mm"]
    terminal_x = -(dimensions.primary_length_mm / 2.0) - cfg["terminal_escape_length_mm"]
    terminal_fanout = 3.0 * via_pair_spacing
    _, midpoint_slope = secondary_wave_value_and_slope(cfg, dimensions, -half_span, -1.0)
    midpoint_horizontal_spacing = (
        pitch * math.hypot(midpoint_slope, 1.0) / abs(midpoint_slope)
    )

    points: dict[str, Point] = {
        # Provisional IC-side fanout, kept outside the primary boundary.
        "A": (terminal_x, terminal_fanout),
        "B": (terminal_x + terminal_fanout, 0.0),
        "C": (-half_span, 0.0),
        # First forward pass.
        "D": (-quarter_span + (via_pair_spacing / 2.0), -amplitude),
        "E": (-quarter_span + (via_pair_spacing / 2.0), -amplitude + via_pair_spacing),
        "F": (-quarter_span + (via_pair_spacing / 2.0), -amplitude + pitch),
        "G": (quarter_span + (via_pair_spacing / 2.0), amplitude),
        "H": (quarter_span + (via_pair_spacing / 2.0), amplitude - via_pair_spacing),
        "I": (quarter_span + (via_pair_spacing / 2.0), amplitude - pitch),
        "J": (half_span, -(pitch / 2.0)),
        # First reverse pass.
        "N": (quarter_span - (via_pair_spacing / 2.0), -amplitude),
        "O": (quarter_span - (via_pair_spacing / 2.0), -amplitude + via_pair_spacing),
        "P": (quarter_span - (via_pair_spacing / 2.0), -amplitude + pitch),
        "Q": (-quarter_span - (via_pair_spacing / 2.0), amplitude),
        "R": (-quarter_span - (via_pair_spacing / 2.0), amplitude - via_pair_spacing),
        "S": (-quarter_span - (via_pair_spacing / 2.0), amplitude - pitch),
        "W": (-half_span + midpoint_horizontal_spacing, 0.0),
        # Second forward pass.
        "X": (-quarter_span - (via_pair_spacing / 2.0), -amplitude + pitch),
        "Y": (-quarter_span - (via_pair_spacing / 2.0), -amplitude + via_pair_spacing),
        "Z": (-quarter_span - (via_pair_spacing / 2.0), -amplitude),
        "ZA": (quarter_span - (via_pair_spacing / 2.0), amplitude - pitch),
        "ZB": (quarter_span - (via_pair_spacing / 2.0), amplitude - via_pair_spacing),
        "ZC": (quarter_span - (via_pair_spacing / 2.0), amplitude),
        "ZG": (half_span, pitch / 2.0),
        # Second reverse pass and terminal escape.
        "ZH": (quarter_span + (via_pair_spacing / 2.0), -amplitude + pitch),
        "ZI": (quarter_span + (via_pair_spacing / 2.0), -amplitude + via_pair_spacing),
        "ZJ": (quarter_span + (via_pair_spacing / 2.0), -amplitude),
        "ZK": (-quarter_span + (via_pair_spacing / 2.0), amplitude - pitch),
        "ZL": (-quarter_span + (via_pair_spacing / 2.0), amplitude - via_pair_spacing),
        "ZM": (-quarter_span + (via_pair_spacing / 2.0), amplitude),
        "ZN": (-half_span, 0.0),
        "ZO": (terminal_x + terminal_fanout, 0.0),
        "ZP": (terminal_x, -terminal_fanout),
    }

    points["K"] = secondary_rail_point(
        cfg, dimensions, points["J"][0] - runup, 1.0, -(pitch / 2.0)
    )
    points["L"] = (points["K"][0], points["K"][1] - detour)
    points["M"] = points["K"]

    t_station_x = points["W"][0] + runup
    points["T"] = secondary_corrected_rail_point(
        cfg,
        dimensions,
        t_station_x,
        1.0,
        -(pitch / 2.0),
        points["S"],
        points["W"],
    )
    points["U"] = (points["T"][0], points["T"][1] - detour)
    points["V"] = points["T"]

    points["ZD"] = secondary_rail_point(
        cfg, dimensions, points["ZG"][0] - runup, -1.0, pitch / 2.0
    )
    points["ZE"] = (points["ZD"][0], points["ZD"][1] + detour)
    points["ZF"] = points["ZD"]

    if fanout_direction(cfg) > 0:
        points = {label: (-point[0], point[1]) for label, point in points.items()}
    return points


def build_cl2_segments(
    cfg: dict,
    dimensions: SensorDimensions,
    points: dict[str, Point],
) -> tuple[tuple[Segment, ...], tuple[Segment, ...]]:
    """Return CL2 copper segments assigned to its two receiver layers."""
    target_segments: list[Segment] = []
    inner_segments: list[Segment] = []

    def line(collection: list[Segment], start: str, end: str) -> None:
        collection.append((points[start], points[end]))

    half_pitch = trace_pitch(cfg) / 2.0

    def curve(
        collection: list[Segment],
        start: str,
        end: str,
        phase_sign: float,
        rail_offset: float,
        reference: tuple[str, str] | None = None,
        stations: tuple[float, float] | None = None,
    ) -> None:
        reference_start = None if reference is None else points[reference[0]]
        reference_end = None if reference is None else points[reference[1]]
        collection.extend(
            secondary_curve_segments(
                cfg,
                dimensions,
                points[start],
                points[end],
                phase_sign,
                rail_offset,
                reference_start,
                reference_end,
                None if stations is None else stations[0],
                None if stations is None else stations[1],
            )
        )

    line(target_segments, "A", "B")
    line(target_segments, "B", "C")
    curve(target_segments, "C", "D", -1.0, -half_pitch)
    line(target_segments, "D", "E")
    line(inner_segments, "E", "F")
    curve(inner_segments, "F", "G", -1.0, half_pitch)
    line(inner_segments, "G", "H")
    line(target_segments, "H", "I")
    curve(target_segments, "I", "J", -1.0, -half_pitch)
    curve(target_segments, "J", "K", 1.0, -half_pitch)
    line(target_segments, "K", "L")
    line(inner_segments, "L", "M")
    curve(inner_segments, "M", "N", 1.0, -half_pitch)
    line(inner_segments, "N", "O")
    line(target_segments, "O", "P")
    curve(target_segments, "P", "Q", 1.0, half_pitch)
    line(target_segments, "Q", "R")
    line(inner_segments, "R", "S")
    u_transition_station_x = (
        points["W"][0]
        - (
            fanout_direction(cfg)
            * cfg["secondary_jump_runup_via_multiplier"]
            * cfg["via_diameter_mm"]
        )
    )
    curve(
        inner_segments,
        "S",
        "T",
        1.0,
        -half_pitch,
        ("S", "W"),
        (points["S"][0], u_transition_station_x),
    )
    line(inner_segments, "T", "U")
    line(target_segments, "U", "V")
    curve(
        target_segments,
        "V",
        "W",
        1.0,
        -half_pitch,
        ("S", "W"),
        (u_transition_station_x, points["W"][0]),
    )
    curve(target_segments, "W", "X", -1.0, half_pitch)
    line(target_segments, "X", "Y")
    line(inner_segments, "Y", "Z")
    curve(inner_segments, "Z", "ZA", -1.0, -half_pitch)
    line(inner_segments, "ZA", "ZB")
    line(target_segments, "ZB", "ZC")
    curve(target_segments, "ZC", "ZD", -1.0, half_pitch)
    line(target_segments, "ZD", "ZE")
    line(inner_segments, "ZE", "ZF")
    curve(inner_segments, "ZF", "ZG", -1.0, half_pitch)
    curve(inner_segments, "ZG", "ZH", 1.0, half_pitch)
    line(inner_segments, "ZH", "ZI")
    line(target_segments, "ZI", "ZJ")
    curve(target_segments, "ZJ", "ZK", 1.0, -half_pitch)
    line(target_segments, "ZK", "ZL")
    line(inner_segments, "ZL", "ZM")
    curve(inner_segments, "ZM", "ZN", 1.0, half_pitch)
    line(inner_segments, "ZN", "ZO")
    line(inner_segments, "ZO", "ZP")
    return tuple(target_segments), tuple(inner_segments)


def validate_cl2_clearance(
    cfg: dict,
    dimensions: SensorDimensions,
    primary_geometry: PrimaryGeometry,
    points: dict[str, Point],
) -> None:
    """Validate mapped CL2 terminal and paired-via clearance constraints."""
    minimum_pad_distance = cfg["via_diameter_mm"] + cfg["trace_spacing_mm"]
    for first, second in (("E", "Y"), ("H", "ZB"), ("O", "ZI"), ("R", "ZL")):
        if distance(points[first], points[second]) + GEOMETRY_TOLERANCE_MM < minimum_pad_distance:
            raise ValueError(f"CL2 paired vias {first}/{second} violate plated via clearance.")

    primary_pads = tuple(primary_geometry.pads.values())
    for terminal in ("A", "ZP"):
        for primary_pad in primary_pads:
            if distance(points[terminal], primary_pad) + GEOMETRY_TOLERANCE_MM < minimum_pad_distance:
                raise ValueError(f"CL2 terminal {terminal} collides with a primary via.")

    pitch = trace_pitch(cfg)
    half_pitch = pitch / 2.0
    polygonal_tolerance = 0.001
    parallel_paths = (
        (
            secondary_curve_segments(cfg, dimensions, points["F"], points["G"], -1.0, half_pitch),
            secondary_curve_segments(cfg, dimensions, points["Z"], points["ZA"], -1.0, -half_pitch),
        ),
        (
            secondary_curve_segments(cfg, dimensions, points["P"], points["Q"], 1.0, half_pitch),
            secondary_curve_segments(cfg, dimensions, points["ZJ"], points["ZK"], 1.0, -half_pitch),
        ),
    )
    for first, second in parallel_paths:
        if path_to_path_distance(first, second) + polygonal_tolerance < pitch:
            raise ValueError("CL2 parallel sinusoidal traces violate configured spacing.")


def build_cl2_geometry(
    cfg: dict | None = None,
    primary_geometry: PrimaryGeometry | None = None,
) -> SecondaryCoil | None:
    """Build the two-turn CL2 receiver coil, or return ``None`` when disabled."""
    cfg = build_config() if cfg is None else cfg
    if not cfg["generate_cl2"]:
        return None
    dimensions = calculate_dimensions(cfg)
    validate_config(cfg, dimensions)
    primary_geometry = primary_geometry or build_primary_geometry(cfg)
    points = build_cl2_point_map(cfg, dimensions)
    target_segments, inner_segments = build_cl2_segments(cfg, dimensions, points)
    validate_cl2_clearance(cfg, dimensions, primary_geometry, points)
    return SecondaryCoil(
        name="CL2",
        target_layer=receiver_layers(cfg)[0],
        inner_layer=receiver_layers(cfg)[1],
        stroke_length_mm=cl2_stroke_length(cfg),
        points=points,
        target_segments=target_segments,
        inner_segments=inner_segments,
        via_labels=("A", "E", "H", "L", "O", "R", "U", "Y", "ZB", "ZE", "ZI", "ZL", "ZP"),
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
    """Render the configured KiCad sensor footprint text."""
    cfg = build_config() if cfg is None else cfg
    geometry = build_primary_geometry(cfg)
    cl2_geometry = build_cl2_geometry(cfg, geometry)
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

    if cl2_geometry is not None:
        for segment in cl2_geometry.target_segments:
            sections.append(
                fp_line(segment[0], segment[1], cfg["trace_width_mm"], cl2_geometry.target_layer)
            )
        for segment in cl2_geometry.inner_segments:
            sections.append(
                fp_line(segment[0], segment[1], cfg["trace_width_mm"], cl2_geometry.inner_layer)
            )
        for label in cl2_geometry.via_labels:
            pad_name = {
                "A": cfg["cl2_output_pad_name"],
                "ZP": cfg["cl2_return_pad_name"],
            }.get(label, f"CL2_{label}")
            sections.append(
                pad_thru_hole(
                    pad_name,
                    cl2_geometry.points[label],
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
    cl2_geometry = build_cl2_geometry(cfg, geometry)
    output_path = write_linear_sensor_footprint(cfg)
    dims = geometry.dimensions
    print(f"Wrote {output_path}")
    print(
        "Primary outer centerline envelope: "
        f"{dims.primary_length_mm:.3f} mm x {dims.primary_width_mm:.3f} mm"
    )
    if cl2_geometry is not None:
        print(f"CL2 active waveform span: {cl2_geometry.stroke_length_mm:.3f} mm")


if __name__ == "__main__":
    main()
