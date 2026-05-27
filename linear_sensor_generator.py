#!/usr/bin/env python3
"""KiCad footprint generator for LX3302A linear sensor coil layouts.

OSC1 and OSC2 are built from their annotated primary-coil point maps. Receiver
coils CL1 and CL2 are built from their annotated two-turn sinusoidal point maps.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path


MIL_TO_MM = 0.0254
GEOMETRY_TOLERANCE_MM = 1e-9

Point = tuple[float, float]
Segment = tuple[Point, Point]
Arc = tuple[Point, Point, Point]


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
    "primary_y_margin_mm": 0.075,
    "number_of_primary_turns": 3,

    # Placement and output
    "target_side": "top",
    "fanout_side": "left",
    "output_dir": "InductiveSensors.pretty",
    "footprint_name": "LX3302A_LINEAR_SENSOR_COILS",
    "reference_text": "REF**",
    "primary_input_pad_name": "VIN",
    "osc1_output_pad_name": "OSC1",
    "osc2_output_pad_name": "OSC2",
    "cl2_output_pad_name": "CL2",
    "cl2_return_pad_name": "CL2-GND",
    "cl1_output_pad_name": "CL1",
    "cl1_return_pad_name": "CL1-GND",
    "generate_osc1": True,
    "generate_osc2": True,
    "generate_cl2": True,
    "generate_cl1": True,

    # The documented secondary point maps currently cover two turns.
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
    # Recommended CL1 transition-column range: 0.02 to 0.05.
    "cl1_transition_column_fraction": 0.03,
    "cl1_primary_end_min_clearance_mm": 1.0,
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


@dataclass(frozen=True)
class CL1Coil:
    """CL1 receiver winding routed across receiver and crossover layers."""

    name: str
    target_layer: str
    inner_layer: str
    crossover_layer: str
    stroke_length_mm: float
    points: dict[str, Point]
    target_segments: tuple[Segment, ...]
    inner_segments: tuple[Segment, ...]
    crossover_segments: tuple[Segment, ...]
    target_arcs: tuple[Arc, ...]
    inner_arcs: tuple[Arc, ...]
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


def receiver_crossover_layer(cfg: dict) -> str:
    """Return the OSC2 layer intentionally used for short CL1 crossovers."""
    return primary_layers(cfg)[1]


def fanout_direction(cfg: dict) -> float:
    """Return -1 for a left breakout or +1 for a right breakout."""
    if cfg["fanout_side"] == "left":
        return -1.0
    if cfg["fanout_side"] == "right":
        return 1.0
    raise ValueError("fanout_side must be 'left' or 'right'.")


def trace_pitch(cfg: dict) -> float:
    return cfg["trace_width_mm"] + cfg["trace_spacing_mm"]


def terminal_pad_pitch(cfg: dict) -> float:
    """Return center spacing for adjacent external through-via pads."""
    return cfg["via_diameter_mm"] + cfg["trace_spacing_mm"]


def terminal_column_x(cfg: dict, dimensions: SensorDimensions) -> float:
    """Return the common fanout column for all external terminal vias."""
    return fanout_direction(cfg) * (
        (dimensions.primary_length_mm / 2.0) + cfg["terminal_escape_length_mm"]
    )


def terminal_row_y(cfg: dict, pad_name: str) -> float:
    """Return a compact terminal row with CL1 between VIN and OSC1."""
    row_index = {
        "CL1-GND": -4,
        "CL2-GND": -3,
        "VIN": -2,
        "CL1": -1,
        "OSC1": 0,
        "OSC2": 1,
        "CL2": 2,
    }[pad_name]
    return row_index * terminal_pad_pitch(cfg)


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


def cross_product(first: Point, second: Point, third: Point) -> float:
    """Return the signed cross product for the turn from first to third."""
    return (
        ((second[0] - first[0]) * (third[1] - first[1]))
        - ((second[1] - first[1]) * (third[0] - first[0]))
    )


def point_on_segment(point: Point, segment: Segment) -> bool:
    """Return whether a collinear point lies on a finite segment."""
    start, end = segment
    return (
        abs(cross_product(start, end, point)) <= GEOMETRY_TOLERANCE_MM
        and min(start[0], end[0]) - GEOMETRY_TOLERANCE_MM
        <= point[0]
        <= max(start[0], end[0]) + GEOMETRY_TOLERANCE_MM
        and min(start[1], end[1]) - GEOMETRY_TOLERANCE_MM
        <= point[1]
        <= max(start[1], end[1]) + GEOMETRY_TOLERANCE_MM
    )


def segments_intersect(first: Segment, second: Segment) -> bool:
    """Return whether two finite segments touch or cross."""
    first_start, first_end = first
    second_start, second_end = second
    turns = (
        cross_product(first_start, first_end, second_start),
        cross_product(first_start, first_end, second_end),
        cross_product(second_start, second_end, first_start),
        cross_product(second_start, second_end, first_end),
    )
    if (
        ((turns[0] > GEOMETRY_TOLERANCE_MM and turns[1] < -GEOMETRY_TOLERANCE_MM)
         or (turns[0] < -GEOMETRY_TOLERANCE_MM and turns[1] > GEOMETRY_TOLERANCE_MM))
        and
        ((turns[2] > GEOMETRY_TOLERANCE_MM and turns[3] < -GEOMETRY_TOLERANCE_MM)
         or (turns[2] < -GEOMETRY_TOLERANCE_MM and turns[3] > GEOMETRY_TOLERANCE_MM))
    ):
        return True
    return (
        (abs(turns[0]) <= GEOMETRY_TOLERANCE_MM and point_on_segment(second_start, first))
        or (abs(turns[1]) <= GEOMETRY_TOLERANCE_MM and point_on_segment(second_end, first))
        or (abs(turns[2]) <= GEOMETRY_TOLERANCE_MM and point_on_segment(first_start, second))
        or (abs(turns[3]) <= GEOMETRY_TOLERANCE_MM and point_on_segment(first_end, second))
    )


def segment_to_segment_distance(first: Segment, second: Segment) -> float:
    """Return the minimum distance between two finite copper segments."""
    if segments_intersect(first, second):
        return 0.0
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


def secondary_stroke_length(cfg: dict) -> float:
    """Return the active waveform span shared by the two secondary coils."""
    return cfg["measurement_range_mm"] + cfg["target_x_mm"]


def primary_inner_half_height(cfg: dict, dimensions: SensorDimensions) -> float:
    """Return the innermost primary horizontal centerline distance from center."""
    return (dimensions.primary_width_mm / 2.0) - (
        (cfg["number_of_primary_turns"] - 1) * trace_pitch(cfg)
    )


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
        "cl1_primary_end_min_clearance_mm",
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
    if not isinstance(cfg["generate_cl1"], bool):
        raise ValueError("generate_cl1 must be a boolean.")
    if cfg["generate_osc2"] and not cfg["generate_osc1"]:
        raise ValueError("OSC2 requires OSC1 because it shares OSC1's VIN transition via.")
    if (
        not isinstance(cfg["number_of_secondary_turns"], int)
        or cfg["number_of_secondary_turns"] != 2
    ):
        raise ValueError("Secondary receiver coils currently support exactly two secondary turns.")
    if (
        not isinstance(cfg["secondary_curve_samples_per_cycle"], int)
        or cfg["secondary_curve_samples_per_cycle"] < 16
    ):
        raise ValueError("secondary_curve_samples_per_cycle must be an integer >= 16.")
    if not 0.0 < cfg["cl1_transition_column_fraction"] < 0.5:
        raise ValueError("cl1_transition_column_fraction must be between 0 and 0.5.")

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
    terminal_x = terminal_column_x(cfg, dimensions)
    points: dict[str, Point] = {
        "A": (terminal_x, terminal_row_y(cfg, "OSC1")),
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
    points["VIN_JOG"] = (
        terminal_x - (side * abs(terminal_row_y(cfg, "VIN") - via_y)),
        via_y,
    )
    points["V"] = (terminal_x, terminal_row_y(cfg, "VIN"))
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
    escape_segments = (
        (points["U"], points["VIN_JOG"]),
        (points["VIN_JOG"], points["V"]),
    )
    return body, escape_segments


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
    pad_clearance = terminal_pad_pitch(cfg)
    turn_count = cfg["number_of_primary_turns"]
    outer_x = osc1_points[osc1_turn_labels(0)[1]][0]

    # Leave the terminal column horizontally, then use one 45 degree jog into
    # the midpoint entry after clearing the adjacent OSC1 terminal via.
    a_jog_x = osc1_points["A"][0] - (side * via_clearance)
    b_x = a_jog_x - (side * pad_clearance)
    points: dict[str, Point] = {
        "A": (osc1_points["A"][0], terminal_row_y(cfg, "OSC2")),
        "A_JOG": (a_jog_x, terminal_row_y(cfg, "OSC2")),
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
    point_sequence = ["A", "A_JOG", "B", "C", "D", "E", "F"]
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
            "osc1_vin_exit_offset_mm is too small for OSC1/VIN "
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
    for segment in escape_segments:
        if point_to_segment_distance(points["A"], segment) < minimum_trace_distance:
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
    phase_offset_radians: float = 0.0,
) -> tuple[float, float]:
    """Return one secondary sinusoid centerline and its slope at ``x``."""
    span = secondary_stroke_length(cfg)
    amplitude = (dimensions.secondary_width_mm / 2.0) - (trace_pitch(cfg) / 2.0)
    angle = ((2.0 * math.pi * (x + (span / 2.0))) / span) + phase_offset_radians
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
    phase_offset_radians: float = 0.0,
) -> Point:
    """Offset one waveform rail perpendicular to its secondary centerline."""
    y, slope = secondary_wave_value_and_slope(
        cfg, dimensions, station_x, phase_sign, phase_offset_radians
    )
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
    phase_offset_radians: float = 0.0,
    mirror_phase_sign: bool = True,
) -> tuple[Segment, ...]:
    """Sample part of a full-span sine rail while connecting mapped transition points."""
    stroke_length = secondary_stroke_length(cfg)
    effective_phase = phase_sign * (-fanout_direction(cfg) if mirror_phase_sign else 1.0)
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
        cfg,
        dimensions,
        reference_start[0],
        effective_phase,
        rail_offset,
        phase_offset_radians,
    )
    raw_end = secondary_rail_point(
        cfg,
        dimensions,
        reference_end[0],
        effective_phase,
        rail_offset,
        phase_offset_radians,
    )
    points: list[Point] = []
    for index in range(sample_count + 1):
        fraction = index / sample_count
        station_x = station_start_x + ((station_end_x - station_start_x) * fraction)
        reference_fraction = (station_x - reference_start[0]) / (
            reference_end[0] - reference_start[0]
        )
        raw_point = secondary_rail_point(
            cfg,
            dimensions,
            station_x,
            effective_phase,
            rail_offset,
            phase_offset_radians,
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
    phase_offset_radians: float = 0.0,
) -> Point:
    """Return a point on a full corrected rail before fanout-side mirroring."""
    raw_start = secondary_rail_point(
        cfg,
        dimensions,
        reference_start[0],
        phase_sign,
        rail_offset,
        phase_offset_radians,
    )
    raw_end = secondary_rail_point(
        cfg,
        dimensions,
        reference_end[0],
        phase_sign,
        rail_offset,
        phase_offset_radians,
    )
    raw_point = secondary_rail_point(
        cfg, dimensions, station_x, phase_sign, rail_offset, phase_offset_radians
    )
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
    half_span = secondary_stroke_length(cfg) / 2.0
    quarter_span = half_span / 2.0
    amplitude = dimensions.secondary_width_mm / 2.0
    pitch = trace_pitch(cfg)
    via_pair_spacing = cfg["via_diameter_mm"] + cfg["trace_spacing_mm"]
    inner_primary_y = primary_inner_half_height(cfg, dimensions)
    primary_via_clearance = osc1_via_trace_clearance(cfg)
    # These plated transition vias belong inside all primary turns, not
    # outside them. Move toward center from the innermost primary centerline.
    upper_via_y = -(inner_primary_y - primary_via_clearance)
    lower_via_y = inner_primary_y - primary_via_clearance
    runup = cfg["secondary_jump_runup_via_multiplier"] * cfg["via_diameter_mm"]
    detour = cfg["secondary_jump_detour_via_multiplier"] * cfg["via_diameter_mm"]
    terminal_x = -((dimensions.primary_length_mm / 2.0) + cfg["terminal_escape_length_mm"])
    terminal_output_y = terminal_row_y(cfg, "CL2")
    terminal_return_y = terminal_row_y(cfg, "CL2-GND")
    _, midpoint_slope = secondary_wave_value_and_slope(cfg, dimensions, -half_span, -1.0)
    midpoint_horizontal_spacing = (
        pitch * math.hypot(midpoint_slope, 1.0) / abs(midpoint_slope)
    )

    points: dict[str, Point] = {
        # Provisional IC-side fanout, kept outside the primary boundary.
        "A": (terminal_x, terminal_output_y),
        "B": (terminal_x - (fanout_direction(cfg) * terminal_output_y), 0.0),
        "C": (-half_span, 0.0),
        # First forward pass.
        "D": (-quarter_span + (via_pair_spacing / 2.0), -amplitude),
        "E": (-quarter_span + (via_pair_spacing / 2.0), upper_via_y),
        "F": (-quarter_span + (via_pair_spacing / 2.0), -amplitude + pitch),
        "G": (quarter_span + (via_pair_spacing / 2.0), amplitude),
        "H": (quarter_span + (via_pair_spacing / 2.0), lower_via_y),
        "I": (quarter_span + (via_pair_spacing / 2.0), amplitude - pitch),
        "J": (half_span, -(pitch / 2.0)),
        # First reverse pass.
        "N": (quarter_span - (via_pair_spacing / 2.0), -amplitude),
        "O": (quarter_span - (via_pair_spacing / 2.0), upper_via_y),
        "P": (quarter_span - (via_pair_spacing / 2.0), -amplitude + pitch),
        "Q": (-quarter_span - (via_pair_spacing / 2.0), amplitude),
        "R": (-quarter_span - (via_pair_spacing / 2.0), lower_via_y),
        "S": (-quarter_span - (via_pair_spacing / 2.0), amplitude - pitch),
        "W": (-half_span + midpoint_horizontal_spacing, 0.0),
        # Second forward pass.
        "X": (-quarter_span - (via_pair_spacing / 2.0), -amplitude + pitch),
        "Y": (-quarter_span - (via_pair_spacing / 2.0), upper_via_y),
        "Z": (-quarter_span - (via_pair_spacing / 2.0), -amplitude),
        "ZA": (quarter_span - (via_pair_spacing / 2.0), amplitude - pitch),
        "ZB": (quarter_span - (via_pair_spacing / 2.0), lower_via_y),
        "ZC": (quarter_span - (via_pair_spacing / 2.0), amplitude),
        "ZG": (half_span, pitch / 2.0),
        # Second reverse pass and terminal escape.
        "ZH": (quarter_span + (via_pair_spacing / 2.0), -amplitude + pitch),
        "ZI": (quarter_span + (via_pair_spacing / 2.0), upper_via_y),
        "ZJ": (quarter_span + (via_pair_spacing / 2.0), -amplitude),
        "ZK": (-quarter_span + (via_pair_spacing / 2.0), amplitude - pitch),
        "ZL": (-quarter_span + (via_pair_spacing / 2.0), lower_via_y),
        "ZM": (-quarter_span + (via_pair_spacing / 2.0), amplitude),
        "ZN": (-half_span, 0.0),
        "ZO": (terminal_x - (fanout_direction(cfg) * abs(terminal_return_y)), 0.0),
        "ZP": (terminal_x, terminal_return_y),
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

    minimum_primary_trace_distance = osc1_via_trace_clearance(cfg)
    for via_label in ("E", "Y", "H", "ZB", "O", "ZI", "R", "ZL"):
        nearest_primary_trace = min(
            point_to_segment_distance(points[via_label], segment)
            for coil in primary_geometry.coils
            for segment in coil.body_segments
        )
        if (
            nearest_primary_trace + GEOMETRY_TOLERANCE_MM
            < minimum_primary_trace_distance
        ):
            raise ValueError(
                f"CL2 via {via_label} violates clearance to the primary winding."
            )

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
        stroke_length_mm=secondary_stroke_length(cfg),
        points=points,
        target_segments=target_segments,
        inner_segments=inner_segments,
        via_labels=("A", "E", "H", "L", "O", "R", "U", "Y", "ZB", "ZE", "ZI", "ZL", "ZP"),
    )


def build_cl1_point_map(cfg: dict, dimensions: SensorDimensions) -> dict[str, Point]:
    """Construct the annotated two-turn CL1 quadrature point map."""
    half_span = secondary_stroke_length(cfg) / 2.0
    amplitude = dimensions.secondary_width_mm / 2.0
    pitch = trace_pitch(cfg)
    half_pitch = pitch / 2.0
    via_spacing = cfg["via_diameter_mm"] + cfg["trace_spacing_mm"]
    via_clearance = osc1_via_trace_clearance(cfg)
    upper_via_y = -(primary_inner_half_height(cfg, dimensions) - via_clearance)
    lower_via_y = -upper_via_y
    left_x = -half_span
    right_x = half_span
    inner_right_x = right_x - pitch
    transition_x = left_x + (secondary_stroke_length(cfg) * cfg["cl1_transition_column_fraction"])
    midpoint_left_x = -(via_spacing / 2.0)
    midpoint_right_x = via_spacing / 2.0
    outer_top = -amplitude
    inner_top = outer_top + pitch
    inner_bottom = amplitude - pitch
    transition_inner_bottom = secondary_rail_point(
        cfg, dimensions, transition_x, 1.0, -(pitch / 2.0), math.pi / 2.0
    )[1]
    transition_inner_top = -transition_inner_bottom
    outer_bottom = amplitude
    end_column_delta = right_x - inner_right_x
    detour_column_y = math.sqrt(
        (via_clearance * via_clearance) - (end_column_delta * end_column_delta)
    )

    # CL1 is the straight-through row between the VIN and OSC1 terminals.
    entrance_y = terminal_row_y(cfg, "CL1")
    terminal_x = -((dimensions.primary_length_mm / 2.0) + cfg["terminal_escape_length_mm"])
    terminal_y = terminal_row_y(cfg, "CL1")
    return_terminal_y = terminal_row_y(cfg, "CL1-GND")
    entry_b_x = terminal_x - (fanout_direction(cfg) * cfg["terminal_escape_length_mm"])
    return_zm_x = terminal_x - (
        fanout_direction(cfg) * abs(return_terminal_y - entrance_y)
    )

    # K-L and ZC-ZD are compact semicircles centered on this column. Their
    # radius leaves one trace pitch to CL2 J/ZG at the adjacent end column.
    cl2_crossing_half_height = math.hypot(pitch, half_pitch) + pitch
    points: dict[str, Point] = {
        "A": (terminal_x, terminal_y),
        "B": (entry_b_x, entrance_y),
        "C": (left_x, entrance_y),
        "D": (left_x, lower_via_y),
        "E": (left_x, outer_bottom),
        "F": (midpoint_left_x, inner_top),
        "G": (midpoint_left_x, upper_via_y),
        "H": (midpoint_left_x, outer_top),
        "I": (inner_right_x, inner_bottom),
        "J": (inner_right_x, lower_via_y),
        "K": (inner_right_x, cl2_crossing_half_height),
        "L": (inner_right_x, -cl2_crossing_half_height),
        # M-N arcs around CL1 via ZE while retaining the mapped x columns.
        "M": (inner_right_x, upper_via_y + via_clearance),
        "N": (right_x, upper_via_y - detour_column_y),
        "O": (right_x, outer_top),
        "P": (midpoint_right_x, inner_bottom),
        "Q": (midpoint_right_x, lower_via_y),
        "R": (midpoint_right_x, outer_bottom),
        "S": (transition_x, transition_inner_top),
        "T": (transition_x, upper_via_y),
        "U": (transition_x, lower_via_y),
        "V": (transition_x, transition_inner_bottom),
        "W": (midpoint_right_x, outer_top),
        "X": (midpoint_right_x, upper_via_y),
        "Y": (midpoint_right_x, inner_top),
        "Z": (right_x, outer_bottom),
        # ZA-ZB is the analogous arc around CL1 via J.
        "ZA": (right_x, lower_via_y + detour_column_y),
        "ZB": (inner_right_x, lower_via_y - via_clearance),
        "ZC": (inner_right_x, cl2_crossing_half_height),
        "ZD": (inner_right_x, -cl2_crossing_half_height),
        "ZE": (inner_right_x, upper_via_y),
        "ZF": (inner_right_x, inner_top),
        "ZG": (midpoint_left_x, outer_bottom),
        "ZH": (midpoint_left_x, lower_via_y),
        "ZI": (midpoint_left_x, inner_bottom),
        "ZJ": (left_x, outer_top),
        "ZK": (left_x, entrance_y - via_clearance),
        "ZL": (left_x - via_clearance, entrance_y),
        "ZM": (return_zm_x, entrance_y),
        "ZN": (terminal_x, return_terminal_y),
    }
    if fanout_direction(cfg) > 0:
        points = {label: (-point[0], point[1]) for label, point in points.items()}
    return points


def via_clearance_arc(cfg: dict, start: Point, end: Point, center: Point) -> Arc:
    """Return an arc on the sensor-end side of a via at legal clearance."""
    radius = osc1_via_trace_clearance(cfg)
    sensor_end_direction = -fanout_direction(cfg)
    return (
        start,
        (center[0] + (sensor_end_direction * radius), center[1]),
        end,
    )


def outside_semicircle_arc(cfg: dict, start: Point, end: Point) -> Arc:
    """Return a sensor-end-facing semicircle centered on the endpoint column."""
    center_y = (start[1] + end[1]) / 2.0
    radius = abs(start[1] - center_y)
    sensor_end_direction = -fanout_direction(cfg)
    return (
        start,
        (start[0] + (sensor_end_direction * radius), center_y),
        end,
    )


def lower_fanout_via_arc(cfg: dict, start: Point, end: Point, center: Point) -> Arc:
    """Return the quarter circle from below a via toward its fanout side."""
    radius = osc1_via_trace_clearance(cfg)
    fanout_side = fanout_direction(cfg)
    diagonal_radius = radius / math.sqrt(2.0)
    return (
        start,
        (
            center[0] + (fanout_side * diagonal_radius),
            center[1] - diagonal_radius,
        ),
        end,
    )


def build_cl1_routes(
    cfg: dict,
    dimensions: SensorDimensions,
    points: dict[str, Point],
) -> tuple[
    tuple[Segment, ...],
    tuple[Segment, ...],
    tuple[Segment, ...],
    tuple[Arc, ...],
    tuple[Arc, ...],
]:
    """Return line and arc primitives for the corrected CL1 point sequence."""
    target_segments: list[Segment] = []
    inner_segments: list[Segment] = []
    crossover_segments: list[Segment] = []
    target_arcs: list[Arc] = []
    inner_arcs: list[Arc] = []
    half_pitch = trace_pitch(cfg) / 2.0
    phase_offset = math.pi / 2.0

    def line(collection: list[Segment], start: str, end: str) -> None:
        collection.append((points[start], points[end]))

    def curve(
        collection: list[Segment],
        start: str,
        end: str,
        phase_sign: float,
        rail_offset: float,
    ) -> None:
        collection.extend(
            secondary_curve_segments(
                cfg,
                dimensions,
                points[start],
                points[end],
                phase_sign,
                rail_offset,
                phase_offset_radians=phase_offset,
                mirror_phase_sign=False,
            )
        )

    line(target_segments, "A", "B")
    line(target_segments, "B", "C")
    line(crossover_segments, "C", "D")
    line(target_segments, "D", "E")
    curve(target_segments, "E", "F", 1.0, half_pitch)
    line(target_segments, "F", "G")
    line(inner_segments, "G", "H")
    curve(inner_segments, "H", "I", 1.0, -half_pitch)
    line(inner_segments, "I", "J")
    line(target_segments, "J", "K")
    target_arcs.append(outside_semicircle_arc(cfg, points["K"], points["L"]))
    line(target_segments, "L", "M")
    target_arcs.append(via_clearance_arc(cfg, points["M"], points["N"], points["ZE"]))
    line(target_segments, "N", "O")
    curve(target_segments, "O", "P", -1.0, -half_pitch)
    line(target_segments, "P", "Q")
    line(inner_segments, "Q", "R")
    curve(inner_segments, "R", "S", -1.0, half_pitch)
    line(inner_segments, "S", "T")
    line(crossover_segments, "T", "U")
    line(target_segments, "U", "V")
    curve(target_segments, "V", "W", 1.0, -half_pitch)
    line(target_segments, "W", "X")
    line(inner_segments, "X", "Y")
    curve(inner_segments, "Y", "Z", 1.0, half_pitch)
    line(inner_segments, "Z", "ZA")
    inner_arcs.append(via_clearance_arc(cfg, points["ZA"], points["ZB"], points["J"]))
    line(inner_segments, "ZB", "ZC")
    inner_arcs.append(outside_semicircle_arc(cfg, points["ZC"], points["ZD"]))
    line(inner_segments, "ZD", "ZE")
    line(target_segments, "ZE", "ZF")
    curve(target_segments, "ZF", "ZG", -1.0, half_pitch)
    line(target_segments, "ZG", "ZH")
    line(inner_segments, "ZH", "ZI")
    curve(inner_segments, "ZI", "ZJ", -1.0, -half_pitch)
    line(inner_segments, "ZJ", "ZK")
    inner_arcs.append(lower_fanout_via_arc(cfg, points["ZK"], points["ZL"], points["C"]))
    line(inner_segments, "ZL", "ZM")
    line(inner_segments, "ZM", "ZN")
    return (
        tuple(target_segments),
        tuple(inner_segments),
        tuple(crossover_segments),
        tuple(target_arcs),
        tuple(inner_arcs),
    )


def validate_cl1_clearance(
    cfg: dict,
    dimensions: SensorDimensions,
    primary_geometry: PrimaryGeometry,
    cl2_geometry: SecondaryCoil | None,
    points: dict[str, Point],
    target_segments: tuple[Segment, ...],
    inner_segments: tuple[Segment, ...],
    crossover_segments: tuple[Segment, ...],
    target_arcs: tuple[Arc, ...],
    inner_arcs: tuple[Arc, ...],
) -> None:
    """Validate CL1 envelope, transition vias, and OSC2-layer crossovers."""
    endpoint_clearance = (
        (dimensions.primary_length_mm - secondary_stroke_length(cfg)) / 2.0
    )
    if endpoint_clearance + GEOMETRY_TOLERANCE_MM < cfg["cl1_primary_end_min_clearance_mm"]:
        raise ValueError("CL1 endpoint violates minimum clearance to the primary end winding.")

    minimum_pad_distance = cfg["via_diameter_mm"] + cfg["trace_spacing_mm"]
    for first, second in (("G", "X"), ("Q", "ZH")):
        if distance(points[first], points[second]) + GEOMETRY_TOLERANCE_MM < minimum_pad_distance:
            raise ValueError(f"CL1 paired vias {first}/{second} violate plated via clearance.")

    minimum_trace_distance = osc1_via_trace_clearance(cfg)
    for via_label in ("G", "X", "Q", "ZH", "J", "ZE"):
        nearest_primary_trace = min(
            point_to_segment_distance(points[via_label], segment)
            for coil in primary_geometry.coils
            for segment in coil.body_segments
        )
        if nearest_primary_trace + GEOMETRY_TOLERANCE_MM < minimum_trace_distance:
            raise ValueError(f"CL1 via {via_label} violates clearance to the primary winding.")

    osc2 = next((coil for coil in primary_geometry.coils if coil.name == "OSC2"), None)
    if osc2 is not None:
        if path_to_path_distance(crossover_segments, osc2.body_segments) + GEOMETRY_TOLERANCE_MM < trace_pitch(cfg):
            raise ValueError("CL1 crossover copper violates clearance to OSC2.")

    if cl2_geometry is not None:
        cl2_terminal_points = (cl2_geometry.points["A"], cl2_geometry.points["ZP"])
        for terminal in ("A", "ZN"):
            for point in cl2_terminal_points + tuple(primary_geometry.pads.values()):
                if distance(points[terminal], point) + GEOMETRY_TOLERANCE_MM < minimum_pad_distance:
                    raise ValueError(f"CL1 terminal {terminal} collides with an existing via.")

    via_centered_arcs = (
        ("M-N", target_arcs[1], points["ZE"]),
        ("ZA-ZB", inner_arcs[0], points["J"]),
        ("ZK-ZL", inner_arcs[2], points["C"]),
    )
    for name, arc, via_center in via_centered_arcs:
        for point in arc:
            if (
                abs(distance(point, via_center) - minimum_trace_distance)
                > GEOMETRY_TOLERANCE_MM
            ):
                raise ValueError(
                    f"CL1 {name} arc does not maintain clearance from its transition via."
                )

    if cl2_geometry is not None:
        crossing_arcs = (("K-L", target_arcs[0]), ("ZC-ZD", inner_arcs[1]))
        for name, arc in crossing_arcs:
            center = (arc[0][0], (arc[0][1] + arc[2][1]) / 2.0)
            radius = distance(center, arc[0])
            for label in ("J", "ZG"):
                radial_clearance = radius - distance(center, cl2_geometry.points[label])
                if radial_clearance + GEOMETRY_TOLERANCE_MM < trace_pitch(cfg):
                    raise ValueError(
                        f"CL1 {name} arc violates clearance to CL2 {label}."
                    )

    half_pitch = trace_pitch(cfg) / 2.0
    phase_offset = math.pi / 2.0
    parallel_curves = (
        (("E", "F", 1.0, half_pitch), ("V", "W", 1.0, -half_pitch)),
        (("H", "I", 1.0, -half_pitch), ("Y", "Z", 1.0, half_pitch)),
        (("O", "P", -1.0, -half_pitch), ("ZF", "ZG", -1.0, half_pitch)),
        (("R", "S", -1.0, half_pitch), ("ZI", "ZJ", -1.0, -half_pitch)),
    )
    polygonal_transition_tolerance = 0.002
    for first, second in parallel_curves:
        first_curve = secondary_curve_segments(
            cfg,
            dimensions,
            points[first[0]],
            points[first[1]],
            first[2],
            first[3],
            phase_offset_radians=phase_offset,
            mirror_phase_sign=False,
        )
        second_curve = secondary_curve_segments(
            cfg,
            dimensions,
            points[second[0]],
            points[second[1]],
            second[2],
            second[3],
            phase_offset_radians=phase_offset,
            mirror_phase_sign=False,
        )
        if (
            path_to_path_distance(first_curve, second_curve)
            + polygonal_transition_tolerance
            < trace_pitch(cfg)
        ):
            raise ValueError("CL1 parallel sinusoidal traces violate configured spacing.")


def build_cl1_geometry(
    cfg: dict | None = None,
    primary_geometry: PrimaryGeometry | None = None,
    cl2_geometry: SecondaryCoil | None = None,
) -> CL1Coil | None:
    """Build the two-turn CL1 receiver coil, or return ``None`` when disabled."""
    cfg = build_config() if cfg is None else cfg
    if not cfg["generate_cl1"]:
        return None
    dimensions = calculate_dimensions(cfg)
    validate_config(cfg, dimensions)
    primary_geometry = primary_geometry or build_primary_geometry(cfg)
    if cl2_geometry is None and cfg["generate_cl2"]:
        cl2_geometry = build_cl2_geometry(cfg, primary_geometry)
    points = build_cl1_point_map(cfg, dimensions)
    (
        target_segments,
        inner_segments,
        crossover_segments,
        target_arcs,
        inner_arcs,
    ) = build_cl1_routes(cfg, dimensions, points)
    validate_cl1_clearance(
        cfg,
        dimensions,
        primary_geometry,
        cl2_geometry,
        points,
        target_segments,
        inner_segments,
        crossover_segments,
        target_arcs,
        inner_arcs,
    )
    return CL1Coil(
        name="CL1",
        target_layer=receiver_layers(cfg)[0],
        inner_layer=receiver_layers(cfg)[1],
        crossover_layer=receiver_crossover_layer(cfg),
        stroke_length_mm=secondary_stroke_length(cfg),
        points=points,
        target_segments=target_segments,
        inner_segments=inner_segments,
        crossover_segments=crossover_segments,
        target_arcs=target_arcs,
        inner_arcs=inner_arcs,
        via_labels=("A", "C", "D", "G", "J", "Q", "T", "U", "X", "ZE", "ZH", "ZN"),
    )


def fp_line(start: Point, end: Point, width: float, layer: str) -> str:
    return f'''  (fp_line (start {start[0]:.6f} {start[1]:.6f}) (end {end[0]:.6f} {end[1]:.6f})
    (stroke (width {width:.6f}) (type solid)) (layer "{layer}"))\n'''


def fp_arc(arc: Arc, width: float, layer: str) -> str:
    start, mid, end = arc
    return f'''  (fp_arc (start {start[0]:.6f} {start[1]:.6f}) (mid {mid[0]:.6f} {mid[1]:.6f}) (end {end[0]:.6f} {end[1]:.6f})
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
    cl1_geometry = build_cl1_geometry(cfg, geometry, cl2_geometry)
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

    if cl1_geometry is not None:
        for segment in cl1_geometry.target_segments:
            sections.append(
                fp_line(segment[0], segment[1], cfg["trace_width_mm"], cl1_geometry.target_layer)
            )
        for arc in cl1_geometry.target_arcs:
            sections.append(fp_arc(arc, cfg["trace_width_mm"], cl1_geometry.target_layer))
        for segment in cl1_geometry.inner_segments:
            sections.append(
                fp_line(segment[0], segment[1], cfg["trace_width_mm"], cl1_geometry.inner_layer)
            )
        for arc in cl1_geometry.inner_arcs:
            sections.append(fp_arc(arc, cfg["trace_width_mm"], cl1_geometry.inner_layer))
        for segment in cl1_geometry.crossover_segments:
            sections.append(
                fp_line(
                    segment[0],
                    segment[1],
                    cfg["trace_width_mm"],
                    cl1_geometry.crossover_layer,
                )
            )
        for label in cl1_geometry.via_labels:
            pad_name = {
                "A": cfg["cl1_output_pad_name"],
                "ZN": cfg["cl1_return_pad_name"],
            }.get(label, f"CL1_{label}")
            sections.append(
                pad_thru_hole(
                    pad_name,
                    cl1_geometry.points[label],
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
    cl1_geometry = build_cl1_geometry(cfg, geometry, cl2_geometry)
    output_path = write_linear_sensor_footprint(cfg)
    dims = geometry.dimensions
    print(f"Wrote {output_path}")
    print(
        "Primary outer centerline envelope: "
        f"{dims.primary_length_mm:.3f} mm x {dims.primary_width_mm:.3f} mm"
    )
    if cl2_geometry is not None:
        print(f"CL2 active waveform span: {cl2_geometry.stroke_length_mm:.3f} mm")
    if cl1_geometry is not None:
        print(f"CL1 active waveform span: {cl1_geometry.stroke_length_mm:.3f} mm")


if __name__ == "__main__":
    main()
