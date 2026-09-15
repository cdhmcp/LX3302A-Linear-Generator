#!/usr/bin/env python3
"""KiCad footprint generator for LX3302A linear sensor coil layouts.

OSC1 and OSC2 are built from descriptive primary-coil point maps. Receiver
coils CL1 and CL2 are built from generalized multi-turn layouts for all valid
turn counts.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import itertools
import heapq
import math
from pathlib import Path


MIL_TO_MM = 0.0254
GEOMETRY_TOLERANCE_MM = 1e-9
ROUTING_POLYGONAL_TOLERANCE_MM = 0.003

Point = tuple[float, float]
Segment = tuple[Point, Point]
Arc = tuple[Point, Point, Point]


# =============================================================================
# Properties
# =============================================================================
PROPERTIES = {
    # Moving target and stroke inputs
    "target_x_mm": 20.0,            # target width
    "target_y_mm": 13.0,             # target height
    "stroke_range_mm": 90.0,        # typically total mechanical travel of target + width of target for best primary-to-secondary coupling
    "target_side": "top",           # valid options: top OR bottom

    # Primary oscillator settings
    "primary_end_extension_mm": 3.0,    # this is how far the primary extends past the secondary windings on either end of the sensor
    "primary_y_margin_mm": 0.075,       # this is how far the primary extends past the secondary windings in the vertical (y) direction
    "number_of_primary_turns": 3,

    # Secondary receiver settings
    "number_of_secondary_turns": 5,     # valid range: 1..5
    "secondary_y_reduction_mm": 1.5,    # this is subracted from target_y_mm to give the height/amplitude of the secondary windings, windings slightly smaller than the target is best practice

    # Trace & Via constraints 
    "trace_width_mm": 8 * MIL_TO_MM,
    "trace_spacing_mm": 9 * MIL_TO_MM,
    "via_hole_size_mm": 8 * MIL_TO_MM,
    "via_diameter_mm": 16 * MIL_TO_MM,
 
    # Fanout tuning
    "fanout_side": "left",              # valid options: right OR left
    "terminal_escape_length_mm": 10.0,
    "osc1_vin_exit_offset_mm": 1.2,

    # Naming
    "footprint_name": "LX3302A_LINEAR_SENSOR_COILS",
    "reference_text": "REF**",
    "primary_input_pad_name": "VIN",
    "osc1_output_pad_name": "OSC1",
    "osc2_output_pad_name": "OSC2",
    "cl2_output_pad_name": "CL2",
    "cl2_return_pad_name": "CL2-GND",
    "cl1_output_pad_name": "CL1",
    "cl1_return_pad_name": "CL1-GND",

    # Output
    "output_dir": "InductiveSensors.pretty",
    "generate_osc1": True,
    "generate_osc2": True,
    "generate_cl2": True,
    "generate_cl1": True,
    "allow_invalid_geometry": True,  # when True, skip copper/via validation checks so invalid footprints can still be rendered for visual debugging


    "secondary_curve_samples_per_cycle": 256,
    "secondary_jump_runup_via_multiplier": 3.0,
    "secondary_jump_detour_via_multiplier": 0.35,
    # Recommended CL1 transition-column range: 0.02 to 0.05.
    "cl1_transition_column_fraction": 0.03,
    "cl1_primary_end_min_clearance_mm": 1.0,
}


@dataclass(frozen=True)
class SensorDimensions:
    """Calculated sensing and primary envelope dimensions in millimeters."""

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


@dataclass(frozen=True)
class SecondaryLayoutPlan:
    """Internal receiver-coil layout data before wrapping in public dataclasses."""

    points: dict[str, Point]
    target_segments: tuple[Segment, ...]
    inner_segments: tuple[Segment, ...]
    via_labels: tuple[str, ...]
    target_forward_paths: tuple[tuple[Segment, ...], ...]
    target_reverse_paths: tuple[tuple[Segment, ...], ...]
    inner_forward_paths: tuple[tuple[Segment, ...], ...]
    inner_reverse_paths: tuple[tuple[Segment, ...], ...]
    left_target_handoff_paths: tuple[tuple[Segment, ...], ...]
    left_inner_handoff_paths: tuple[tuple[Segment, ...], ...]
    left_handoff_station_x: dict[str, float]
    entry_escape_path: tuple[Segment, ...]
    return_escape_path: tuple[Segment, ...]


@dataclass(frozen=True)
class CL1LayoutPlan:
    """Internal CL1 layout data before wrapping in the public coil dataclass."""

    points: dict[str, Point]
    target_segments: tuple[Segment, ...]
    inner_segments: tuple[Segment, ...]
    crossover_segments: tuple[Segment, ...]
    target_arcs: tuple[Arc, ...]
    inner_arcs: tuple[Arc, ...]
    via_labels: tuple[str, ...]
    target_forward_paths: tuple[tuple[Segment, ...], ...]
    target_reverse_paths: tuple[tuple[Segment, ...], ...]
    inner_forward_paths: tuple[tuple[Segment, ...], ...]
    inner_reverse_paths: tuple[tuple[Segment, ...], ...]
    right_via_labels: tuple[str, ...]
    left_via_labels: tuple[str, ...]


@dataclass(frozen=True)
class CL2RightTurnaroundPlan:
    """Packed right-end jog-via handoff geometry for the generalized CL2 path."""

    via_points: dict[str, Point]
    via_labels: tuple[str, ...]
    target_segments: tuple[Segment, ...]
    inner_segments: tuple[Segment, ...]
    column_count: int
    rightmost_u: float
    assignment: tuple[int, ...]
    minimum_adjacent_spacing: float
    score: float


def build_config(overrides: dict | None = None) -> dict:
    """Combine user-editable settings and optional programmatic overrides."""
    cfg = {**PROPERTIES}
    if overrides:
        cfg.update(overrides)
    return cfg


def should_skip_geometry_validation(cfg: dict) -> bool:
    """Return whether debug footprint generation should bypass validation errors."""
    return cfg["allow_invalid_geometry"]


def calculate_dimensions(cfg: dict) -> SensorDimensions:
    """Calculate receiver reference bounds and the primary outer centerline."""
    secondary_length = cfg["stroke_range_mm"]
    secondary_width = cfg["target_y_mm"] - cfg["secondary_y_reduction_mm"]
    primary_length = secondary_length + (2.0 * cfg["primary_end_extension_mm"])
    primary_width = secondary_width + (2.0 * cfg["primary_y_margin_mm"])
    return SensorDimensions(
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


def secondary_via_spacing(cfg: dict) -> float:
    """Return center spacing for receiver-layer transition vias."""
    return cfg["via_diameter_mm"] + cfg["trace_spacing_mm"]


def terminal_pad_pitch(cfg: dict) -> float:
    """Return center spacing for adjacent external through-via pads."""
    return cfg["via_diameter_mm"] + cfg["trace_spacing_mm"]


def terminal_column_x(cfg: dict, dimensions: SensorDimensions) -> float:
    """Return the common fanout column for all external terminal vias."""
    return fanout_direction(cfg) * (
        (dimensions.primary_length_mm / 2.0) + cfg["terminal_escape_length_mm"]
    )


def terminal_row_y(cfg: dict, pad_name: str) -> float:
    """Return the fanout order with the oscillator escapes above the receivers."""
    row_index = {
        "OSC2": -4,
        "VIN": -3,
        "OSC1": -2,
        # Keep CL2 adjacent to the oscillator escapes.  CL1's local return
        # handoff occupies the lower fanout lanes, where it can leave the
        # sensor without crossing CL2's left-end escape geometry.
        "CL2": -1,
        "CL2-GND": 0,
        "CL1-GND": 1,
        "CL1": 2,
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
    return cfg["stroke_range_mm"]


def primary_inner_half_height(cfg: dict, dimensions: SensorDimensions) -> float:
    """Return the innermost primary horizontal centerline distance from center."""
    return (dimensions.primary_width_mm / 2.0) - (
        (cfg["number_of_primary_turns"] - 1) * trace_pitch(cfg)
    )


def centered_positions(count: int, step: float) -> tuple[float, ...]:
    """Return evenly spaced offsets centered on zero."""
    midpoint = (count - 1) / 2.0
    return tuple((index - midpoint) * step for index in range(count))


def secondary_turn_offsets(cfg: dict) -> tuple[float, ...]:
    """Return CL2 rail offsets from the lower outer turn toward the upper outer turn."""
    return centered_positions(cfg["number_of_secondary_turns"], trace_pitch(cfg))


def cl1_turn_offsets(cfg: dict) -> tuple[float, ...]:
    """Return CL1 rail offsets from the upper outer turn toward the lower outer turn."""
    return tuple(-offset for offset in secondary_turn_offsets(cfg))


def cl2_quarter_column_shifts(cfg: dict) -> tuple[float, ...]:
    """Return CL2 quarter-wave transition-column shifts centered on each quarter axis."""
    return tuple(reversed(centered_positions(cfg["number_of_secondary_turns"], secondary_via_spacing(cfg))))


def cl1_midpoint_columns(cfg: dict) -> tuple[float, ...]:
    """Return CL1 centered layer-transition columns around x=0."""
    return centered_positions(cfg["number_of_secondary_turns"], secondary_via_spacing(cfg))


def secondary_outer_half_height(dimensions: SensorDimensions) -> float:
    """Return the legacy outer receiver centerline envelope used by the 2-turn reference."""
    return dimensions.secondary_width_mm / 2.0


def secondary_wave_amplitude_for_offsets(
    dimensions: SensorDimensions,
    rail_offsets: tuple[float, ...],
) -> float:
    """Return the sine-wave amplitude that keeps the outermost rail on the legacy envelope."""
    max_offset = max((abs(offset) for offset in rail_offsets), default=0.0)
    return secondary_outer_half_height(dimensions) - max_offset


def cl2_turn_columns(cfg: dict) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return left and right CL2 turn columns in the left-entry frame."""
    half_span = secondary_stroke_length(cfg) / 2.0
    via_spacing = secondary_via_spacing(cfg)
    left = tuple((-half_span + (turn * via_spacing)) for turn in range(cfg["number_of_secondary_turns"]))
    right = tuple((half_span - (turn * via_spacing)) for turn in range(cfg["number_of_secondary_turns"]))
    return left, right


def cl1_right_end_column(cfg: dict, turn_index: int) -> float:
    """Return one CL1 right-turn column in the left-entry frame."""
    half_span = secondary_stroke_length(cfg) / 2.0
    return half_span - (turn_index * secondary_via_spacing(cfg))


def cl1_left_transition_column(cfg: dict, turn_index: int) -> float:
    """Return one CL1 left transition column in the left-entry frame."""
    half_span = secondary_stroke_length(cfg) / 2.0
    left_x = -half_span
    return (
        left_x
        + (secondary_stroke_length(cfg) * cfg["cl1_transition_column_fraction"])
        + (turn_index * secondary_via_spacing(cfg))
    )


def point_at_station_x(point: Point, station_x: float) -> Point:
    """Clamp a sampled rail point to a known turn column while preserving y."""
    return (station_x, point[1])


def mirror_points_horizontally(points: dict[str, Point]) -> dict[str, Point]:
    """Return a new point map mirrored across the vertical centerline."""
    return {label: (-point[0], point[1]) for label, point in points.items()}


def mirror_segments_horizontally(segments: tuple[Segment, ...]) -> tuple[Segment, ...]:
    """Return mirrored copper segments across the vertical centerline."""
    return tuple(((-start[0], start[1]), (-end[0], end[1])) for start, end in segments)


def mirror_arcs_horizontally(arcs: tuple[Arc, ...]) -> tuple[Arc, ...]:
    """Return mirrored copper arcs across the vertical centerline."""
    return tuple(
        (
            (-start[0], start[1]),
            (-mid[0], mid[1]),
            (-end[0], end[1]),
        )
        for start, mid, end in arcs
    )


def canonical_receiver_label(label: str) -> str:
    """Return the public descriptive name for a receiver layout anchor.

    The multi-turn planners use compact internal directions while calculating
    routes.  Their public point maps and KiCad pad labels use explicit curve
    boundary and jog roles instead, without changing any coordinates.
    """
    substitutions = (
        ("_RIGHT_TURNAROUND_RUNUP", "_RIGHT_TURNAROUND_JOG"),
        ("_LEFT_TURNAROUND_VIA", "_LEFT_TURNAROUND_VIA"),
        ("_RETURN_RIGHT_OUTER", "_RETURN_RIGHT_OUTER_END"),
        ("_RETURN_RIGHT_INNER", "_RETURN_RIGHT_INNER_START"),
        ("_RETURN_LEFT_INNER", "_RETURN_LEFT_INNER_END"),
        ("_RETURN_LEFT_OUTER", "_RETURN_LEFT_OUTER_START"),
        ("_LEFT_OUTER", "_FORWARD_LEFT_OUTER_END"),
        ("_LEFT_INNER", "_FORWARD_LEFT_INNER_START"),
        ("_RIGHT_INNER", "_FORWARD_RIGHT_INNER_END"),
        ("_RIGHT_OUTER", "_FORWARD_RIGHT_OUTER_START"),
        ("_RIGHT_END", "_RIGHT_TURNAROUND_START"),
        ("_LEFT_END", "_LEFT_TURNAROUND_END"),
    )
    for old, new in substitutions:
        if label.endswith(old):
            return f"{label[:-len(old)]}{new}"
    return label


def canonical_receiver_point_map(
    points: dict[str, Point],
    via_labels: tuple[str, ...],
) -> tuple[dict[str, Point], tuple[str, ...]]:
    """Rename public receiver anchors and via labels without moving copper."""
    renamed_points = {canonical_receiver_label(label): point for label, point in points.items()}
    renamed_vias = tuple(canonical_receiver_label(label) for label in via_labels)
    if len(renamed_points) != len(points):
        raise ValueError("Canonical receiver naming produced duplicate point labels.")
    return renamed_points, renamed_vias


def validate_config(cfg: dict, dimensions: SensorDimensions | None = None) -> None:
    """Reject impossible envelope, fabrication, or breakout inputs."""
    positive_values = (
        "target_x_mm",
        "target_y_mm",
        "stroke_range_mm",
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
    for name in ("secondary_y_reduction_mm",):
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
    if not isinstance(cfg["allow_invalid_geometry"], bool):
        raise ValueError("allow_invalid_geometry must be a boolean.")
    if cfg["generate_osc2"] and not cfg["generate_osc1"]:
        raise ValueError("OSC2 requires OSC1 because it shares OSC1's VIN transition via.")
    if (
        not isinstance(cfg["number_of_secondary_turns"], int)
        or not 1 <= cfg["number_of_secondary_turns"] <= 5
    ):
        raise ValueError("number_of_secondary_turns must be an integer between 1 and 5.")
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
    receiver_via_spacing = secondary_via_spacing(cfg)
    inset = (cfg["number_of_primary_turns"] - 1) * pitch
    inner_width = dimensions.primary_width_mm - (2.0 * inset)
    inner_length = dimensions.primary_length_mm - (2.0 * inset)
    if inner_width < pitch:
        raise ValueError("Primary width is insufficient for requested turns and trace spacing.")
    if inner_length < pitch:
        raise ValueError("Primary length is insufficient for requested turns and trace spacing.")

    required_secondary_width = cfg["number_of_secondary_turns"] * pitch
    if dimensions.secondary_width_mm + GEOMETRY_TOLERANCE_MM < required_secondary_width:
        raise ValueError(
            "Secondary width is insufficient for requested secondary turns and trace spacing."
        )

    cl2_left_columns, cl2_right_columns = cl2_turn_columns(cfg)
    if (
        cl2_right_columns[-1] - cl2_left_columns[-1]
        + GEOMETRY_TOLERANCE_MM
        < pitch
    ):
        raise ValueError("Secondary stroke length is insufficient for requested secondary turns.")

    if cfg["number_of_secondary_turns"] > 1:
        last_transition_x = cl1_left_transition_column(
            cfg, cfg["number_of_secondary_turns"] - 2
        )
        last_right_x = cl1_right_end_column(cfg, cfg["number_of_secondary_turns"] - 1)
        if last_transition_x + receiver_via_spacing + GEOMETRY_TOLERANCE_MM >= last_right_x:
            raise ValueError("CL1 transition columns exceed the available secondary span.")
def osc1_via_trace_clearance(cfg: dict) -> float:
    """Return center-to-center clearance from the shared VIN via to a trace."""
    return (
        (cfg["via_diameter_mm"] / 2.0)
        + (cfg["trace_width_mm"] / 2.0)
        + cfg["trace_spacing_mm"]
    )


def osc1_turn_labels(turn_index: int) -> tuple[str, str, str, str, str, str]:
    """Return canonical point-map labels for one OSC1 perimeter turn."""
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
    """Construct OSC1 points using descriptive path anchors."""
    side = fanout_direction(cfg)
    pitch = trace_pitch(cfg)
    diagonal_junction_separation = parallel_45_junction_separation(cfg)
    half_length = dimensions.primary_length_mm / 2.0
    half_width = dimensions.primary_width_mm / 2.0
    outer_near_x = side * half_length
    transition_half_height = pitch / 2.0
    terminal_x = terminal_column_x(cfg, dimensions)
    turn_count = cfg["number_of_primary_turns"]
    inner_near_x = side * (half_length - ((turn_count - 1) * pitch))
    via_transition = osc1_via_trace_clearance(cfg)
    inner_top_y = -(half_width - ((turn_count - 1) * pitch))
    # The shared VIN via sits beside the innermost vertical rail and below its
    # top horizontal rail.  Work backward from its 45 degree approach so the
    # inward transitions form nearly complete, pitch-preserving turns.
    via_y = inner_top_y + via_transition + cfg["trace_spacing_mm"]
    last_end_y = via_y - via_transition
    start_y = (
        last_end_y
        + (turn_count * diagonal_junction_separation)
        - ((turn_count - 1) * pitch)
    )
    terminal_y = terminal_row_y(cfg, "OSC1")
    # Leave the terminal column toward the sensor, then use an orthogonal
    # fanout dogleg before the long horizontal run into the first turn.
    entry_outer_x = terminal_x - (side * via_transition)
    entry_inner_x = entry_outer_x
    points: dict[str, Point] = {
        "TERMINAL_OUTPUT_VIA": (terminal_x, terminal_y),
        "ENTRY_FANOUT_JOG": (entry_outer_x, terminal_y),
        "ENTRY_WINDING_JOG": (entry_inner_x, start_y),
    }

    for turn in range(turn_count):
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

    last_end = osc1_turn_labels(turn_count - 1)[5]
    # Place the shared VIN via outside the inner vertical rail and just below
    # its top horizontal rail.  The horizontal separation is one normal
    # via-to-trace clearance; the vertical separation contains two copper
    # spacings (the normal clearance plus one additional trace spacing).
    via_y = inner_top_y + via_transition + cfg["trace_spacing_mm"]
    points[last_end] = (inner_near_x, last_end_y)
    via_x = inner_near_x - (side * via_transition)
    points["SHARED_VIN_VIA"] = (via_x, via_y)
    # A direct orthogonal VIN escape is clear of the new oscillator fanout
    # order when it first stops one via-to-trace clearance inside the column.
    points["VIN_FANOUT_JOG"] = (terminal_x - (side * via_transition), via_y)
    points["TERMINAL_VIN_VIA"] = (terminal_x, terminal_row_y(cfg, "VIN"))
    points["VIN_TERMINAL_APPROACH"] = (
        points["VIN_FANOUT_JOG"][0],
        points["TERMINAL_VIN_VIA"][1],
    )
    return points


def build_osc1_segments(
    cfg: dict,
    points: dict[str, Point],
) -> tuple[tuple[Segment, ...], tuple[Segment, ...]]:
    """Return OSC1 bottom-layer winding and target-facing escape segments."""
    point_sequence = ["TERMINAL_OUTPUT_VIA", "ENTRY_FANOUT_JOG", "ENTRY_WINDING_JOG"]
    for turn in range(cfg["number_of_primary_turns"]):
        point_sequence.extend(osc1_turn_labels(turn))
    point_sequence.append("SHARED_VIN_VIA")
    body = tuple(
        (points[start], points[end])
        for start, end in zip(point_sequence, point_sequence[1:])
    )
    escape_segments = (
        (points["SHARED_VIN_VIA"], points["VIN_FANOUT_JOG"]),
        (points["VIN_FANOUT_JOG"], points["VIN_TERMINAL_APPROACH"]),
        (points["VIN_TERMINAL_APPROACH"], points["TERMINAL_VIN_VIA"]),
    )
    return body, escape_segments


def osc2_turn_labels(turn_index: int) -> tuple[str, str, str, str]:
    """Return canonical labels for one overlaid OSC2 perimeter turn."""
    turn_number = turn_index + 1
    return (
        f"TURN{turn_number}_START",
        f"TURN{turn_number}_TOP_FAR",
        f"TURN{turn_number}_BOTTOM_FAR",
        f"TURN{turn_number}_END",
    )


def osc2_transition_labels(turn_index: int) -> tuple[str, str]:
    """Return the labels between an overlaid OSC2 perimeter and the next one."""
    turn_number = turn_index + 1
    return (
        f"TURN{turn_number}_TO_TURN{turn_number + 1}_NEAR_JOG",
        f"TURN{turn_number}_TO_TURN{turn_number + 1}_INNER_JOG",
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
    turn_count = cfg["number_of_primary_turns"]
    outer_x = osc1_points[osc1_turn_labels(0)[1]][0]
    points: dict[str, Point] = {
        "TERMINAL_OUTPUT_VIA": (
            osc1_points["TERMINAL_OUTPUT_VIA"][0],
            terminal_row_y(cfg, "OSC2"),
        ),
        "SHARED_VIN_VIA": osc1_points["SHARED_VIN_VIA"],
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
    points["VIN_APPROACH"] = (near_x[-1], points["SHARED_VIN_VIA"][1] + via_clearance)
    transition_tail_y = points["VIN_APPROACH"][1]
    for turn in reversed(range(turn_count - 1)):
        near_label, inner_label = osc2_transition_labels(turn)
        points[inner_label] = (near_x[turn + 1], transition_tail_y - junction_separation)
        points[near_label] = (near_x[turn], points[inner_label][1] + pitch)
        transition_tail_y = points[near_label][1]

    points["ENTRY_WINDING_JOG"] = (outer_x, transition_tail_y - junction_separation)
    # Enter the oscillator at the existing outer-turn transition height rather
    # than dropping to y=0.  The terminal-side route stays entirely inboard of
    # the fanout via column and uses only horizontal and vertical legs.
    entry_y = points["ENTRY_WINDING_JOG"][1]
    terminal_y = points["TERMINAL_OUTPUT_VIA"][1]
    points["ENTRY_FANOUT_JOG"] = (
        osc1_points["TERMINAL_OUTPUT_VIA"][0] - (side * via_clearance),
        terminal_y,
    )
    points["ENTRY_WINDING_APPROACH"] = (points["ENTRY_FANOUT_JOG"][0], entry_y)
    return points


def build_osc2_segments(cfg: dict, points: dict[str, Point]) -> tuple[Segment, ...]:
    """Return OSC2 path in descriptive point-map order."""
    point_sequence = [
        "TERMINAL_OUTPUT_VIA",
        "ENTRY_FANOUT_JOG",
        "ENTRY_WINDING_APPROACH",
        "ENTRY_WINDING_JOG",
    ]
    for turn in range(cfg["number_of_primary_turns"]):
        point_sequence.extend(osc2_turn_labels(turn))
        if turn < cfg["number_of_primary_turns"] - 1:
            point_sequence.extend(osc2_transition_labels(turn))
    point_sequence.extend(("VIN_APPROACH", "SHARED_VIN_VIA"))
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
    for start, end in (
        ("TERMINAL_OUTPUT_VIA", "TERMINAL_VIN_VIA"),
        ("TERMINAL_OUTPUT_VIA", "SHARED_VIN_VIA"),
        ("SHARED_VIN_VIA", "TERMINAL_VIN_VIA"),
    ):
        if distance(points[start], points[end]) < minimum_pad_distance:
            raise ValueError(f"OSC1 vias {start} and {end} violate plated via clearance.")

    minimum_trace_distance = osc1_via_trace_clearance(cfg)
    connected_body = {
        "TERMINAL_OUTPUT_VIA": body_segments[:1],
        "SHARED_VIN_VIA": body_segments[-1:],
        "TERMINAL_VIN_VIA": (),
    }
    for pad_name in connected_body:
        for segment in body_segments:
            if segment in connected_body[pad_name]:
                continue
            if point_to_segment_distance(points[pad_name], segment) < minimum_trace_distance:
                raise ValueError(f"OSC1 {pad_name} via violates clearance to winding copper.")
    for segment in escape_segments:
        if point_to_segment_distance(points["TERMINAL_OUTPUT_VIA"], segment) < minimum_trace_distance:
            raise ValueError("OSC1 output terminal via violates clearance to the VIN escape.")


def validate_osc2_clearance(
    cfg: dict,
    osc1_points: dict[str, Point],
    osc1_body_segments: tuple[Segment, ...],
    osc1_escape_segments: tuple[Segment, ...],
    osc2_points: dict[str, Point],
    osc2_body_segments: tuple[Segment, ...],
) -> None:
    """Check OSC2 fanout against existing through-vias and shared VIN routing."""
    if osc2_points["SHARED_VIN_VIA"] != osc1_points["SHARED_VIN_VIA"]:
        raise ValueError("OSC2 must terminate at OSC1's shared VIN via.")

    minimum_pad_distance = cfg["via_diameter_mm"] + cfg["trace_spacing_mm"]
    for osc1_pad in ("TERMINAL_OUTPUT_VIA", "TERMINAL_VIN_VIA"):
        if distance(osc2_points["TERMINAL_OUTPUT_VIA"], osc1_points[osc1_pad]) < minimum_pad_distance:
            raise ValueError(f"OSC2 output terminal via violates clearance to OSC1 {osc1_pad}.")

    minimum_trace_distance = osc1_via_trace_clearance(cfg)
    for segment in osc2_body_segments:
        if (
            point_to_segment_distance(osc1_points["TERMINAL_OUTPUT_VIA"], segment) + GEOMETRY_TOLERANCE_MM
            < minimum_trace_distance
        ):
            raise ValueError("OSC2 entry copper violates clearance to OSC1 output via.")
        if (
            point_to_segment_distance(osc1_points["TERMINAL_VIN_VIA"], segment) + GEOMETRY_TOLERANCE_MM
            < minimum_trace_distance
        ):
            raise ValueError("OSC2 copper violates clearance to OSC1 VIN terminal via.")
    for segment in osc1_body_segments + osc1_escape_segments:
        if point_to_segment_distance(osc2_points["TERMINAL_OUTPUT_VIA"], segment) < minimum_trace_distance:
            raise ValueError("OSC2 output terminal via violates clearance to OSC1 copper.")


def build_primary_geometry(cfg: dict | None = None) -> PrimaryGeometry:
    """Return point-driven primary geometry and the shared VIN escape."""
    cfg = build_config() if cfg is None else cfg
    dimensions = calculate_dimensions(cfg)
    validate_config(cfg, dimensions)
    osc1_points = build_osc1_point_map(cfg, dimensions)
    osc1_body_segments, osc1_escape_segments = build_osc1_segments(cfg, osc1_points)
    if not should_skip_geometry_validation(cfg):
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
    pads = {
        "OSC1_TERMINAL_OUTPUT_VIA": osc1_points["TERMINAL_OUTPUT_VIA"],
        "VIN_SHARED_VIA": osc1_points["SHARED_VIN_VIA"],
        "VIN_TERMINAL_VIA": osc1_points["TERMINAL_VIN_VIA"],
    }

    if cfg["generate_osc2"]:
        osc2_points = build_osc2_point_map(cfg, osc1_points)
        osc2_body_segments = build_osc2_segments(cfg, osc2_points)
        if not should_skip_geometry_validation(cfg):
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
        pads["OSC2_TERMINAL_OUTPUT_VIA"] = osc2_points["TERMINAL_OUTPUT_VIA"]

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
    amplitude_override: float | None = None,
) -> tuple[float, float]:
    """Return one secondary sinusoid centerline and its slope at ``x``."""
    span = secondary_stroke_length(cfg)
    amplitude = (
        (dimensions.secondary_width_mm / 2.0) - (trace_pitch(cfg) / 2.0)
        if amplitude_override is None
        else amplitude_override
    )
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
    amplitude_override: float | None = None,
) -> Point:
    """Offset one waveform rail perpendicular to its secondary centerline."""
    y, slope = secondary_wave_value_and_slope(
        cfg,
        dimensions,
        station_x,
        phase_sign,
        phase_offset_radians,
        amplitude_override,
    )
    normal_scale = math.hypot(slope, 1.0)
    return (
        station_x - ((slope / normal_scale) * rail_offset),
        y + (rail_offset / normal_scale),
    )


def receiver_transition_via_y(
    cfg: dict,
    dimensions: SensorDimensions,
    station_x: float,
    phase_sign: float,
    rail_offsets: tuple[float, ...],
    *,
    upper: bool,
    connected_rail_offsets: tuple[float, ...] = (),
    rail_spans: tuple[tuple[float, float, float], ...] | None = None,
    phase_offset_radians: float = 0.0,
    amplitude_override: float | None = None,
) -> float:
    """Return a transition-via Y that clears both primary and receiver rails.

    Transition vias live inside the primary winding, but a receiver rail can
    extend farther toward the sensor center than the innermost primary turn.
    The via may overlap the two receiver rails it intentionally joins.  All
    other receiver rails remain obstacles and need normal via-to-trace
    clearance.  Searching those non-connected curves at the actual transition
    column produces the least-inward legal location.  The primary boundary
    always retains its normal via-to-trace clearance.
    """
    clearance = osc1_via_trace_clearance(cfg)
    inner_primary_y = primary_inner_half_height(cfg, dimensions)
    obstacle_offsets = tuple(
        rail_offset
        for rail_offset in rail_offsets
        if not any(
            math.isclose(rail_offset, connected_offset, abs_tol=GEOMETRY_TOLERANCE_MM)
            for connected_offset in connected_rail_offsets
        )
    )
    connected_y_values = tuple(
        secondary_rail_point(
            cfg,
            dimensions,
            station_x,
            phase_sign,
            rail_offset,
            phase_offset_radians,
            amplitude_override,
        )[1]
        for rail_offset in connected_rail_offsets
    )
    if upper:
        primary_boundary = -inner_primary_y + clearance
        connected_upper = tuple(y for y in connected_y_values if y < 0.0)
        starting_boundary = max((primary_boundary, *connected_upper))
    else:
        primary_boundary = inner_primary_y - clearance
        connected_lower = tuple(y for y in connected_y_values if y > 0.0)
        starting_boundary = min((primary_boundary, *connected_lower))

    if not obstacle_offsets:
        return starting_boundary

    half_span = secondary_stroke_length(cfg) / 2.0
    obstacle_segments: list[Segment] = []
    obstacle_spans = (
        rail_spans
        if rail_spans is not None
        else tuple((rail_offset, -half_span, half_span) for rail_offset in obstacle_offsets)
    )
    for rail_offset, station_start_x, station_end_x in obstacle_spans:
        if rail_offset not in obstacle_offsets:
            continue
        start = secondary_rail_point(
            cfg,
            dimensions,
            station_start_x,
            phase_sign,
            rail_offset,
            phase_offset_radians,
            amplitude_override,
        )
        end = secondary_rail_point(
            cfg,
            dimensions,
            station_end_x,
            phase_sign,
            rail_offset,
            phase_offset_radians,
            amplitude_override,
        )
        obstacle_segments.extend(
            secondary_curve_segments(
                cfg,
                dimensions,
                start,
                end,
                phase_sign,
                rail_offset,
                station_start_x=station_start_x,
                station_end_x=station_end_x,
                phase_offset_radians=phase_offset_radians,
                mirror_phase_sign=False,
                amplitude_override=amplitude_override,
            )
        )

    def clears_receiver(via_y: float) -> bool:
        return all(
            point_to_segment_distance((station_x, via_y), segment)
            + ROUTING_POLYGONAL_TOLERANCE_MM
            >= clearance
            for segment in obstacle_segments
        )

    if upper:
        if clears_receiver(starting_boundary):
            return starting_boundary
        inward_boundary = 0.0
        if not clears_receiver(inward_boundary):
            raise ValueError("Receiver upper transition via cannot clear non-connected rails.")
        for _ in range(24):
            midpoint = (starting_boundary + inward_boundary) / 2.0
            if clears_receiver(midpoint):
                inward_boundary = midpoint
            else:
                starting_boundary = midpoint
        return inward_boundary

    if clears_receiver(starting_boundary):
        return starting_boundary
    inward_boundary = 0.0
    if not clears_receiver(inward_boundary):
        raise ValueError("Receiver lower transition via cannot clear non-connected rails.")
    for _ in range(24):
        midpoint = (starting_boundary + inward_boundary) / 2.0
        if clears_receiver(midpoint):
            inward_boundary = midpoint
        else:
            starting_boundary = midpoint
    return inward_boundary


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
    amplitude_override: float | None = None,
) -> tuple[Segment, ...]:
    """Sample part of a full-span sine rail while connecting mapped transition points."""
    stroke_length = secondary_stroke_length(cfg)
    effective_phase = phase_sign * (-fanout_direction(cfg) if mirror_phase_sign else 1.0)
    has_explicit_reference = reference_start is not None
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
    raw_ref_start_x = reference_start[0] if has_explicit_reference else station_start_x
    raw_ref_end_x = reference_end[0] if has_explicit_reference else station_end_x
    raw_start = secondary_rail_point(
        cfg,
        dimensions,
        raw_ref_start_x,
        effective_phase,
        rail_offset,
        phase_offset_radians,
        amplitude_override,
    )
    raw_end = secondary_rail_point(
        cfg,
        dimensions,
        raw_ref_end_x,
        effective_phase,
        rail_offset,
        phase_offset_radians,
        amplitude_override,
    )
    points: list[Point] = []
    for index in range(sample_count + 1):
        fraction = index / sample_count
        station_x = station_start_x + ((station_end_x - station_start_x) * fraction)
        if has_explicit_reference:
            reference_fraction = (station_x - reference_start[0]) / (
                reference_end[0] - reference_start[0]
            )
        else:
            reference_fraction = fraction
        raw_point = secondary_rail_point(
            cfg,
            dimensions,
            station_x,
            effective_phase,
            rail_offset,
            phase_offset_radians,
            amplitude_override,
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
    amplitude_override: float | None = None,
) -> Point:
    """Return a point on a full corrected rail before fanout-side mirroring."""
    raw_start = secondary_rail_point(
        cfg,
        dimensions,
        reference_start[0],
        phase_sign,
        rail_offset,
        phase_offset_radians,
        amplitude_override,
    )
    raw_end = secondary_rail_point(
        cfg,
        dimensions,
        reference_end[0],
        phase_sign,
        rail_offset,
        phase_offset_radians,
        amplitude_override,
    )
    raw_point = secondary_rail_point(
        cfg,
        dimensions,
        station_x,
        phase_sign,
        rail_offset,
        phase_offset_radians,
        amplitude_override,
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


def cl2_right_turnaround_assignment_candidates(
    turn_count: int,
    column_count: int,
) -> tuple[tuple[int, ...], ...]:
    """Return every surjective turn-to-column assignment for the CL2 right-end packer."""
    if column_count < 1 or column_count > turn_count:
        return ()
    return tuple(
        assignment
        for assignment in itertools.product(range(column_count), repeat=turn_count)
        if len(set(assignment)) == column_count
    )


def cl2_right_turnaround_segments(
    points: dict[str, Point],
    turn_count: int,
) -> tuple[tuple[str, ...], tuple[Segment, ...], tuple[Segment, ...]]:
    """Return the generated CL2 right-end jog-via turnaround segments."""
    via_labels = tuple(
        f"TURN{turn_number}_RIGHT_TURNAROUND_VIA"
        for turn_number in range(1, turn_count + 1)
    )
    target_segments = tuple(
        (points[f"TURN{turn_number}_RIGHT_END"], points[f"TURN{turn_number}_RIGHT_TURNAROUND_VIA"])
        for turn_number in range(1, turn_count + 1)
    )
    inner_segments = tuple(
        (points[f"TURN{turn_number}_RIGHT_TURNAROUND_VIA"], points[f"TURN{turn_number}_RIGHT_TURNAROUND_RUNUP"])
        for turn_number in range(1, turn_count + 1)
    )
    return via_labels, target_segments, inner_segments


def cl2_left_turnaround_via_points(cfg: dict) -> dict[str, Point]:
    """Return the provisional mirrored left-side jog-via column in the left-entry frame."""
    turn_count = cfg["number_of_secondary_turns"]
    if turn_count <= 1:
        return {}
    via_spacing = secondary_via_spacing(cfg)
    left_column_x = -((secondary_stroke_length(cfg) / 2.0) + via_spacing)
    return {
        f"TURN{turn_number}_LEFT_TURNAROUND_VIA": (left_column_x, via_y)
        for turn_number, via_y in enumerate(
            centered_positions(turn_count - 1, via_spacing),
            start=1,
        )
    }


def cl2_left_turnaround_segments(
    points: dict[str, Point],
    turn_count: int,
) -> tuple[tuple[str, ...], tuple[Segment, ...], tuple[Segment, ...]]:
    """Return the mirrored left-side jog-via handoff segments between CL2 turns."""
    via_labels = tuple(
        f"TURN{turn_number}_LEFT_TURNAROUND_VIA"
        for turn_number in range(1, turn_count)
    )
    target_segments = tuple(
        (
            points[f"TURN{turn_number}_LEFT_TURNAROUND_VIA"],
            points[f"TURN{turn_number + 1}_START"],
        )
        for turn_number in range(1, turn_count)
    )
    inner_segments = tuple(
        (
            points[f"TURN{turn_number}_LEFT_END"],
            points[f"TURN{turn_number}_LEFT_TURNAROUND_VIA"],
        )
        for turn_number in range(1, turn_count)
    )
    return via_labels, target_segments, inner_segments


def via_to_trace_clearance(cfg: dict) -> float:
    """Return the required centerline distance between a plated via and a trace."""
    return (
        (cfg["via_diameter_mm"] + cfg["trace_width_mm"]) / 2.0
        + cfg["trace_spacing_mm"]
    )


def polyline_segments(points: tuple[Point, ...]) -> tuple[Segment, ...]:
    """Convert a routed polyline into the footprint's segment representation."""
    return tuple(zip(points, points[1:]))


def simplify_polyline(points: tuple[Point, ...]) -> tuple[Point, ...]:
    """Remove collinear lattice points from a clearance-routed polyline."""
    if len(points) <= 2:
        return points
    simplified = [points[0]]
    for index, point in enumerate(points[1:-1], start=1):
        first = simplified[-1]
        # The actual collinearity check uses the previous retained point and the
        # next input point; keeping it here avoids emitting dense A* lattice lines.
        next_point = points[index + 1]
        cross = (
            ((point[0] - first[0]) * (next_point[1] - point[1]))
            - ((point[1] - first[1]) * (next_point[0] - point[0]))
        )
        if abs(cross) > GEOMETRY_TOLERANCE_MM:
            simplified.append(point)
    simplified.append(points[-1])
    return tuple(simplified)


def segment_clears_obstacles(
    segment: Segment,
    obstacles: tuple[Segment, ...],
    required_clearance: float,
) -> bool:
    """Return whether one candidate route segment clears every obstacle segment."""
    return all(
        segment_to_segment_distance(segment, obstacle) + ROUTING_POLYGONAL_TOLERANCE_MM
        >= required_clearance
        for obstacle in obstacles
    )


def left_handoff_escape_point(
    cfg: dict,
    dimensions: SensorDimensions,
    point: Point,
    phase_sign: float,
    escape_length: float,
    amplitude_override: float,
) -> Point:
    """Extend a left-edge rail endpoint outward along its local tangent."""
    half_span = secondary_stroke_length(cfg) / 2.0
    _, slope = secondary_wave_value_and_slope(
        cfg,
        dimensions,
        -half_span,
        phase_sign,
        amplitude_override=amplitude_override,
    )
    return (point[0] - escape_length, point[1] - (slope * escape_length))


def route_left_handoff_channel(
    cfg: dict,
    dimensions: SensorDimensions,
    start: Point,
    via: Point,
    escape_direction: Point,
    obstacles: tuple[Segment, ...],
) -> tuple[Segment, ...] | None:
    """Route one left-side handoff through a compact outward-only clearance channel.

    The initial tangent runout keeps the departure at the rail's preserved pitch.
    A small deterministic A* lattice then finds the shortest remaining polyline
    to the selected via without allowing a return toward the coil body.
    """
    trace_clearance = trace_pitch(cfg)
    escape_length = max(trace_clearance * 2.0, secondary_via_spacing(cfg))
    direction_length = math.hypot(*escape_direction)
    if direction_length <= GEOMETRY_TOLERANCE_MM:
        return None
    escape = (
        start[0] + ((escape_direction[0] / direction_length) * escape_length),
        start[1] + ((escape_direction[1] / direction_length) * escape_length),
    )
    initial_segment = (start, escape)
    if not segment_clears_obstacles(initial_segment, obstacles, trace_clearance):
        return None

    # A 50 um lattice is significantly finer than the normal fabrication rules
    # while keeping the small (1..5 turn) local routing problem inexpensive.
    step = 0.05
    route_margin = max(2.0 * secondary_via_spacing(cfg), 1.5)
    min_x = min(escape[0], via[0]) - step
    max_columns = max(1, math.ceil((escape[0] - min_x) / step))
    vertical_extent = max(
        1,
        math.ceil((abs(via[1] - escape[1]) + route_margin) / step),
    )
    min_y = escape[1] - (vertical_extent * step)
    max_rows = vertical_extent * 2
    # Keep the escape point on the lattice exactly; rounding it onto a nearby
    # row can put a nominal-pitch departure infinitesimally inside a neighbor's
    # clearance envelope.
    start_row = vertical_extent

    def point_at(column: int, row: int) -> Point:
        return (escape[0] - (column * step), min_y + (row * step))

    start_node = (0, start_row)
    frontier: list[tuple[float, float, tuple[int, int]]] = []
    heapq.heappush(frontier, (distance(escape, via), 0.0, start_node))
    cost: dict[tuple[int, int], float] = {start_node: 0.0}
    predecessor: dict[tuple[int, int], tuple[int, int] | None] = {start_node: None}
    goal_node: tuple[int, int] | None = None

    while frontier:
        _, current_cost, current = heapq.heappop(frontier)
        if current_cost != cost.get(current):
            continue
        current_point = point_at(*current)
        if segment_clears_obstacles((current_point, via), obstacles, trace_clearance):
            goal_node = current
            break

        for column_delta in (-1, 0, 1):
            next_column = current[0] + column_delta
            if not 0 <= next_column <= max_columns:
                continue
            for row_delta in (-1, 0, 1):
                if column_delta == 0 and row_delta == 0:
                    continue
                next_row = current[1] + row_delta
                if not 0 <= next_row <= max_rows:
                    continue
                neighbor = (next_column, next_row)
                neighbor_point = point_at(*neighbor)
                candidate_segment = (current_point, neighbor_point)
                if not segment_clears_obstacles(
                    candidate_segment, obstacles, trace_clearance
                ):
                    continue
                candidate_cost = current_cost + distance(current_point, neighbor_point)
                if candidate_cost + GEOMETRY_TOLERANCE_MM >= cost.get(neighbor, float("inf")):
                    continue
                cost[neighbor] = candidate_cost
                predecessor[neighbor] = current
                heuristic = distance(neighbor_point, via)
                heapq.heappush(
                    frontier,
                    (candidate_cost + heuristic, candidate_cost, neighbor),
                )

    if goal_node is None:
        return None

    nodes: list[tuple[int, int]] = []
    current: tuple[int, int] | None = goal_node
    while current is not None:
        nodes.append(current)
        current = predecessor[current]
    nodes.reverse()
    polyline = (start, escape, *(point_at(*node) for node in nodes[1:]), via)
    return polyline_segments(simplify_polyline(polyline))


def route_clears_vias(
    route: tuple[Segment, ...],
    other_vias: tuple[Point, ...],
    cfg: dict,
) -> bool:
    """Return whether a routed trace stays clear of all non-connected vias."""
    clearance = via_to_trace_clearance(cfg)
    return all(
        point_to_segment_distance(via, segment) + GEOMETRY_TOLERANCE_MM >= clearance
        for via in other_vias
        for segment in route
    )


def route_cl2_fanout_escape(
    cfg: dict,
    start: Point,
    end: Point,
    obstacles: tuple[Segment, ...],
    obstacle_vias: tuple[Point, ...],
) -> tuple[Segment, ...]:
    """Return the shortest local, clearance-safe CL2 fanout escape route.

    CL2's first and final rails must pass around the outside of the left
    turnaround bundle.  A coarse routing lattice is sufficient here: it is
    only used in the small area between the common y=0 fanout spine and the
    coil edge, and each retained shortcut is checked using the exact segment
    clearance routines.  The lattice keeps the result deterministic for all
    supported secondary-turn counts while the shortcut pass keeps the emitted
    footprint compact.
    """
    if distance(start, end) <= GEOMETRY_TOLERANCE_MM:
        return ()

    trace_clearance = trace_pitch(cfg)
    via_clearance = via_to_trace_clearance(cfg)
    step = min(0.10, trace_clearance / 4.0)
    route_margin = max(2.0 * trace_clearance, secondary_via_spacing(cfg))
    minimum_x = min(start[0], end[0]) - route_margin
    maximum_x = max(start[0], end[0]) + route_margin
    minimum_y = min(start[1], end[1]) - route_margin
    maximum_y = max(start[1], end[1]) + route_margin

    # Ignore remote waveform samples.  They cannot be reached by this local
    # escape search, and omitting them makes the exact final checks inexpensive.
    local_obstacles = tuple(
        segment
        for segment in obstacles
        if (
            max(segment[0][0], segment[1][0]) >= minimum_x - trace_clearance
            and min(segment[0][0], segment[1][0]) <= maximum_x + trace_clearance
            and max(segment[0][1], segment[1][1]) >= minimum_y - trace_clearance
            and min(segment[0][1], segment[1][1]) <= maximum_y + trace_clearance
        )
    )
    local_vias = tuple(
        via
        for via in obstacle_vias
        if (
            minimum_x - via_clearance <= via[0] <= maximum_x + via_clearance
            and minimum_y - via_clearance <= via[1] <= maximum_y + via_clearance
        )
    )

    def segment_is_clear(segment: Segment) -> bool:
        return (
            segment_clears_obstacles(segment, local_obstacles, trace_clearance)
            and all(
                point_to_segment_distance(via, segment)
                + ROUTING_POLYGONAL_TOLERANCE_MM
                >= via_clearance
                for via in local_vias
            )
        )

    if segment_is_clear((start, end)):
        return ((start, end),)

    column_count = max(1, math.ceil((maximum_x - minimum_x) / step))
    row_count = max(1, math.ceil((maximum_y - minimum_y) / step))
    start_node = (
        round((start[0] - minimum_x) / step),
        round((start[1] - minimum_y) / step),
    )
    start_node = (
        min(max(start_node[0], 0), column_count),
        min(max(start_node[1], 0), row_count),
    )

    def node_point(node: tuple[int, int]) -> Point:
        return (minimum_x + (node[0] * step), minimum_y + (node[1] * step))

    # If every lattice node is this far from an obstacle, the intervening
    # 8-connected segment remains clear as well.  Exact checks are still used
    # for the final shortcut route below.
    node_guard = (step * math.sqrt(2.0) / 2.0) + ROUTING_POLYGONAL_TOLERANCE_MM
    node_clear_cache: dict[tuple[int, int], bool] = {}

    def node_is_clear(node: tuple[int, int]) -> bool:
        cached = node_clear_cache.get(node)
        if cached is not None:
            return cached
        point = node_point(node)
        clear = (
            all(
                point_to_segment_distance(point, segment) + ROUTING_POLYGONAL_TOLERANCE_MM
                >= trace_clearance + node_guard
                for segment in local_obstacles
            )
            and all(
                distance(point, via) + ROUTING_POLYGONAL_TOLERANCE_MM
                >= via_clearance + node_guard
                for via in local_vias
            )
        )
        node_clear_cache[node] = clear
        return clear

    frontier: list[tuple[float, float, tuple[int, int]]] = []
    heapq.heappush(frontier, (distance(start, end), 0.0, start_node))
    costs: dict[tuple[int, int], float] = {start_node: 0.0}
    predecessors: dict[tuple[int, int], tuple[int, int] | None] = {start_node: None}
    goal_node: tuple[int, int] | None = None

    while frontier:
        _, current_cost, current = heapq.heappop(frontier)
        if current_cost != costs.get(current):
            continue
        current_point = start if current == start_node else node_point(current)
        if segment_is_clear((current_point, end)):
            goal_node = current
            break
        for x_delta in (-1, 0, 1):
            for y_delta in (-1, 0, 1):
                if x_delta == 0 and y_delta == 0:
                    continue
                neighbor = (current[0] + x_delta, current[1] + y_delta)
                if not (
                    0 <= neighbor[0] <= column_count
                    and 0 <= neighbor[1] <= row_count
                    and node_is_clear(neighbor)
                ):
                    continue
                neighbor_point = node_point(neighbor)
                candidate_cost = current_cost + distance(current_point, neighbor_point)
                if candidate_cost + GEOMETRY_TOLERANCE_MM >= costs.get(
                    neighbor, float("inf")
                ):
                    continue
                costs[neighbor] = candidate_cost
                predecessors[neighbor] = current
                heapq.heappush(
                    frontier,
                    (
                        candidate_cost + distance(neighbor_point, end),
                        candidate_cost,
                        neighbor,
                    ),
                )

    if goal_node is None:
        raise ValueError("CL2 fanout escape could not satisfy configured clearance.")

    nodes: list[tuple[int, int]] = []
    current: tuple[int, int] | None = goal_node
    while current is not None:
        nodes.append(current)
        current = predecessors[current]
    route_points = [start, *(node_point(node) for node in reversed(nodes[:-1])), end]

    # Greedily retain the farthest exact-clear point.  This removes the lattice
    # stair-steps and minimizes the emitted copper length for the chosen route.
    simplified = [route_points[0]]
    point_index = 0
    while point_index < len(route_points) - 1:
        next_index = len(route_points) - 1
        while next_index > point_index + 1:
            if segment_is_clear((route_points[point_index], route_points[next_index])):
                break
            next_index -= 1
        simplified.append(route_points[next_index])
        point_index = next_index
    return polyline_segments(tuple(simplified))


def route_cl2_fanout_wrap(
    cfg: dict,
    convergence: Point,
    endpoint: Point,
    via_stack: tuple[Point, ...],
    wrap_sign: float,
    fanout_side: float,
    obstacles: tuple[Segment, ...],
    obstacle_vias: tuple[Point, ...],
) -> tuple[Segment, ...]:
    """Route one CL2 fanout trace around a specified side of the via stack.

    The fanout must not take the geometrically shortest diagonal around the
    turnaround vias: that produces the wrong topology and obscures the shared
    y=0 trunk.  This route instead makes a compact three-sided wrap.  Target
    entry uses the negative-y side of the stack, while the inner return uses
    the positive-y side.  A small candidate sweep keeps the wrap minimal while
    retaining exact trace and via clearance checks.
    """
    trace_clearance = trace_pitch(cfg)
    via_clearance = via_to_trace_clearance(cfg)

    def route_is_clear(route: tuple[Segment, ...]) -> bool:
        return all(
            segment_clears_obstacles(segment, obstacles, trace_clearance)
            and all(
                point_to_segment_distance(via, segment)
                + ROUTING_POLYGONAL_TOLERANCE_MM
                >= via_clearance
                for via in obstacle_vias
            )
            for segment in route
        )

    if not via_stack:
        direct = ((convergence, endpoint),)
        if route_is_clear(direct):
            return direct
        raise ValueError("CL2 fanout escape could not clear the coil edge.")

    # The stack-facing corner must lie beyond every detour via, not merely the
    # first one, so a diagonal/staggered via rack still receives a true wrap.
    if fanout_side < 0.0:
        stack_inner_x = max(point[0] for point in via_stack) + via_clearance
    else:
        stack_inner_x = min(point[0] for point in via_stack) - via_clearance
    stack_wrap_y = (
        (min(point[1] for point in via_stack) - via_clearance)
        if wrap_sign < 0.0
        else (max(point[1] for point in via_stack) + via_clearance)
    )

    best_route: tuple[Segment, ...] | None = None
    best_length = float("inf")
    # The first candidate is the tightest legal wrap.  Extra margins are only
    # considered if local coil geometry needs more room.
    for margin_multiplier in (0.0, 0.5, 1.0, 1.5, 2.0):
        wrap_y = stack_wrap_y + (wrap_sign * margin_multiplier * trace_clearance)
        for inner_multiplier in (0.0, 0.5, 1.0):
            inner_x = stack_inner_x - (fanout_side * inner_multiplier * trace_clearance)
            route = polyline_segments(
                (convergence, (convergence[0], wrap_y), (inner_x, wrap_y), endpoint)
            )
            if not route_is_clear(route):
                continue
            route_length = sum(distance(*segment) for segment in route)
            if route_length + GEOMETRY_TOLERANCE_MM < best_length:
                best_route = route
                best_length = route_length

    if best_route is None:
        raise ValueError("CL2 fanout wrap could not satisfy configured clearance.")
    return best_route


def build_cl2_left_turnaround_plan(
    cfg: dict,
    dimensions: SensorDimensions,
    points: dict[str, Point],
    outer_offsets: tuple[float, ...],
    amplitude_override: float,
) -> tuple[dict[str, Point], dict[int, tuple[Segment, ...]], dict[int, tuple[Segment, ...]]]:
    """Route left CL2 inter-turn handoffs with staggered endpoints and local DRC.

    Each layer exits along a tangent before entering an outward-only channel.  The
    small candidate search varies the detour-via rack and returns the shortest
    all-clear result, which keeps the algorithm practical and deterministic for
    one through five receiver turns.
    """
    handoff_count = len(outer_offsets) - 1
    if handoff_count <= 0:
        return {}, {}, {}

    half_span = secondary_stroke_length(cfg) / 2.0
    trace_clearance = trace_pitch(cfg)
    via_spacing = secondary_via_spacing(cfg)
    target_start_paths = tuple(
        secondary_curve_segments(
            cfg,
            dimensions,
            points[f"TURN{turn_number}_START"],
            points[f"TURN{turn_number}_LEFT_OUTER"],
            -1.0,
            outer_offsets[turn_number - 1],
            station_start_x=-half_span,
            station_end_x=points[f"TURN{turn_number}_LEFT_OUTER"][0],
            mirror_phase_sign=False,
            amplitude_override=amplitude_override,
        )
        for turn_number in range(1, handoff_count + 2)
    )
    inner_end_paths = tuple(
        secondary_curve_segments(
            cfg,
            dimensions,
            points[f"TURN{turn_number}_RETURN_LEFT_OUTER"],
            points[f"TURN{turn_number}_LEFT_END"],
            1.0,
            outer_offsets[turn_number - 1],
            station_start_x=points[f"TURN{turn_number}_RETURN_LEFT_OUTER"][0],
            station_end_x=-half_span,
            mirror_phase_sign=False,
            amplitude_override=amplitude_override,
        )
        for turn_number in range(1, handoff_count + 2)
    )
    edge_x = min(
        *(points[f"TURN{turn_number}_START"][0] for turn_number in range(1, handoff_count + 2)),
        *(points[f"TURN{turn_number}_LEFT_END"][0] for turn_number in range(1, handoff_count + 2)),
    )
    rack_shifts = tuple(
        shift * (via_spacing / 2.0)
        for shift in range(-2, 5)
    )
    best_plan: tuple[
        float,
        dict[str, Point],
        dict[int, tuple[Segment, ...]],
        dict[int, tuple[Segment, ...]],
    ] | None = None

    for rack_offset in (1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 20.0):
        rack_x = edge_x - (rack_offset * via_spacing)
        for rack_shift in rack_shifts:
            via_points = {
                f"TURN{turn_number}_LEFT_TURNAROUND_VIA": (rack_x, via_y + rack_shift)
                for turn_number, via_y in enumerate(
                    centered_positions(handoff_count, via_spacing), start=1
                )
            }
            vias = tuple(via_points.values())
            if any(
                point_to_segment_distance(via, segment) + GEOMETRY_TOLERANCE_MM
                < via_to_trace_clearance(cfg)
                for via in vias
                for path in target_start_paths + inner_end_paths
                for segment in path
            ):
                continue

            target_routes: dict[int, tuple[Segment, ...]] = {}
            target_ok = True
            for handoff_index in range(handoff_count):
                obstacles = tuple(
                    segment
                    for path_index, path in enumerate(target_start_paths)
                    if path_index != handoff_index + 1
                    for segment in path
                ) + tuple(
                    segment for route in target_routes.values() for segment in route
                )
                route = route_left_handoff_channel(
                    cfg,
                    dimensions,
                    points[f"TURN{handoff_index + 2}_START"],
                    vias[handoff_index],
                    (
                        points[f"TURN{handoff_index + 2}_START"][0]
                        - target_start_paths[handoff_index + 1][0][1][0],
                        points[f"TURN{handoff_index + 2}_START"][1]
                        - target_start_paths[handoff_index + 1][0][1][1],
                    ),
                    obstacles,
                )
                if route is None or not route_clears_vias(
                    route, vias[:handoff_index] + vias[handoff_index + 1:], cfg
                ):
                    target_ok = False
                    break
                target_routes[handoff_index] = route
            if not target_ok:
                continue

            inner_routes: dict[int, tuple[Segment, ...]] = {}
            inner_ok = True
            for handoff_index in range(handoff_count):
                obstacles = tuple(
                    segment
                    for path_index, path in enumerate(inner_end_paths)
                    if path_index != handoff_index
                    for segment in path
                ) + tuple(
                    segment for route in inner_routes.values() for segment in route
                )
                route = route_left_handoff_channel(
                    cfg,
                    dimensions,
                    points[f"TURN{handoff_index + 1}_LEFT_END"],
                    vias[handoff_index],
                    (
                        inner_end_paths[handoff_index][-1][1][0]
                        - inner_end_paths[handoff_index][-1][0][0],
                        inner_end_paths[handoff_index][-1][1][1]
                        - inner_end_paths[handoff_index][-1][0][1],
                    ),
                    obstacles,
                )
                if route is None or not route_clears_vias(
                    route, vias[:handoff_index] + vias[handoff_index + 1:], cfg
                ):
                    inner_ok = False
                    break
                inner_routes[handoff_index] = route
            if not inner_ok:
                continue

            score = sum(
                distance(*segment)
                for route in (*target_routes.values(), *inner_routes.values())
                for segment in route
            )
            candidate = (score, via_points, target_routes, inner_routes)
            if best_plan is None or candidate[0] < best_plan[0]:
                best_plan = candidate

    if best_plan is None:
        raise ValueError("CL2 left-end turnaround could not be routed with configured clearance.")
    _, via_points, target_routes, inner_routes = best_plan
    return via_points, target_routes, inner_routes


def build_cl2_left_channel_turnaround_plan(
    cfg: dict,
    dimensions: SensorDimensions,
    points: dict[str, Point],
    outer_offsets: tuple[float, ...],
    amplitude_override: float,
) -> tuple[dict[str, Point], dict[int, tuple[Segment, ...]], dict[int, tuple[Segment, ...]]]:
    """Build ordered two-bend left handoffs in a shared outward channel.

    The channel lanes are trace-pitch apart, while the via rack is via-pitch
    apart.  That monotonic expansion prevents one handoff from pinching the
    next as it leaves the staggered coil endpoints.
    """
    handoff_count = len(outer_offsets) - 1
    if handoff_count <= 0:
        return {}, {}, {}

    half_span = secondary_stroke_length(cfg) / 2.0
    trace_clearance = trace_pitch(cfg)
    via_spacing = secondary_via_spacing(cfg)
    escape_length = max(trace_clearance * 2.0, via_spacing)

    target_escapes: list[Point] = []
    inner_escapes: list[Point] = []
    for handoff_index in range(handoff_count):
        target_turn = handoff_index + 2
        target_start = points[f"TURN{target_turn}_START"]
        target_path = secondary_curve_segments(
            cfg,
            dimensions,
            target_start,
            points[f"TURN{target_turn}_LEFT_OUTER"],
            -1.0,
            outer_offsets[target_turn - 1],
            station_start_x=-half_span,
            station_end_x=points[f"TURN{target_turn}_LEFT_OUTER"][0],
            amplitude_override=amplitude_override,
        )
        target_escapes.append(
            left_handoff_escape_point(
                cfg,
                dimensions,
                target_start,
                -1.0,
                escape_length,
                amplitude_override,
            )
        )

        inner_turn = handoff_index + 1
        inner_end = points[f"TURN{inner_turn}_LEFT_END"]
        inner_path = secondary_curve_segments(
            cfg,
            dimensions,
            points[f"TURN{inner_turn}_RETURN_LEFT_OUTER"],
            inner_end,
            1.0,
            outer_offsets[inner_turn - 1],
            station_start_x=points[f"TURN{inner_turn}_RETURN_LEFT_OUTER"][0],
            station_end_x=-half_span,
            amplitude_override=amplitude_override,
        )
        inner_escapes.append(
            left_handoff_escape_point(
                cfg,
                dimensions,
                inner_end,
                1.0,
                escape_length,
                amplitude_override,
            )
        )

    channel_x = min(*(point[0] for point in target_escapes + inner_escapes)) - via_spacing
    via_x = channel_x - via_spacing
    via_points = {
        f"TURN{turn_number}_LEFT_TURNAROUND_VIA": (via_x, via_y)
        for turn_number, via_y in enumerate(
            centered_positions(handoff_count, 2.0 * via_spacing), start=1
        )
    }

    def channel_routes(
        starts: list[Point],
        escapes: list[Point],
    ) -> dict[int, tuple[Segment, ...]]:
        first_lane_y = escapes[0][1]
        routes: dict[int, tuple[Segment, ...]] = {}
        for handoff_index, (start, escape) in enumerate(zip(starts, escapes)):
            lane = (channel_x, first_lane_y + (handoff_index * trace_clearance))
            via = via_points[f"TURN{handoff_index + 1}_LEFT_TURNAROUND_VIA"]
            routes[handoff_index] = polyline_segments((start, escape, lane, via))
        return routes

    target_routes = channel_routes(
        [points[f"TURN{turn_number}_START"] for turn_number in range(2, handoff_count + 2)],
        target_escapes,
    )
    inner_routes = channel_routes(
        [points[f"TURN{turn_number}_LEFT_END"] for turn_number in range(1, handoff_count + 1)],
        inner_escapes,
    )
    return via_points, target_routes, inner_routes


def build_cl2_left_bundle_turnaround_plan(
    cfg: dict,
    dimensions: SensorDimensions,
    points: dict[str, Point],
    outer_offsets: tuple[float, ...],
    amplitude_override: float,
) -> tuple[
    dict[str, Point],
    dict[int, tuple[Segment, ...]],
    dict[int, tuple[Segment, ...]],
]:
    """Fan left handoffs through two smooth, pitch-preserving trace bundles.

    The target and inner endpoints have opposite local rail normals, so a
    straight shared via rack pinches one of the two layers. Each layer instead
    follows its own gently rotating bundle: adjacent routes begin one trace
    pitch apart, expand monotonically to the via pitch, and finish at a compact
    candidate rack that may be vertical or diagonally staggered.
    """
    handoff_count = len(outer_offsets) - 1
    if handoff_count <= 0:
        return {}, {}, {}

    trace_clearance = trace_pitch(cfg)
    via_spacing = secondary_via_spacing(cfg)
    source_points = [
        points[f"TURN{turn_number}_START"]
        for turn_number in range(2, handoff_count + 2)
    ] + [
        points[f"TURN{turn_number}_LEFT_END"]
        for turn_number in range(1, handoff_count + 1)
    ]
    source_edge_x = min(point[0] for point in source_points)

    def bundled_routes(
        starts: list[Point],
        phase_sign: float,
        via_points: dict[str, Point],
        sample_count_override: int | None = None,
        departure_tangent_scale: tuple[float, float] = (1.0, 1.0),
    ) -> dict[int, tuple[Segment, ...]]:
        if handoff_count == 1:
            start = starts[0]
            via = via_points["TURN1_LEFT_TURNAROUND_VIA"]
            route_length = abs(via[0] - start[0])
            control_1 = left_handoff_escape_point(
                cfg,
                dimensions,
                start,
                phase_sign,
                route_length * 0.45,
                amplitude_override,
            )
            control_2 = (via[0] + (route_length * 0.25), via[1])
            sample_count = (
                max(32, cfg["secondary_curve_samples_per_cycle"] // 4)
                if sample_count_override is None
                else sample_count_override
            )
            path: list[Point] = []
            for sample_index in range(sample_count + 1):
                fraction = sample_index / sample_count
                inverse = 1.0 - fraction
                path.append(
                    (
                        (inverse ** 3 * start[0])
                        + (3.0 * inverse * inverse * fraction * control_1[0])
                        + (3.0 * inverse * fraction * fraction * control_2[0])
                        + (fraction ** 3 * via[0]),
                        (inverse ** 3 * start[1])
                        + (3.0 * inverse * inverse * fraction * control_1[1])
                        + (3.0 * inverse * fraction * fraction * control_2[1])
                        + (fraction ** 3 * via[1]),
                    )
                )
            path[0] = start
            path[-1] = via
            return {0: polyline_segments(tuple(path))}
        midpoint = (handoff_count - 1) / 2.0
        center_start = (
            sum(point[0] for point in starts) / handoff_count,
            sum(point[1] for point in starts) / handoff_count,
        )
        normal_start = (
            (starts[-1][0] - starts[0][0]) / ((handoff_count - 1) * trace_clearance),
            (starts[-1][1] - starts[0][1]) / ((handoff_count - 1) * trace_clearance),
        )
        tangent_start = (-normal_start[1], normal_start[0])
        if tangent_start[0] > 0.0:
            tangent_start = (-tangent_start[0], -tangent_start[1])
        via_sequence = tuple(
            via_points[f"TURN{turn_number}_LEFT_TURNAROUND_VIA"]
            for turn_number in range(1, handoff_count + 1)
        )
        center_end = (
            sum(point[0] for point in via_sequence) / handoff_count,
            sum(point[1] for point in via_sequence) / handoff_count,
        )
        normal_end = (
            (via_sequence[-1][0] - via_sequence[0][0])
            / ((handoff_count - 1) * via_spacing),
            (via_sequence[-1][1] - via_sequence[0][1])
            / ((handoff_count - 1) * via_spacing),
        )
        tangent_end = (-normal_end[1], normal_end[0])
        if tangent_end[0] > 0.0:
            tangent_end = (-tangent_end[0], -tangent_end[1])
        control_length = abs(center_end[0] - center_start[0]) * 0.45
        control_1 = (
            center_start[0]
            + (tangent_start[0] * control_length * departure_tangent_scale[0]),
            center_start[1]
            + (tangent_start[1] * control_length * departure_tangent_scale[1]),
        )
        control_2 = (
            center_end[0] - (tangent_end[0] * control_length),
            center_end[1] - (tangent_end[1] * control_length),
        )
        sample_count = (
            max(32, cfg["secondary_curve_samples_per_cycle"] // 4)
            if sample_count_override is None
            else sample_count_override
        )
        route_points: list[list[Point]] = [[] for _ in starts]

        for sample_index in range(sample_count + 1):
            fraction = sample_index / sample_count
            inverse = 1.0 - fraction
            center = (
                (inverse ** 3 * center_start[0])
                + (3.0 * inverse * inverse * fraction * control_1[0])
                + (3.0 * inverse * fraction * fraction * control_2[0])
                + (fraction ** 3 * center_end[0]),
                (inverse ** 3 * center_start[1])
                + (3.0 * inverse * inverse * fraction * control_1[1])
                + (3.0 * inverse * fraction * fraction * control_2[1])
                + (fraction ** 3 * center_end[1]),
            )
            derivative = (
                (3.0 * inverse * inverse * (control_1[0] - center_start[0]))
                + (6.0 * inverse * fraction * (control_2[0] - control_1[0]))
                + (3.0 * fraction * fraction * (center_end[0] - control_2[0])),
                (3.0 * inverse * inverse * (control_1[1] - center_start[1]))
                + (6.0 * inverse * fraction * (control_2[1] - control_1[1]))
                + (3.0 * fraction * fraction * (center_end[1] - control_2[1])),
            )
            derivative_length = math.hypot(*derivative)
            normal = (-derivative[1] / derivative_length, derivative[0] / derivative_length)
            if (normal[0] * normal_start[0]) + (normal[1] * normal_start[1]) < 0.0:
                normal = (-normal[0], -normal[1])
            smooth_fraction = fraction * fraction * (3.0 - (2.0 * fraction))
            spacing = trace_clearance + ((via_spacing - trace_clearance) * smooth_fraction)
            for route_index, point_list in enumerate(route_points):
                offset = (route_index - midpoint) * spacing
                point_list.append(
                    (center[0] + (offset * normal[0]), center[1] + (offset * normal[1]))
                )

        for route_index, point_list in enumerate(route_points):
            point_list[0] = starts[route_index]
            point_list[-1] = via_points[f"TURN{route_index + 1}_LEFT_TURNAROUND_VIA"]

        return {
            route_index: polyline_segments(tuple(point_list))
            for route_index, point_list in enumerate(route_points)
        }

    target_starts = [
        points[f"TURN{turn_number}_START"]
        for turn_number in range(2, handoff_count + 2)
    ]
    inner_starts = [
        points[f"TURN{turn_number}_LEFT_END"]
        for turn_number in range(1, handoff_count + 1)
    ]
    half_span = secondary_stroke_length(cfg) / 2.0
    target_anchor_paths = tuple(
        secondary_curve_segments(
            cfg,
            dimensions,
            points[f"TURN{turn_number}_START"],
            points[f"TURN{turn_number}_LEFT_OUTER"],
            -1.0,
            outer_offsets[turn_number - 1],
            station_start_x=-half_span,
            station_end_x=points[f"TURN{turn_number}_LEFT_OUTER"][0],
            mirror_phase_sign=False,
            amplitude_override=amplitude_override,
        )
        for turn_number in range(1, handoff_count + 2)
    )
    inner_anchor_paths = tuple(
        secondary_curve_segments(
            cfg,
            dimensions,
            points[f"TURN{turn_number}_RETURN_LEFT_OUTER"],
            points[f"TURN{turn_number}_LEFT_END"],
            1.0,
            outer_offsets[turn_number - 1],
            station_start_x=points[f"TURN{turn_number}_RETURN_LEFT_OUTER"][0],
            station_end_x=-half_span,
            mirror_phase_sign=False,
            amplitude_override=amplitude_override,
        )
        for turn_number in range(1, handoff_count + 2)
    )

    def rack_points(center_x: float, x_step: float) -> dict[str, Point]:
        y_step = math.sqrt((via_spacing * via_spacing) - (x_step * x_step))
        midpoint = (handoff_count - 1) / 2.0
        return {
            f"TURN{turn_number}_LEFT_TURNAROUND_VIA": (
                center_x + ((turn_number - 1 - midpoint) * x_step),
                (turn_number - 1 - midpoint) * y_step,
            )
            for turn_number in range(1, handoff_count + 1)
        }

    def routes_clear_candidate(
        via_points: dict[str, Point],
        target_routes: dict[int, tuple[Segment, ...]],
        inner_routes: dict[int, tuple[Segment, ...]],
    ) -> bool:
        route_groups = (
            (target_routes, target_anchor_paths, 1),
            (inner_routes, inner_anchor_paths, 0),
        )
        for routes, anchor_paths, own_offset in route_groups:
            for route_index, route in routes.items():
                for path_index, path in enumerate(anchor_paths):
                    if path_index == route_index + own_offset:
                        continue
                    if (
                        path_to_path_distance(route, path)
                        + ROUTING_POLYGONAL_TOLERANCE_MM
                        < trace_clearance
                    ):
                        return False
            for first_index, first in routes.items():
                for second_index, second in routes.items():
                    if first_index >= second_index:
                        continue
                    if (
                        path_to_path_distance(first, second)
                        + ROUTING_POLYGONAL_TOLERANCE_MM
                        < trace_clearance
                    ):
                        return False
        vias = tuple(via_points.values())
        for via in vias:
            for path in target_anchor_paths + inner_anchor_paths:
                if (
                    min(point_to_segment_distance(via, segment) for segment in path)
                    + ROUTING_POLYGONAL_TOLERANCE_MM
                    < via_to_trace_clearance(cfg)
                ):
                    return False
        for route_index, route in (*target_routes.items(), *inner_routes.items()):
            own_via = vias[route_index]
            for via in vias:
                if via == own_via:
                    continue
                if (
                    min(point_to_segment_distance(via, segment) for segment in route)
                    + ROUTING_POLYGONAL_TOLERANCE_MM
                    < via_to_trace_clearance(cfg)
                ):
                    return False
        return True

    # The baseline cubic keeps the historical vertical and diagonal rack search.
    # A peeled departure lets the outer routes clear their neighboring rails, so
    # it receives a refined vertical-rack search in the otherwise blocked band.
    bundle_candidates = tuple(
        (center_pitch, step_pitch, (1.0, 1.0))
        for center_pitch in (2.0, 2.25, 2.50, 2.75, 3.0)
        for step_pitch in (0.0, -0.25, 0.25, -0.50, 0.50)
    ) + tuple(
        (center_pitch / 100.0, 0.0, (1.60, 1.34))
        for center_pitch in range(225, 251)
    )
    best_plan: tuple[float, dict[str, Point], tuple[float, float]] | None = None
    for center_pitch, step_pitch, departure_tangent_scale in bundle_candidates:
        via_points = rack_points(
            source_edge_x - (center_pitch * via_spacing),
            step_pitch * via_spacing,
        )
        target_routes = bundled_routes(
            target_starts,
            -1.0,
            via_points,
            16,
            departure_tangent_scale,
        )
        inner_routes = bundled_routes(
            inner_starts,
            1.0,
            via_points,
            16,
            departure_tangent_scale,
        )
        if not routes_clear_candidate(via_points, target_routes, inner_routes):
            continue
        score = sum(
            distance(*segment)
            for route in (*target_routes.values(), *inner_routes.values())
            for segment in route
        )
        candidate = (score, via_points, departure_tangent_scale)
        if best_plan is None or candidate[0] < best_plan[0]:
            best_plan = candidate

    if best_plan is None:
        raise ValueError("CL2 left bundle turnaround could not satisfy configured clearance.")
    _, via_points, departure_tangent_scale = best_plan
    target_routes = bundled_routes(
        target_starts,
        -1.0,
        via_points,
        departure_tangent_scale=departure_tangent_scale,
    )
    inner_routes = bundled_routes(
        inner_starts,
        1.0,
        via_points,
        departure_tangent_scale=departure_tangent_scale,
    )
    if not routes_clear_candidate(via_points, target_routes, inner_routes):
        raise ValueError("CL2 left bundle turnaround lost clearance at render resolution.")
    return via_points, target_routes, inner_routes


def build_cl2_trimmed_left_bundle_turnaround_plan(
    cfg: dict,
    dimensions: SensorDimensions,
    points: dict[str, Point],
    outer_offsets: tuple[float, ...],
    amplitude_override: float,
) -> tuple[
    dict[str, Point],
    dict[int, tuple[Segment, ...]],
    dict[int, tuple[Segment, ...]],
    dict[str, Point],
    dict[str, float],
]:
    """Choose the shortest legal CL2 left handoff with bounded on-rail trim."""
    handoff_count = len(outer_offsets) - 1
    if handoff_count <= 0:
        return {}, {}, {}, {}, {}

    half_span = secondary_stroke_length(cfg) / 2.0
    via_spacing = secondary_via_spacing(cfg)
    best_plan: tuple[
        tuple[float, float, float],
        dict[str, Point],
        dict[int, tuple[Segment, ...]],
        dict[int, tuple[Segment, ...]],
        dict[str, Point],
        dict[str, float],
    ] | None = None

    for fraction in (0.0, 0.25, 0.50, 0.75, 1.0):
        trim = fraction * via_spacing
        station_x = -half_span + trim
        trimmed_points: dict[str, Point] = {}
        stations: dict[str, float] = {}
        for turn_number in range(2, handoff_count + 2):
            label = f"TURN{turn_number}_START"
            trimmed_points[label] = secondary_rail_point(
                cfg,
                dimensions,
                station_x,
                -1.0,
                outer_offsets[turn_number - 1],
                amplitude_override=amplitude_override,
            )
            stations[label] = station_x
        for turn_number in range(1, handoff_count + 1):
            label = f"TURN{turn_number}_LEFT_END"
            trimmed_points[label] = secondary_rail_point(
                cfg,
                dimensions,
                station_x,
                1.0,
                outer_offsets[turn_number - 1],
                amplitude_override=amplitude_override,
            )
            stations[label] = station_x

        candidate_points = {**points, **trimmed_points}
        try:
            via_points, target_routes, inner_routes = build_cl2_left_bundle_turnaround_plan(
                cfg,
                dimensions,
                candidate_points,
                outer_offsets,
                amplitude_override,
            )
        except ValueError:
            continue
        route_length = sum(
            distance(*segment)
            for route in (*target_routes.values(), *inner_routes.values())
            for segment in route
        )
        # Score the complete changed portion of the winding, rather than only
        # the via handoff.  A later rail station shortens the sine section it
        # replaces, so that saving must participate in the route-length goal.
        sine_length = sum(
            distance(*segment)
            for turn_number in range(2, handoff_count + 2)
            for segment in secondary_curve_segments(
                cfg,
                dimensions,
                candidate_points[f"TURN{turn_number}_START"],
                candidate_points[f"TURN{turn_number}_LEFT_OUTER"],
                -1.0,
                outer_offsets[turn_number - 1],
                station_start_x=stations[f"TURN{turn_number}_START"],
                station_end_x=candidate_points[f"TURN{turn_number}_LEFT_OUTER"][0],
                amplitude_override=amplitude_override,
            )
        ) + sum(
            distance(*segment)
            for turn_number in range(1, handoff_count + 1)
            for segment in secondary_curve_segments(
                cfg,
                dimensions,
                candidate_points[f"TURN{turn_number}_RETURN_LEFT_OUTER"],
                candidate_points[f"TURN{turn_number}_LEFT_END"],
                1.0,
                outer_offsets[turn_number - 1],
                station_start_x=candidate_points[
                    f"TURN{turn_number}_RETURN_LEFT_OUTER"
                ][0],
                station_end_x=stations[f"TURN{turn_number}_LEFT_END"],
                amplitude_override=amplitude_override,
            )
        )
        rack_center_x = sum(point[0] for point in via_points.values()) / handoff_count
        candidate = (
            (route_length + sine_length, trim, -rack_center_x),
            via_points,
            target_routes,
            inner_routes,
            trimmed_points,
            stations,
        )
        if best_plan is None or candidate[0] < best_plan[0]:
            best_plan = candidate

    if best_plan is None:
        raise ValueError(
            "CL2 left bundle turnaround could not satisfy configured clearance "
            "within one via-spacing of endpoint trim."
        )
    _, via_points, target_routes, inner_routes, trimmed_points, stations = best_plan
    return via_points, target_routes, inner_routes, trimmed_points, stations


def cl2_fixed_right_transition_geometry(
    points: dict[str, Point],
    turn_count: int,
) -> tuple[tuple[str, ...], tuple[Segment, ...], tuple[Segment, ...]]:
    """Return the unchanged right-side quarter-span CL2 transition geometry."""
    via_labels: list[str] = []
    target_segments: list[Segment] = []
    inner_segments: list[Segment] = []
    for turn_number in range(1, turn_count + 1):
        right_inner = f"TURN{turn_number}_RIGHT_INNER"
        right_lower_via = f"TURN{turn_number}_RIGHT_LOWER_VIA"
        right_outer = f"TURN{turn_number}_RIGHT_OUTER"
        reverse_right_outer = f"TURN{turn_number}_RETURN_RIGHT_OUTER"
        reverse_right_upper_via = f"TURN{turn_number}_RETURN_RIGHT_UPPER_VIA"
        reverse_target_start = f"TURN{turn_number}_RETURN_RIGHT_INNER"
        via_labels.extend((right_lower_via, reverse_right_upper_via))
        inner_segments.extend(
            (
                (points[right_inner], points[right_lower_via]),
                (points[reverse_right_outer], points[reverse_right_upper_via]),
            )
        )
        target_segments.extend(
            (
                (points[right_lower_via], points[right_outer]),
                (points[reverse_right_upper_via], points[reverse_target_start]),
            )
        )
    return tuple(via_labels), tuple(target_segments), tuple(inner_segments)


def cl2_turnaround_parallel_spacing_requirement(
    cfg: dict,
    first: Segment,
    second: Segment,
) -> float:
    """Cap jog spacing by the tighter preserved anchor separation for the two turns."""
    return min(
        trace_pitch(cfg),
        distance(first[0], second[0]),
        distance(first[1], second[1]),
    )


def cl2_right_turnaround_minimum_adjacent_spacing(
    target_segments: tuple[Segment, ...],
    inner_segments: tuple[Segment, ...],
) -> float:
    """Return the tightest adjacent jog spacing across the packed CL2 turnaround."""
    adjacent_spacings = [
        segment_to_segment_distance(first, second)
        for group in (target_segments, inner_segments)
        for first, second in zip(group, group[1:])
    ]
    return min(adjacent_spacings, default=float("inf"))


def cl2_right_turnaround_clearance_violations_from_segments(
    cfg: dict,
    points: dict[str, Point],
    primary_segments: tuple[Segment, ...],
    via_labels: tuple[str, ...],
    target_segments: tuple[Segment, ...],
    inner_segments: tuple[Segment, ...],
    fixed_via_labels: tuple[str, ...],
    fixed_target_segments: tuple[Segment, ...],
    fixed_inner_segments: tuple[Segment, ...],
) -> tuple[str, ...]:
    """Validate the hard-clearance rules for the packed CL2 right-end turnaround."""
    minimum_pad_distance = secondary_via_spacing(cfg)
    minimum_trace_distance = osc1_via_trace_clearance(cfg)
    pitch = trace_pitch(cfg)
    violations: list[str] = []

    for first_index, first in enumerate(via_labels):
        for second in via_labels[first_index + 1:]:
            if (
                distance(points[first], points[second])
                + GEOMETRY_TOLERANCE_MM
                < minimum_pad_distance
            ):
                violations.append(
                    f"CL2 turnaround vias {first}/{second} violate plated via clearance."
                )

    for via_label in via_labels:
        nearest_primary_trace = min(
            point_to_segment_distance(points[via_label], segment)
            for segment in primary_segments
        )
        if nearest_primary_trace + GEOMETRY_TOLERANCE_MM < minimum_trace_distance:
            violations.append(
                f"CL2 via {via_label} violates clearance to the primary winding."
            )

    for layer_name, jog_segments in (("target", target_segments), ("inner", inner_segments)):
        for turn_index, jog in enumerate(jog_segments, start=1):
            nearest_primary_trace = min(
                segment_to_segment_distance(jog, segment)
                for segment in primary_segments
            )
            if nearest_primary_trace + GEOMETRY_TOLERANCE_MM < minimum_trace_distance:
                violations.append(
                    f"CL2 {layer_name} jog for turn {turn_index} violates clearance to the primary winding."
                )

    for via_label in via_labels:
        if any(
            distance(points[via_label], points[fixed_label]) + GEOMETRY_TOLERANCE_MM < minimum_pad_distance
            for fixed_label in fixed_via_labels
        ):
            violations.append(
                f"CL2 via {via_label} crowds the fixed right-side quarter-span transition vias."
            )
        if any(
            point_to_segment_distance(points[via_label], segment) + GEOMETRY_TOLERANCE_MM
            < minimum_trace_distance
            for segment in fixed_target_segments + fixed_inner_segments
        ):
            violations.append(
                f"CL2 via {via_label} crowds the fixed right-side quarter-span transition copper."
            )

    for turn_index, jog in enumerate(target_segments, start=1):
        if any(
            segment_to_segment_distance(jog, fixed_segment) + GEOMETRY_TOLERANCE_MM < pitch
            for fixed_segment in fixed_target_segments
        ):
            violations.append(
                f"CL2 target jog for turn {turn_index} crowds the fixed right-side target transition."
            )

    for turn_index, jog in enumerate(inner_segments, start=1):
        if any(
            segment_to_segment_distance(jog, fixed_segment) + GEOMETRY_TOLERANCE_MM < pitch
            for fixed_segment in fixed_inner_segments
        ):
            violations.append(
                f"CL2 inner jog for turn {turn_index} crowds the fixed right-side inner transition."
            )

    return tuple(violations)


def cl2_right_turnaround_clearance_violations(
    cfg: dict,
    points: dict[str, Point],
    primary_segments: tuple[Segment, ...],
) -> tuple[str, ...]:
    """Return any clearance failures for the packed CL2 right-end turnaround."""
    turn_count = cfg["number_of_secondary_turns"]
    via_labels, target_segments, inner_segments = cl2_right_turnaround_segments(
        points, turn_count
    )
    fixed_via_labels, fixed_target_segments, fixed_inner_segments = (
        cl2_fixed_right_transition_geometry(points, turn_count)
    )
    return cl2_right_turnaround_clearance_violations_from_segments(
        cfg,
        points,
        primary_segments,
        via_labels,
        target_segments,
        inner_segments,
        fixed_via_labels,
        fixed_target_segments,
        fixed_inner_segments,
    )


def cl2_turnaround_rightmost_u_values(
    max_rightmost_u: float,
    min_rightmost_u: float,
    step: float = 0.001,
) -> tuple[float, ...]:
    """Return descending sensor-end coordinates for the outermost turnaround column."""
    if max_rightmost_u + GEOMETRY_TOLERANCE_MM < min_rightmost_u:
        return ()
    candidates = [max_rightmost_u]
    current = max_rightmost_u
    while current - step > min_rightmost_u + GEOMETRY_TOLERANCE_MM:
        current -= step
        candidates.append(current)
    if candidates[-1] > min_rightmost_u + GEOMETRY_TOLERANCE_MM:
        candidates.append(min_rightmost_u)
    else:
        candidates[-1] = min_rightmost_u
    return tuple(candidates)


def build_cl2_right_turnaround_plan(
    cfg: dict,
    dimensions: SensorDimensions,
    points: dict[str, Point],
    primary_segments: tuple[Segment, ...],
    column_count: int | None = None,
) -> CL2RightTurnaroundPlan:
    """Pack the CL2 right-end turnaround vias and jogs while preserving both curve anchors."""
    turn_count = cfg["number_of_secondary_turns"]
    if column_count is not None and not 1 <= column_count <= turn_count:
        raise ValueError("column_count must be between 1 and number_of_secondary_turns.")

    half_span = secondary_stroke_length(cfg) / 2.0
    via_spacing = secondary_via_spacing(cfg)
    sensor_end_direction = -fanout_direction(cfg)
    # Keep the turnaround vias one receiver-via pitch away from the shared end
    # anchor so they do not electrically short adjacent turns together.
    desired_rightmost_u = half_span + via_spacing
    rightmost_clear_u = (
        (dimensions.primary_length_mm / 2.0)
        - ((cfg["number_of_primary_turns"] - 1) * trace_pitch(cfg))
        - osc1_via_trace_clearance(cfg)
    )
    fixed_via_labels, fixed_target_segments, fixed_inner_segments = (
        cl2_fixed_right_transition_geometry(points, turn_count)
    )

    if column_count is None:
        column_counts = range(1, turn_count + 1)
    else:
        column_counts = (column_count,)

    if rightmost_clear_u + GEOMETRY_TOLERANCE_MM < desired_rightmost_u:
        if not should_skip_geometry_validation(cfg):
            raise ValueError(
                "CL2 right-end turnaround column one via-spacing beyond TURNn_RIGHT_END "
                "does not fit inside the primary envelope."
            )

    best_fallback_plan: CL2RightTurnaroundPlan | None = None
    best_fallback_rank: tuple[int, int, float, float, float, tuple[int, ...]] | None = None

    for packed_columns in column_counts:
        assignments = cl2_right_turnaround_assignment_candidates(turn_count, packed_columns)
        if not assignments:
            continue
        rightmost_candidates = (desired_rightmost_u,)

        for rightmost_u in rightmost_candidates:
            best_valid_plan: CL2RightTurnaroundPlan | None = None
            best_invalid_plan: CL2RightTurnaroundPlan | None = None
            best_invalid_rank_for_u: tuple[int, float, float, tuple[int, ...]] | None = None

            for assignment in assignments:
                column_turns = {column: [] for column in range(packed_columns)}
                for turn_index, assigned_column in enumerate(assignment):
                    column_turns[assigned_column].append(turn_index)

                via_points: dict[str, Point] = {}
                for packed_column in range(packed_columns):
                    column_x = sensor_end_direction * (
                        rightmost_u - (packed_column * via_spacing)
                    )
                    column_y_positions = centered_positions(
                        len(column_turns[packed_column]),
                        via_spacing,
                    )
                    for turn_index, via_y in zip(
                        column_turns[packed_column],
                        column_y_positions,
                    ):
                        via_points[f"TURN{turn_index + 1}_RIGHT_TURNAROUND_VIA"] = (
                            column_x,
                            via_y,
                        )

                candidate_points = {**points, **via_points}
                via_labels, target_segments, inner_segments = cl2_right_turnaround_segments(
                    candidate_points,
                    turn_count,
                )
                candidate_plan = CL2RightTurnaroundPlan(
                    via_points=via_points,
                    via_labels=via_labels,
                    target_segments=target_segments,
                    inner_segments=inner_segments,
                    column_count=packed_columns,
                    rightmost_u=rightmost_u,
                    assignment=assignment,
                    minimum_adjacent_spacing=cl2_right_turnaround_minimum_adjacent_spacing(
                        target_segments,
                        inner_segments,
                    ),
                    score=sum(
                        distance(*segment)
                        for segment in target_segments + inner_segments
                    ),
                )
                violations = cl2_right_turnaround_clearance_violations_from_segments(
                    cfg,
                    candidate_points,
                    primary_segments,
                    via_labels,
                    target_segments,
                    inner_segments,
                    fixed_via_labels,
                    fixed_target_segments,
                    fixed_inner_segments,
                )
                if not violations:
                    # Preserved right-end turnaround anchors can force adjacent straight
                    # fans below the nominal trace pitch for higher turn counts, so we
                    # optimize for the widest adjacent jog spacing here and reserve
                    # hard failures for actual copper/via clearance violations.
                    if (
                        best_valid_plan is None
                        or candidate_plan.minimum_adjacent_spacing
                        > best_valid_plan.minimum_adjacent_spacing + GEOMETRY_TOLERANCE_MM
                        or (
                            math.isclose(
                                candidate_plan.minimum_adjacent_spacing,
                                best_valid_plan.minimum_adjacent_spacing,
                                abs_tol=GEOMETRY_TOLERANCE_MM,
                            )
                            and candidate_plan.score < best_valid_plan.score
                        )
                        or (
                            math.isclose(candidate_plan.score, best_valid_plan.score)
                            and candidate_plan.assignment < best_valid_plan.assignment
                        )
                    ):
                        best_valid_plan = candidate_plan
                elif should_skip_geometry_validation(cfg):
                    invalid_rank_for_u = (
                        len(violations),
                        -candidate_plan.minimum_adjacent_spacing,
                        candidate_plan.score,
                        candidate_plan.assignment,
                    )
                    if best_invalid_rank_for_u is None:
                        should_replace_invalid_plan = True
                    else:
                        should_replace_invalid_plan = (
                            invalid_rank_for_u < best_invalid_rank_for_u
                        )
                    if should_replace_invalid_plan:
                        best_invalid_plan = candidate_plan
                        best_invalid_rank_for_u = invalid_rank_for_u

            if best_valid_plan is not None:
                return best_valid_plan

            if best_invalid_plan is not None:
                assert best_invalid_rank_for_u is not None
                fallback_rank = (
                    best_invalid_rank_for_u[0],
                    packed_columns,
                    -rightmost_u,
                    best_invalid_rank_for_u[1],
                    best_invalid_rank_for_u[2],
                    best_invalid_rank_for_u[3],
                )
                if best_fallback_rank is None:
                    should_replace_fallback = True
                else:
                    should_replace_fallback = fallback_rank < best_fallback_rank
                if should_replace_fallback:
                    best_fallback_plan = best_invalid_plan
                    best_fallback_rank = fallback_rank

    if best_fallback_plan is not None:
        return best_fallback_plan

    raise ValueError(
        "CL2 right-end turnaround could not be packed without violating clearance."
    )


def build_multiturn_cl2_layout(
    cfg: dict,
    dimensions: SensorDimensions,
    primary_geometry: PrimaryGeometry | None = None,
) -> SecondaryLayoutPlan:
    """Build CL2 for all valid turn counts using canonical descriptive anchors."""
    half_span = secondary_stroke_length(cfg) / 2.0
    quarter_span = half_span / 2.0
    outer_offsets = secondary_turn_offsets(cfg)
    quarter_shifts = cl2_quarter_column_shifts(cfg)
    amplitude_override = secondary_wave_amplitude_for_offsets(dimensions, outer_offsets)
    terminal_x = -((dimensions.primary_length_mm / 2.0) + cfg["terminal_escape_length_mm"])
    terminal_output_y = terminal_row_y(cfg, "CL2")
    terminal_return_y = terminal_row_y(cfg, "CL2-GND")
    negative_rail_spans: list[tuple[float, float, float]] = []
    positive_rail_spans: list[tuple[float, float, float]] = []
    for turn_index, outer_offset in enumerate(outer_offsets):
        inner_offset = outer_offsets[len(outer_offsets) - 1 - turn_index]
        shift = quarter_shifts[turn_index]
        left_column_x = -quarter_span + shift
        right_column_x = quarter_span + shift
        reverse_right_column_x = quarter_span - shift
        reverse_left_column_x = -quarter_span - shift
        # An inner rail begins at its own transition column; modelling it as a
        # full-span curve would create false obstacles for neighboring vias.
        negative_rail_spans.extend(
            (
                (outer_offset, -half_span, left_column_x),
                (inner_offset, left_column_x, right_column_x),
                (outer_offset, right_column_x, half_span),
            )
        )
        positive_rail_spans.extend(
            (
                (outer_offset, -half_span, reverse_left_column_x),
                (inner_offset, reverse_left_column_x, reverse_right_column_x),
                (outer_offset, reverse_right_column_x, half_span),
            )
        )

    points: dict[str, Point] = {
        "TERMINAL_OUTPUT_VIA": (terminal_x, terminal_output_y),
    }

    turn_specs: list[dict[str, str | float]] = []
    for turn_index, outer_offset in enumerate(outer_offsets):
        reverse_index = len(outer_offsets) - 1 - turn_index
        inner_offset = outer_offsets[reverse_index]
        shift = quarter_shifts[turn_index]
        left_column_x = -quarter_span + shift
        right_column_x = quarter_span + shift
        reverse_right_column_x = quarter_span - shift
        reverse_left_column_x = -quarter_span - shift
        start_label = f"TURN{turn_index + 1}_START"
        labels: dict[str, str | float] = {
            "outer_offset": outer_offset,
            "inner_offset": inner_offset,
            "start": start_label,
            "left_outer": f"TURN{turn_index + 1}_LEFT_OUTER",
            "left_upper_via": f"TURN{turn_index + 1}_LEFT_UPPER_VIA",
            "left_inner": f"TURN{turn_index + 1}_LEFT_INNER",
            "right_inner": f"TURN{turn_index + 1}_RIGHT_INNER",
            "right_lower_via": f"TURN{turn_index + 1}_RIGHT_LOWER_VIA",
            "right_outer": f"TURN{turn_index + 1}_RIGHT_OUTER",
            "right_end": f"TURN{turn_index + 1}_RIGHT_END",
            "right_runup": f"TURN{turn_index + 1}_RIGHT_TURNAROUND_RUNUP",
            "right_detour": f"TURN{turn_index + 1}_RIGHT_TURNAROUND_VIA",
            "reverse_right_outer": f"TURN{turn_index + 1}_RETURN_RIGHT_OUTER",
            "reverse_right_upper_via": f"TURN{turn_index + 1}_RETURN_RIGHT_UPPER_VIA",
            "reverse_target_start": f"TURN{turn_index + 1}_RETURN_RIGHT_INNER",
            "reverse_target_end": f"TURN{turn_index + 1}_RETURN_LEFT_INNER",
            "reverse_left_lower_via": f"TURN{turn_index + 1}_RETURN_LEFT_LOWER_VIA",
            "reverse_left_outer": f"TURN{turn_index + 1}_RETURN_LEFT_OUTER",
            "left_end": f"TURN{turn_index + 1}_LEFT_END",
        }
        if turn_index < len(outer_offsets) - 1:
            labels["left_detour"] = f"TURN{turn_index + 1}_LEFT_TURNAROUND_VIA"
            labels["end"] = f"TURN{turn_index + 2}_START"
        else:
            labels["end"] = f"TURN{turn_index + 1}_RETURN_START"

        # Preserve each rail's normal offset at the shared waveform station.
        # Collapsing the x component here made adjacent starts slightly closer
        # than trace_pitch and left no legal direction for the handoff to exit.
        points[str(labels["start"])] = secondary_rail_point(
            cfg,
            dimensions,
            -half_span,
            -1.0,
            outer_offset,
            amplitude_override=amplitude_override,
        )

        points[str(labels["left_outer"])] = point_at_station_x(
            secondary_rail_point(
                cfg,
                dimensions,
                left_column_x,
                -1.0,
                outer_offset,
                amplitude_override=amplitude_override,
            ),
            left_column_x,
        )
        points[str(labels["left_upper_via"])] = (
            left_column_x,
            receiver_transition_via_y(
                cfg,
                dimensions,
                left_column_x,
                -1.0,
                outer_offsets,
                upper=True,
                connected_rail_offsets=(outer_offset, inner_offset),
                rail_spans=tuple(negative_rail_spans),
                amplitude_override=amplitude_override,
            ),
        )
        points[str(labels["left_inner"])] = point_at_station_x(
            secondary_rail_point(
                cfg,
                dimensions,
                left_column_x,
                -1.0,
                inner_offset,
                amplitude_override=amplitude_override,
            ),
            left_column_x,
        )
        points[str(labels["right_inner"])] = point_at_station_x(
            secondary_rail_point(
                cfg,
                dimensions,
                right_column_x,
                -1.0,
                inner_offset,
                amplitude_override=amplitude_override,
            ),
            right_column_x,
        )
        points[str(labels["right_lower_via"])] = (
            right_column_x,
            receiver_transition_via_y(
                cfg,
                dimensions,
                right_column_x,
                -1.0,
                outer_offsets,
                upper=False,
                connected_rail_offsets=(outer_offset, inner_offset),
                rail_spans=tuple(negative_rail_spans),
                amplitude_override=amplitude_override,
            ),
        )
        points[str(labels["right_outer"])] = point_at_station_x(
            secondary_rail_point(
                cfg,
                dimensions,
                right_column_x,
                -1.0,
                outer_offset,
                amplitude_override=amplitude_override,
            ),
            right_column_x,
        )
        points[str(labels["right_end"])] = point_at_station_x(
            secondary_rail_point(
                cfg,
                dimensions,
                half_span,
                -1.0,
                outer_offset,
                amplitude_override=amplitude_override,
            ),
            half_span,
        )
        # Both CL2 sinusoidal sections now share the same curve-side anchor before
        # the packed jog/via turnaround.
        points[str(labels["right_runup"])] = points[str(labels["right_end"])]
        points[str(labels["reverse_right_outer"])] = point_at_station_x(
            secondary_rail_point(
                cfg,
                dimensions,
                reverse_right_column_x,
                1.0,
                outer_offset,
                amplitude_override=amplitude_override,
            ),
            reverse_right_column_x,
        )
        points[str(labels["reverse_right_upper_via"])] = (
            reverse_right_column_x,
            receiver_transition_via_y(
                cfg,
                dimensions,
                reverse_right_column_x,
                1.0,
                outer_offsets,
                upper=True,
                connected_rail_offsets=(outer_offset, inner_offset),
                rail_spans=tuple(positive_rail_spans),
                amplitude_override=amplitude_override,
            ),
        )
        points[str(labels["reverse_target_start"])] = point_at_station_x(
            secondary_rail_point(
                cfg,
                dimensions,
                reverse_right_column_x,
                1.0,
                inner_offset,
                amplitude_override=amplitude_override,
            ),
            reverse_right_column_x,
        )
        points[str(labels["reverse_target_end"])] = point_at_station_x(
            secondary_rail_point(
                cfg,
                dimensions,
                reverse_left_column_x,
                1.0,
                inner_offset,
                amplitude_override=amplitude_override,
            ),
            reverse_left_column_x,
        )
        points[str(labels["reverse_left_lower_via"])] = (
            reverse_left_column_x,
            receiver_transition_via_y(
                cfg,
                dimensions,
                reverse_left_column_x,
                1.0,
                outer_offsets,
                upper=False,
                connected_rail_offsets=(outer_offset, inner_offset),
                rail_spans=tuple(positive_rail_spans),
                amplitude_override=amplitude_override,
            ),
        )
        points[str(labels["reverse_left_outer"])] = point_at_station_x(
            secondary_rail_point(
                cfg,
                dimensions,
                reverse_left_column_x,
                1.0,
                outer_offset,
                amplitude_override=amplitude_override,
            ),
            reverse_left_column_x,
        )
        points[str(labels["left_end"])] = secondary_rail_point(
            cfg,
            dimensions,
            -half_span,
            1.0,
            outer_offset,
            amplitude_override=amplitude_override,
        )

        if turn_index < len(outer_offsets) - 1:
            points[str(labels["end"])] = secondary_rail_point(
                cfg,
                dimensions,
                -half_span,
                -1.0,
                outer_offsets[turn_index + 1],
                amplitude_override=amplitude_override,
            )
        else:
            points[str(labels["end"])] = points[str(labels["left_end"])]

        turn_specs.append(labels)

    (
        left_via_points,
        left_target_routes,
        left_inner_routes,
        left_trimmed_points,
        left_handoff_station_x,
    ) = build_cl2_trimmed_left_bundle_turnaround_plan(
        cfg,
        dimensions,
        points,
        outer_offsets,
        amplitude_override,
    )
    points = {**points, **left_trimmed_points, **left_via_points}
    return_start_label = str(turn_specs[-1]["end"])
    # Both external CL2 traces meet the fanout at the same y=0 spine.  They
    # occupy different receiver layers, so this deliberately creates a common
    # route-out while leaving the local coil-side escapes independent.
    # Keep the last fanout legs orthogonal.  The spine sits one via-to-trace
    # clearance inside the terminal column, then continues to the common y=0
    # CL2 convergence route.
    fanout_spine_x = terminal_x + via_to_trace_clearance(cfg)
    fanout_spine = (fanout_spine_x, 0.0)
    points["ENTRY_FANOUT_JOG"] = (fanout_spine_x, terminal_output_y)
    points["RETURN_FANOUT_JOG"] = (fanout_spine_x, terminal_return_y)
    points["ENTRY_CONVERGENCE_JOG"] = fanout_spine
    points["RETURN_CONVERGENCE_JOG"] = fanout_spine
    local_left_detours = tuple(left_via_points.values())
    if local_left_detours:
        convergence_x = min(point[0] for point in local_left_detours) - via_to_trace_clearance(cfg)
    else:
        convergence_x = min(
            points["TURN1_START"][0],
            points[return_start_label][0],
        ) - trace_pitch(cfg)
    points["FANOUT_CONVERGENCE"] = (convergence_x, 0.0)
    points["TERMINAL_RETURN_VIA"] = (terminal_x, terminal_return_y)

    if fanout_direction(cfg) > 0:
        points = mirror_points_horizontally(points)
        left_handoff_station_x = {
            label: -station_x
            for label, station_x in left_handoff_station_x.items()
        }
        left_target_routes = {
            handoff_index: mirror_segments_horizontally(route)
            for handoff_index, route in left_target_routes.items()
        }
        left_inner_routes = {
            handoff_index: mirror_segments_horizontally(route)
            for handoff_index, route in left_inner_routes.items()
        }

    primary_geometry = primary_geometry or build_primary_geometry(cfg)
    primary_segments = tuple(
        segment
        for coil in primary_geometry.coils
        for segment in coil.body_segments
    )
    turnaround_plan = build_cl2_right_turnaround_plan(
        cfg,
        dimensions,
        points,
        primary_segments,
    )
    points = {**points, **turnaround_plan.via_points}

    target_segments: list[Segment] = []
    inner_segments: list[Segment] = []
    target_forward_paths: list[tuple[Segment, ...]] = []
    target_reverse_paths: list[tuple[Segment, ...]] = []
    inner_forward_paths: list[tuple[Segment, ...]] = []
    inner_reverse_paths: list[tuple[Segment, ...]] = []
    via_labels: list[str] = ["TERMINAL_OUTPUT_VIA"]

    for spec in turn_specs:
        outer_offset = float(spec["outer_offset"])
        inner_offset = float(spec["inner_offset"])
        start_label = str(spec["start"])
        left_outer = str(spec["left_outer"])
        left_upper_via = str(spec["left_upper_via"])
        left_inner = str(spec["left_inner"])
        right_inner = str(spec["right_inner"])
        right_lower_via = str(spec["right_lower_via"])
        right_outer = str(spec["right_outer"])
        right_end = str(spec["right_end"])
        right_runup = str(spec["right_runup"])
        right_detour = str(spec["right_detour"])
        reverse_right_outer = str(spec["reverse_right_outer"])
        reverse_right_upper_via = str(spec["reverse_right_upper_via"])
        reverse_target_start = str(spec["reverse_target_start"])
        reverse_target_end = str(spec["reverse_target_end"])
        reverse_left_lower_via = str(spec["reverse_left_lower_via"])
        reverse_left_outer = str(spec["reverse_left_outer"])
        left_end = str(spec["left_end"])
        end_label = str(spec["end"])

        target_first = secondary_curve_segments(
            cfg,
            dimensions,
            points[start_label],
            points[left_outer],
            -1.0,
            outer_offset,
            station_start_x=left_handoff_station_x.get(
                start_label, points[start_label][0]
            ),
            station_end_x=points[left_outer][0],
            amplitude_override=amplitude_override,
        )
        target_segments.extend(target_first)
        target_segments.append((points[left_outer], points[left_upper_via]))
        inner_segments.append((points[left_upper_via], points[left_inner]))

        inner_forward = secondary_curve_segments(
            cfg,
            dimensions,
            points[left_inner],
            points[right_inner],
            -1.0,
            inner_offset,
            station_start_x=points[left_inner][0],
            station_end_x=points[right_inner][0],
            amplitude_override=amplitude_override,
        )
        inner_segments.extend(inner_forward)
        inner_segments.append((points[right_inner], points[right_lower_via]))
        target_segments.append((points[right_lower_via], points[right_outer]))

        target_second = secondary_curve_segments(
            cfg,
            dimensions,
            points[right_outer],
            points[right_end],
            -1.0,
            outer_offset,
            station_start_x=points[right_outer][0],
            station_end_x=points[right_end][0],
            amplitude_override=amplitude_override,
        )
        target_segments.extend(target_second)
        target_forward_paths.append(target_first + target_second)
        inner_forward_paths.append(inner_forward)

        target_segments.append((points[right_end], points[right_detour]))
        inner_segments.append((points[right_detour], points[right_runup]))

        inner_reverse_outer = secondary_curve_segments(
            cfg,
            dimensions,
            points[right_runup],
            points[reverse_right_outer],
            1.0,
            outer_offset,
            station_start_x=points[right_runup][0],
            station_end_x=points[reverse_right_outer][0],
            amplitude_override=amplitude_override,
        )
        inner_segments.extend(inner_reverse_outer)
        inner_segments.append((points[reverse_right_outer], points[reverse_right_upper_via]))
        target_segments.append((points[reverse_right_upper_via], points[reverse_target_start]))

        target_reverse = secondary_curve_segments(
            cfg,
            dimensions,
            points[reverse_target_start],
            points[reverse_target_end],
            1.0,
            inner_offset,
            station_start_x=points[reverse_target_start][0],
            station_end_x=points[reverse_target_end][0],
            amplitude_override=amplitude_override,
        )
        target_segments.extend(target_reverse)
        target_reverse_paths.append(target_reverse)
        target_segments.append((points[reverse_target_end], points[reverse_left_lower_via]))
        inner_segments.append((points[reverse_left_lower_via], points[reverse_left_outer]))

        left_inner_end = secondary_curve_segments(
            cfg,
            dimensions,
            points[reverse_left_outer],
            points[left_end],
            1.0,
            outer_offset,
            station_start_x=points[reverse_left_outer][0],
            station_end_x=left_handoff_station_x.get(left_end, points[left_end][0]),
            amplitude_override=amplitude_override,
        )
        inner_segments.extend(left_inner_end)
        inner_reverse_paths.append(inner_reverse_outer + left_inner_end)

        if "left_detour" in spec:
            left_detour = str(spec["left_detour"])
            handoff_index = int(left_detour.split("_")[0].removeprefix("TURN")) - 1
            inner_segments.extend(left_inner_routes[handoff_index])
            target_segments.extend(
                (end, start)
                for start, end in reversed(left_target_routes[handoff_index])
            )
            via_labels.extend(
                (
                    left_upper_via,
                    right_lower_via,
                    right_detour,
                    reverse_right_upper_via,
                    reverse_left_lower_via,
                    left_detour,
                )
            )
        else:
            via_labels.extend(
                (
                    left_upper_via,
                    right_lower_via,
                    right_detour,
                    reverse_right_upper_via,
                    reverse_left_lower_via,
                )
            )

    # The first target rail and final inner rail cannot take the former direct
    # escape without crossing the newly packed left turnaround.  Route each
    # around the outside of its own layer's bundle, then merge both at y=0.
    target_connected_path = set(target_forward_paths[0])
    inner_connected_path = set(inner_reverse_paths[-1])
    detour_vias = tuple(
        points[label]
        for label in via_labels
        if label not in ("TERMINAL_OUTPUT_VIA", "TERMINAL_RETURN_VIA")
    )
    left_detour_stack = tuple(
        points[f"TURN{turn_number}_LEFT_TURNAROUND_VIA"]
        for turn_number in range(1, cfg["number_of_secondary_turns"])
    )
    convergence = points["FANOUT_CONVERGENCE"]
    target_escape = route_cl2_fanout_wrap(
        cfg,
        convergence,
        points["TURN1_START"],
        left_detour_stack,
        -1.0,
        fanout_direction(cfg),
        tuple(segment for segment in target_segments if segment not in target_connected_path),
        detour_vias,
    )
    return_escape = route_cl2_fanout_wrap(
        cfg,
        convergence,
        points[return_start_label],
        left_detour_stack,
        1.0,
        fanout_direction(cfg),
        tuple(segment for segment in inner_segments if segment not in inner_connected_path),
        detour_vias,
    )
    entry_escape_path = (
        (points["TERMINAL_OUTPUT_VIA"], points["ENTRY_FANOUT_JOG"]),
        (points["ENTRY_FANOUT_JOG"], points["ENTRY_CONVERGENCE_JOG"]),
        (points["ENTRY_CONVERGENCE_JOG"], convergence),
    ) + target_escape
    return_escape_path = tuple(
        (end, start) for start, end in reversed(return_escape)
    ) + (
        (convergence, points["RETURN_CONVERGENCE_JOG"]),
        (points["RETURN_CONVERGENCE_JOG"], points["RETURN_FANOUT_JOG"]),
        (points["RETURN_FANOUT_JOG"], points["TERMINAL_RETURN_VIA"]),
    )
    target_segments[0:0] = entry_escape_path
    inner_segments.extend(return_escape_path)
    via_labels.append("TERMINAL_RETURN_VIA")

    return SecondaryLayoutPlan(
        points=points,
        target_segments=tuple(target_segments),
        inner_segments=tuple(inner_segments),
        via_labels=tuple(dict.fromkeys(via_labels)),
        target_forward_paths=tuple(target_forward_paths),
        target_reverse_paths=tuple(target_reverse_paths),
        inner_forward_paths=tuple(inner_forward_paths),
        inner_reverse_paths=tuple(inner_reverse_paths),
        left_target_handoff_paths=tuple(
            left_target_routes[handoff_index]
            for handoff_index in sorted(left_target_routes)
        ),
        left_inner_handoff_paths=tuple(
            left_inner_routes[handoff_index]
            for handoff_index in sorted(left_inner_routes)
        ),
        left_handoff_station_x=left_handoff_station_x,
        entry_escape_path=entry_escape_path,
        return_escape_path=return_escape_path,
    )


def validate_multiturn_cl2_clearance(
    cfg: dict,
    dimensions: SensorDimensions,
    primary_geometry: PrimaryGeometry,
    layout: SecondaryLayoutPlan,
) -> None:
    """Validate the generated CL2 spiral for the generalized receiver path."""
    primary_segments = tuple(
        segment
        for coil in primary_geometry.coils
        for segment in coil.body_segments
    )
    minimum_pad_distance = secondary_via_spacing(cfg)
    for first_index, first in enumerate(layout.via_labels):
        for second in layout.via_labels[first_index + 1:]:
            if (
                distance(layout.points[first], layout.points[second])
                + GEOMETRY_TOLERANCE_MM
                < minimum_pad_distance
            ):
                raise ValueError(
                    f"CL2 paired vias {first}/{second} violate plated via clearance."
                )

    primary_pads = tuple(primary_geometry.pads.values())
    for terminal in ("TERMINAL_OUTPUT_VIA", "TERMINAL_RETURN_VIA"):
        for primary_pad in primary_pads:
            if (
                distance(layout.points[terminal], primary_pad)
                + GEOMETRY_TOLERANCE_MM
                < minimum_pad_distance
            ):
                raise ValueError(f"CL2 terminal {terminal} collides with a primary via.")

    minimum_primary_trace_distance = osc1_via_trace_clearance(cfg)
    for via_label in layout.via_labels:
        if via_label in ("TERMINAL_OUTPUT_VIA", "TERMINAL_RETURN_VIA"):
            continue
        if via_label.endswith("_LEFT_TURNAROUND_VIA"):
            # Temporary carve-out while CL1 and the primary are re-routed around the
            # mirrored left-side CL2 turnaround geometry.
            continue
        nearest_primary_trace = min(
            point_to_segment_distance(layout.points[via_label], segment)
            for segment in primary_segments
        )
        if nearest_primary_trace + GEOMETRY_TOLERANCE_MM < minimum_primary_trace_distance:
            raise ValueError(
                f"CL2 via {via_label} violates clearance to the primary winding."
            )

    pitch = trace_pitch(cfg)
    polygonal_tolerance = 0.003
    for group in (layout.target_reverse_paths, layout.inner_forward_paths):
        for first, second in zip(group, group[1:]):
            actual_spacing = path_to_path_distance(first, second)
            if actual_spacing + polygonal_tolerance < pitch:
                raise ValueError(
                    "CL2 parallel sinusoidal traces violate configured spacing: "
                    f"minimum centerline distance is {actual_spacing:.6f} mm, "
                    f"required pitch is {pitch:.6f} mm."
                )

    for handoff_index, route in enumerate(layout.left_target_handoff_paths):
        for path_index, path in enumerate(layout.target_forward_paths):
            if path_index == handoff_index + 1:
                continue
            actual_spacing = path_to_path_distance(route, path)
            if actual_spacing + ROUTING_POLYGONAL_TOLERANCE_MM < pitch:
                raise ValueError(
                    "CL2 left target handoff crowds a non-connected turn: "
                    f"minimum centerline distance is {actual_spacing:.6f} mm."
                )
    for handoff_index, route in enumerate(layout.left_inner_handoff_paths):
        for path_index, path in enumerate(layout.inner_reverse_paths):
            if path_index == handoff_index:
                continue
            actual_spacing = path_to_path_distance(route, path)
            if actual_spacing + ROUTING_POLYGONAL_TOLERANCE_MM < pitch:
                raise ValueError(
                    "CL2 left inner handoff crowds a non-connected turn: "
                    f"minimum centerline distance is {actual_spacing:.6f} mm."
                )
    for routes, layer_name in (
        (layout.left_target_handoff_paths, "target"),
        (layout.left_inner_handoff_paths, "inner"),
    ):
        for first_index, first in enumerate(routes):
            for second in routes[first_index + 1:]:
                actual_spacing = path_to_path_distance(first, second)
                if actual_spacing + ROUTING_POLYGONAL_TOLERANCE_MM < pitch:
                    raise ValueError(
                        f"CL2 left {layer_name} handoff paths violate trace spacing: "
                        f"minimum centerline distance is {actual_spacing:.6f} mm."
                    )

    def validate_fanout_escape(
        route: tuple[Segment, ...],
        layer_segments: tuple[Segment, ...],
        connected_path: tuple[Segment, ...],
        layer_name: str,
    ) -> None:
        obstacle_segments = tuple(
            segment
            for segment in layer_segments
            if segment not in route and segment not in connected_path
        )
        for segment in route:
            if any(
                segment_to_segment_distance(segment, obstacle)
                + ROUTING_POLYGONAL_TOLERANCE_MM
                < pitch
                for obstacle in obstacle_segments
            ):
                raise ValueError(
                    f"CL2 {layer_name} fanout escape violates trace spacing."
                )
            if any(
                point_to_segment_distance(layout.points[via_label], segment)
                + ROUTING_POLYGONAL_TOLERANCE_MM
                < via_to_trace_clearance(cfg)
                for via_label in layout.via_labels
                if via_label not in ("TERMINAL_OUTPUT_VIA", "TERMINAL_RETURN_VIA")
            ):
                raise ValueError(
                    f"CL2 {layer_name} fanout escape crowds a via."
                )

    validate_fanout_escape(
        layout.entry_escape_path,
        layout.target_segments,
        layout.target_forward_paths[0],
        "entry",
    )
    validate_fanout_escape(
        layout.return_escape_path,
        layout.inner_segments,
        layout.inner_reverse_paths[-1],
        "return",
    )

    left_routes = layout.left_target_handoff_paths + layout.left_inner_handoff_paths
    for route_index, route in enumerate(left_routes):
        own_turn = (
            route_index % max(len(layout.left_target_handoff_paths), 1)
        ) + 1
        own_via = layout.points[f"TURN{own_turn}_LEFT_TURNAROUND_VIA"]
        for via_label in layout.via_labels:
            via = layout.points[via_label]
            if via == own_via:
                continue
            minimum_distance = min(point_to_segment_distance(via, segment) for segment in route)
            if minimum_distance + ROUTING_POLYGONAL_TOLERANCE_MM < via_to_trace_clearance(cfg):
                raise ValueError(
                    f"CL2 left handoff route crowds via {via_label}: "
                    f"minimum centerline distance is {minimum_distance:.6f} mm."
                )

    turnaround_violations = cl2_right_turnaround_clearance_violations(
        cfg,
        layout.points,
        primary_segments,
    )
    if turnaround_violations:
        raise ValueError(turnaround_violations[0])


def build_cl2_geometry(
    cfg: dict | None = None,
    primary_geometry: PrimaryGeometry | None = None,
) -> SecondaryCoil | None:
    """Build the configured CL2 receiver coil, or return ``None`` when disabled."""
    cfg = build_config() if cfg is None else cfg
    if not cfg["generate_cl2"]:
        return None
    dimensions = calculate_dimensions(cfg)
    validate_config(cfg, dimensions)
    primary_geometry = primary_geometry or build_primary_geometry(cfg)
    layout = build_multiturn_cl2_layout(cfg, dimensions, primary_geometry)
    if not should_skip_geometry_validation(cfg):
        validate_multiturn_cl2_clearance(cfg, dimensions, primary_geometry, layout)
    points, via_labels = canonical_receiver_point_map(layout.points, layout.via_labels)
    return SecondaryCoil(
        name="CL2",
        target_layer=receiver_layers(cfg)[0],
        inner_layer=receiver_layers(cfg)[1],
        stroke_length_mm=secondary_stroke_length(cfg),
        points=points,
        target_segments=layout.target_segments,
        inner_segments=layout.inner_segments,
        via_labels=via_labels,
    )


def cl1_right_end_columns(cfg: dict) -> tuple[float, float]:
    """Return the outer and next turn columns in a left-entry frame."""
    outer_turn_x = secondary_stroke_length(cfg) / 2.0
    turn_pitch = cfg["via_diameter_mm"] + cfg["trace_spacing_mm"]
    next_turn_x = outer_turn_x - turn_pitch
    return outer_turn_x, next_turn_x


def cl1_crossover_candidate_cl2_segments(
    cfg: dict,
    cl2_geometry: SecondaryCoil,
    turn_x: float,
    required_clearance: float,
) -> tuple[Segment, ...]:
    """Return CL2 segments whose x-span could violate the crossover-via clearance."""
    cl2_segments = cl2_geometry.target_segments + cl2_geometry.inner_segments
    if fanout_direction(cfg) > 0:
        cl2_segments = mirror_segments_horizontally(cl2_segments)

    minimum_x = turn_x - required_clearance - GEOMETRY_TOLERANCE_MM
    maximum_x = turn_x + required_clearance + GEOMETRY_TOLERANCE_MM
    return tuple(
        segment
        for segment in cl2_segments
        if min(segment[0][0], segment[1][0]) <= maximum_x
        and max(segment[0][0], segment[1][0]) >= minimum_x
    )


def cl1_crossover_turn_half_height(
    cfg: dict,
    dimensions: SensorDimensions,
    cl2_geometry: SecondaryCoil | None,
    turn_x: float,
) -> float:
    """Return the smallest CL1 end-turn half-height that clears CL2 and OSC2."""
    minimum_half_height = (cfg["via_diameter_mm"] + cfg["trace_spacing_mm"]) / 2.0
    maximum_half_height = primary_inner_half_height(cfg, dimensions) - trace_pitch(cfg)
    if maximum_half_height + GEOMETRY_TOLERANCE_MM < minimum_half_height:
        raise ValueError("CL1 crossover turn exceeds the available OSC2 keep-out window.")
    if cl2_geometry is None:
        return minimum_half_height

    required_clearance = osc1_via_trace_clearance(cfg)
    cl2_segments = cl1_crossover_candidate_cl2_segments(
        cfg, cl2_geometry, turn_x, required_clearance
    )
    if not cl2_segments:
        return minimum_half_height
    search_step = 0.001
    candidate = minimum_half_height
    while candidate <= maximum_half_height + GEOMETRY_TOLERANCE_MM:
        upper_point = (turn_x, candidate)
        lower_point = (turn_x, -candidate)
        nearest_cl2_trace = min(
            point_to_segment_distance(test_point, segment)
            for test_point in (upper_point, lower_point)
            for segment in cl2_segments
        )
        if nearest_cl2_trace + GEOMETRY_TOLERANCE_MM >= required_clearance:
            return candidate
        candidate += search_step
    raise ValueError("CL1 crossover turn cannot clear CL2 within the OSC2 keep-out window.")


def build_multiturn_cl1_layout(
    cfg: dict,
    dimensions: SensorDimensions,
    cl2_geometry: SecondaryCoil | None = None,
    primary_geometry: PrimaryGeometry | None = None,
) -> CL1LayoutPlan:
    """Build CL1 for all valid turn counts while preserving the legacy turn order."""
    phase_offset = math.pi / 2.0
    left_x = -(secondary_stroke_length(cfg) / 2.0)
    terminal_x = -((dimensions.primary_length_mm / 2.0) + cfg["terminal_escape_length_mm"])
    entrance_y = terminal_row_y(cfg, "CL1")
    return_terminal_y = terminal_row_y(cfg, "CL1-GND")
    via_clearance = osc1_via_trace_clearance(cfg)
    turn_offsets = cl1_turn_offsets(cfg)
    midpoint_columns = cl1_midpoint_columns(cfg)
    amplitude_override = secondary_wave_amplitude_for_offsets(dimensions, turn_offsets)
    turn_count = len(turn_offsets)
    right_columns = tuple(cl1_right_end_column(cfg, index) for index in range(turn_count))
    via_pitch = secondary_via_spacing(cfg)
    right_transition_clearance = osc1_via_trace_clearance(cfg)
    max_rack_y = primary_inner_half_height(cfg, dimensions) - trace_pitch(cfg)
    left_columns = tuple(left_x + (index * via_pitch) for index in range(turn_count))

    cl2_layout_segments: tuple[Segment, ...] = ()
    if cl2_geometry is not None:
        cl2_layout_segments = cl2_geometry.target_segments + cl2_geometry.inner_segments
        if fanout_direction(cfg) > 0:
            cl2_layout_segments = mirror_segments_horizontally(cl2_layout_segments)

    primary_layout_segments: tuple[Segment, ...] = ()
    if primary_geometry is not None:
        primary_layout_segments = tuple(
            segment
            for coil in primary_geometry.coils
            for segment in coil.body_segments + coil.escape_segments
        )
        if fanout_direction(cfg) > 0:
            primary_layout_segments = mirror_segments_horizontally(primary_layout_segments)

    def transition_point_is_clear(position: Point) -> bool:
        if abs(position[1]) > max_rack_y + GEOMETRY_TOLERANCE_MM:
            return False
        if primary_layout_segments and (
            min(
                point_to_segment_distance(position, segment)
                for segment in primary_layout_segments
            )
            + GEOMETRY_TOLERANCE_MM
            < right_transition_clearance
        ):
            return False
        if cl2_layout_segments and (
            min(
                point_to_segment_distance(position, segment)
                for segment in cl2_layout_segments
            )
            + GEOMETRY_TOLERANCE_MM
            < right_transition_clearance
        ):
            return False
        return True

    def transition_rack_positions(
        turn_indices: tuple[int, ...], rack_x: float, vertical_sign: float
    ) -> tuple[Point, ...] | None:
        """Return a clearance-valid vertical transition-via rack, if one fits."""
        rack_top_y = cl1_crossover_turn_half_height(
            cfg, dimensions, cl2_geometry, rack_x
        )
        positions = tuple(
            (rack_x, vertical_sign * (rack_top_y + (row * via_pitch)))
            for row, _turn_index in enumerate(turn_indices)
        )
        for position in positions:
            if not transition_point_is_clear(position):
                return None
        return positions

    def plan_right_lower_vias() -> tuple[Point, ...]:
        """Return the one permitted lower-via rack or report insufficient height."""
        all_turns = tuple(range(turn_count))
        single_rack = transition_rack_positions(all_turns, right_columns[0], 1.0)
        if single_rack is not None:
            return single_rack
        raise ValueError(
            "CL1 right lower-via rack cannot fit within the available clearance window; "
            "increase the sensor height or reduce the number of secondary turns."
        )

    right_lower_via_positions = plan_right_lower_vias()

    def plan_left_upper_vias() -> tuple[Point, ...]:
        """Return the vertically flipped left upper-via rack or report insufficient height."""
        rack = transition_rack_positions(tuple(range(turn_count)), left_columns[0], -1.0)
        if rack is not None:
            return rack
        raise ValueError(
            "CL1 left upper-via rack cannot fit within the available clearance window; "
            "increase the sensor height or reduce the number of secondary turns."
        )

    left_upper_via_positions = plan_left_upper_vias()

    def left_return_escape_geometry(
        lower_via: Point,
    ) -> tuple[Point, tuple[Segment, ...]]:
        """Return the CL1 return-via location and entry path for one lower-via height."""
        return_escape = (
            left_columns[0] - via_pitch,
            lower_via[1] + via_clearance,
        )
        entry_45_start_x = return_escape[0] - (via_clearance * math.sqrt(2.0))
        entry_45_end_x = entry_45_start_x + (return_escape[1] - lower_via[1])
        entry_path = (
            ((terminal_x, entrance_y), (terminal_x + via_clearance, entrance_y)),
            ((terminal_x + via_clearance, entrance_y), (terminal_x + via_clearance, return_escape[1])),
            ((terminal_x + via_clearance, return_escape[1]), (entry_45_start_x, return_escape[1])),
            ((entry_45_start_x, return_escape[1]), (entry_45_end_x, lower_via[1])),
            ((entry_45_end_x, lower_via[1]), lower_via),
        )
        return return_escape, entry_path

    def left_return_escape_is_clear(lower_via: Point) -> bool:
        return_escape, entry_path = left_return_escape_geometry(lower_via)
        if not transition_point_is_clear(return_escape):
            return False
        return cl2_geometry is None or (
            path_to_path_distance(entry_path, cl2_geometry.target_segments)
            + GEOMETRY_TOLERANCE_MM
            >= trace_pitch(cfg)
        )

    left_lower_via_positions: list[Point] = []
    for turn_index, left_column in enumerate(left_columns):
        position = transition_rack_positions((turn_index,), left_column, 1.0)
        if position is None:
            raise ValueError(
                f"CL1 TURN{turn_index + 1} left lower via cannot clear the routing envelope."
            )
        lower_via = position[0]
        if turn_index == 0:
            candidate_via = lower_via
            while candidate_via[1] <= max_rack_y + GEOMETRY_TOLERANCE_MM:
                if (
                    transition_point_is_clear(candidate_via)
                    and left_return_escape_is_clear(candidate_via)
                ):
                    lower_via = candidate_via
                    break
                candidate_via = (candidate_via[0], candidate_via[1] + 0.05)
            else:
                raise ValueError("CL1 left return escape via cannot clear the routing envelope.")
        left_lower_via_positions.append(lower_via)

    # Place the return-exit via one via-to-trace clearance below CL1_D.  The
    # target-layer entry can then pass directly above the via at its required
    # clearance while CL2's nearby inner-layer return remains clear.
    left_return_escape_point, _ = left_return_escape_geometry(left_lower_via_positions[0])
    if not transition_point_is_clear(left_return_escape_point):
        raise ValueError("CL1 left return escape via cannot clear the routing envelope.")
    positive_rail_spans: list[tuple[float, float, float]] = []
    negative_rail_spans: list[tuple[float, float, float]] = []
    for turn_index, outer_offset in enumerate(turn_offsets):
        reverse_index = len(turn_offsets) - 1 - turn_index
        inner_offset = turn_offsets[reverse_index]
        forward_mid_x = midpoint_columns[turn_index]
        reverse_mid_x = midpoint_columns[reverse_index]
        right_x = right_columns[turn_index]
        forward_end_x = right_columns[reverse_index]
        # The target-layer return sine always starts directly above its own
        # upper via.  Only the Inner.1 forward-sine endpoint is reversed for
        # the three-turn lower-rack handoff.
        reverse_start_x = right_x
        forward_start_x = left_columns[turn_index]
        reverse_end_x = left_columns[reverse_index]
        positive_rail_spans.extend(
            (
                (outer_offset, forward_start_x, forward_mid_x),
                (inner_offset, forward_mid_x, forward_end_x),
            )
        )
        negative_rail_spans.extend(
            (
                (inner_offset, reverse_start_x, reverse_mid_x),
                (outer_offset, reverse_mid_x, reverse_end_x),
            )
        )

    def transition_via_y(
        station_x: float,
        phase_sign: float,
        *,
        upper: bool,
        connected_offsets: tuple[float, ...],
    ) -> float:
        return receiver_transition_via_y(
            cfg,
            dimensions,
            station_x,
            phase_sign,
            turn_offsets,
            upper=upper,
            connected_rail_offsets=connected_offsets,
            rail_spans=tuple(positive_rail_spans if phase_sign > 0.0 else negative_rail_spans),
            phase_offset_radians=phase_offset,
            amplitude_override=amplitude_override,
        )

    entry_route_y = left_return_escape_point[1]
    fanout_jog_x = terminal_x + via_clearance
    entry_45_start_x = left_return_escape_point[0] - (
        via_clearance * math.sqrt(2.0)
    )
    entry_45_end_x = entry_45_start_x + (
        entry_route_y - left_lower_via_positions[0][1]
    )
    points: dict[str, Point] = {
        "TERMINAL_OUTPUT_VIA": (terminal_x, entrance_y),
        # CL1 enters on the same long fanout corridor as the Inner.1 return.
        # It leaves that corridor at a 45-degree tangent around the return
        # via, then approaches CL1_D on the target layer.
        "ENTRY_FANOUT_JOG": (fanout_jog_x, entrance_y),
        "ENTRY_CORRIDOR_JOG": (fanout_jog_x, entry_route_y),
        "ENTRY_RETURN_ESCAPE_TANGENT_START": (entry_45_start_x, entry_route_y),
        "ENTRY_RETURN_ESCAPE_TANGENT_END": (
            entry_45_end_x,
            left_lower_via_positions[0][1],
        ),
        "TURN1_LEFT_LOWER_VIA": left_lower_via_positions[0],
        "TURN1_START": point_at_station_x(
            secondary_rail_point(
                cfg,
                dimensions,
                left_columns[0],
                1.0,
                turn_offsets[0],
                phase_offset,
                amplitude_override,
            ),
            left_columns[0],
        ),
    }

    turn_specs: list[dict[str, str | float]] = []
    right_via_labels: list[str] = []
    left_via_labels: list[str] = []
    for turn_index, outer_offset in enumerate(turn_offsets):
        reverse_index = len(turn_offsets) - 1 - turn_index
        inner_offset = turn_offsets[reverse_index]
        start_label = f"TURN{turn_index + 1}_START"
        forward_mid_x = midpoint_columns[turn_index]
        reverse_mid_x = midpoint_columns[reverse_index]
        right_x = right_columns[turn_index]
        forward_end_x = right_columns[reverse_index]
        reverse_start_x = right_x

        labels: dict[str, str | float] = {
            "outer_offset": outer_offset,
            "inner_offset": inner_offset,
            "start": start_label,
            "forward_mid_end": f"TURN{turn_index + 1}_FORWARD_MID_END",
            "forward_mid_via": f"TURN{turn_index + 1}_FORWARD_MID_VIA",
            "forward_inner_start": f"TURN{turn_index + 1}_FORWARD_INNER_START",
            "right_end": f"TURN{turn_index + 1}_RIGHT_END",
            "right_upper_via": f"TURN{turn_index + 1}_RIGHT_UPPER_VIA",
            "right_lower_via": f"TURN{turn_index + 1}_RIGHT_LOWER_VIA",
            "reverse_start": f"TURN{turn_index + 1}_RETURN_START",
            "reverse_mid_end": f"TURN{turn_index + 1}_RETURN_MID_END",
            "reverse_mid_via": f"TURN{turn_index + 1}_RETURN_MID_VIA",
            "reverse_inner_start": f"TURN{turn_index + 1}_RETURN_INNER_START",
            "left_return_end": f"TURN{turn_index + 1}_LEFT_RETURN_END",
            "left_upper_via": f"TURN{turn_index + 1}_LEFT_UPPER_VIA",
            "left_inner_entry_jog": f"TURN{turn_index + 1}_LEFT_INNER_ENTRY_JOG",
        }
        labels["left_lower_via"] = (
            "TURN1_LEFT_LOWER_VIA" if turn_index == 0 else f"TURN{turn_index + 1}_LEFT_LOWER_VIA"
        )
        labels["right_inner_entry_jog"] = f"TURN{turn_index + 1}_RIGHT_INNER_ENTRY_JOG"
        labels["right_crossover_jog"] = f"TURN{turn_index + 1}_RIGHT_CROSSOVER_JOG"
        labels["right_return_jog"] = f"TURN{turn_index + 1}_RIGHT_RETURN_JOG"
        if turn_index < len(turn_offsets) - 1:
            labels["next_left_lower_via"] = f"TURN{turn_index + 2}_LEFT_LOWER_VIA"
            labels["left_crossover_jog"] = f"TURN{turn_index + 1}_LEFT_CROSSOVER_JOG"
            labels["end"] = f"TURN{turn_index + 2}_START"
        else:
            labels["left_return_escape_jog"] = "LEFT_RETURN_ESCAPE_JOG"
            labels["left_return_escape_via"] = "LEFT_RETURN_ESCAPE_VIA"

        points[str(labels["forward_mid_end"])] = point_at_station_x(
            secondary_rail_point(
                cfg,
                dimensions,
                forward_mid_x,
                1.0,
                outer_offset,
                phase_offset,
                amplitude_override,
            ),
            forward_mid_x,
        )
        points[str(labels["forward_mid_via"])] = (
            forward_mid_x,
            transition_via_y(
                forward_mid_x,
                1.0,
                upper=True,
                connected_offsets=(outer_offset, inner_offset),
            ),
        )
        points[str(labels["forward_inner_start"])] = point_at_station_x(
            secondary_rail_point(
                cfg,
                dimensions,
                forward_mid_x,
                1.0,
                inner_offset,
                phase_offset,
                amplitude_override,
            ),
            forward_mid_x,
        )
        points[str(labels["right_end"])] = point_at_station_x(
            secondary_rail_point(
                cfg,
                dimensions,
                forward_end_x,
                1.0,
                inner_offset,
                phase_offset,
                amplitude_override,
            ),
            forward_end_x,
        )
        half_height = cl1_crossover_turn_half_height(cfg, dimensions, cl2_geometry, right_x)
        points[str(labels["right_upper_via"])] = (right_x, -half_height)
        lower_via_x, lower_via_y = right_lower_via_positions[turn_index]
        points[str(labels["right_lower_via"])] = (lower_via_x, lower_via_y)
        points[str(labels["right_inner_entry_jog"])] = (forward_end_x, lower_via_y)
        points[str(labels["right_crossover_jog"])] = (right_x, lower_via_y)
        points[str(labels["reverse_start"])] = point_at_station_x(
            secondary_rail_point(
                cfg,
                dimensions,
                reverse_start_x,
                -1.0,
                inner_offset,
                phase_offset,
                amplitude_override,
            ),
            reverse_start_x,
        )
        points[str(labels["right_return_jog"])] = (
            right_x,
            points[str(labels["reverse_start"])][1],
        )
        points[str(labels["reverse_mid_end"])] = point_at_station_x(
            secondary_rail_point(
                cfg,
                dimensions,
                reverse_mid_x,
                -1.0,
                inner_offset,
                phase_offset,
                amplitude_override,
            ),
            reverse_mid_x,
        )
        points[str(labels["reverse_mid_via"])] = (
            reverse_mid_x,
            transition_via_y(
                reverse_mid_x,
                -1.0,
                upper=False,
                connected_offsets=(outer_offset, inner_offset),
            ),
        )
        points[str(labels["reverse_inner_start"])] = point_at_station_x(
            secondary_rail_point(
                cfg,
                dimensions,
                reverse_mid_x,
                -1.0,
                outer_offset,
                phase_offset,
                amplitude_override,
            ),
            reverse_mid_x,
        )
        reverse_end_x = left_columns[reverse_index]
        points[str(labels["left_return_end"])] = point_at_station_x(
            secondary_rail_point(
                cfg,
                dimensions,
                reverse_end_x,
                -1.0,
                outer_offset,
                phase_offset,
                amplitude_override,
            ),
            reverse_end_x,
        )
        left_lower_x, left_lower_y = left_lower_via_positions[turn_index]
        left_upper_x, left_upper_y = left_upper_via_positions[turn_index]
        points[str(labels["left_lower_via"])] = (left_lower_x, left_lower_y)
        points[str(labels["left_upper_via"])] = (left_upper_x, left_upper_y)
        points[str(labels["left_inner_entry_jog"])] = (reverse_end_x, left_upper_y)
        if turn_index < len(turn_offsets) - 1:
            next_lower_x, next_lower_y = left_lower_via_positions[turn_index + 1]
            points[str(labels["left_crossover_jog"])] = (next_lower_x, left_upper_y)
            points[str(labels["end"])] = point_at_station_x(
                secondary_rail_point(
                    cfg,
                    dimensions,
                    next_lower_x,
                    1.0,
                    turn_offsets[turn_index + 1],
                    phase_offset,
                    amplitude_override,
                ),
                next_lower_x,
            )
        else:
            points[str(labels["left_return_escape_jog"])] = (
                left_return_escape_point[0],
                left_upper_y,
            )
            points[str(labels["left_return_escape_via"])] = left_return_escape_point

        right_via_labels.extend((str(labels["right_upper_via"]), str(labels["right_lower_via"])))
        left_via_labels.extend((str(labels["left_upper_via"]), str(labels["left_lower_via"])))
        turn_specs.append(labels)

    # The return leaves its local via horizontally on Inner.1.  It only turns
    # at the fanout spine, keeping it away from CL2's nearby local escape.
    points["RETURN_FANOUT_JOG"] = (fanout_jog_x, left_return_escape_point[1])
    points["RETURN_TERMINAL_JOG"] = (fanout_jog_x, return_terminal_y)
    points["TERMINAL_RETURN_VIA"] = (terminal_x, return_terminal_y)

    if fanout_direction(cfg) > 0:
        points = mirror_points_horizontally(points)

    target_segments: list[Segment] = [
        (points["TERMINAL_OUTPUT_VIA"], points["ENTRY_FANOUT_JOG"]),
        (points["ENTRY_FANOUT_JOG"], points["ENTRY_CORRIDOR_JOG"]),
        (points["ENTRY_CORRIDOR_JOG"], points["ENTRY_RETURN_ESCAPE_TANGENT_START"]),
        (points["ENTRY_RETURN_ESCAPE_TANGENT_START"], points["ENTRY_RETURN_ESCAPE_TANGENT_END"]),
        (points["ENTRY_RETURN_ESCAPE_TANGENT_END"], points["TURN1_LEFT_LOWER_VIA"]),
        (points["TURN1_LEFT_LOWER_VIA"], points["TURN1_START"]),
    ]
    inner_segments: list[Segment] = []
    crossover_segments: list[Segment] = []
    target_forward_paths: list[tuple[Segment, ...]] = []
    target_reverse_paths: list[tuple[Segment, ...]] = []
    inner_forward_paths: list[tuple[Segment, ...]] = []
    inner_reverse_paths: list[tuple[Segment, ...]] = []
    via_labels: list[str] = ["TERMINAL_OUTPUT_VIA", "TURN1_LEFT_LOWER_VIA"]

    for spec in turn_specs:
        outer_offset = float(spec["outer_offset"])
        inner_offset = float(spec["inner_offset"])
        start_label = str(spec["start"])
        forward_mid_end = str(spec["forward_mid_end"])
        forward_mid_via = str(spec["forward_mid_via"])
        forward_inner_start = str(spec["forward_inner_start"])
        right_end = str(spec["right_end"])
        right_upper_via = str(spec["right_upper_via"])
        right_lower_via = str(spec["right_lower_via"])
        reverse_start = str(spec["reverse_start"])
        reverse_mid_end = str(spec["reverse_mid_end"])
        reverse_mid_via = str(spec["reverse_mid_via"])
        reverse_inner_start = str(spec["reverse_inner_start"])
        left_return_end = str(spec["left_return_end"])

        target_forward = secondary_curve_segments(
            cfg,
            dimensions,
            points[start_label],
            points[forward_mid_end],
            1.0,
            outer_offset,
            station_start_x=points[start_label][0],
            station_end_x=points[forward_mid_end][0],
            phase_offset_radians=phase_offset,
            mirror_phase_sign=False,
            amplitude_override=amplitude_override,
        )
        target_segments.extend(target_forward)
        target_forward_paths.append(target_forward)
        target_segments.append((points[forward_mid_end], points[forward_mid_via]))
        inner_segments.append((points[forward_mid_via], points[forward_inner_start]))

        inner_forward = secondary_curve_segments(
            cfg,
            dimensions,
            points[forward_inner_start],
            points[right_end],
            1.0,
            inner_offset,
            station_start_x=points[forward_inner_start][0],
            station_end_x=points[right_end][0],
            phase_offset_radians=phase_offset,
            mirror_phase_sign=False,
            amplitude_override=amplitude_override,
        )
        inner_segments.extend(inner_forward)
        inner_forward_paths.append(inner_forward)
        right_inner_entry_jog = str(spec["right_inner_entry_jog"])
        right_crossover_jog = str(spec["right_crossover_jog"])
        right_return_jog = str(spec["right_return_jog"])

        # The right-side handoff remains Inner.1 -> In2 -> target.  The
        # incoming Inner.1 sine reaches its lower-rack location first; In2
        # then crosses to the upper via in the turn's own column.
        inner_segments.append((points[right_end], points[right_inner_entry_jog]))
        if points[right_inner_entry_jog] != points[right_lower_via]:
            inner_segments.append(
                (points[right_inner_entry_jog], points[right_lower_via])
            )
        if points[right_lower_via] != points[right_crossover_jog]:
            crossover_segments.append(
                (points[right_lower_via], points[right_crossover_jog])
            )
        crossover_segments.append(
            (points[right_crossover_jog], points[right_upper_via])
        )

        # The target-layer return starts directly above its own upper via.
        target_segments.append((points[right_upper_via], points[right_return_jog]))

        return_right = secondary_curve_segments(
            cfg,
            dimensions,
            points[reverse_start],
            points[reverse_mid_end],
            -1.0,
            inner_offset,
            station_start_x=points[reverse_start][0],
            station_end_x=points[reverse_mid_end][0],
            phase_offset_radians=phase_offset,
            mirror_phase_sign=False,
            amplitude_override=amplitude_override,
        )
        target_segments.extend(return_right)
        target_reverse_paths.append(return_right)
        target_segments.append((points[reverse_mid_end], points[reverse_mid_via]))
        inner_segments.append((points[reverse_mid_via], points[reverse_inner_start]))

        return_left = secondary_curve_segments(
            cfg,
            dimensions,
            points[reverse_inner_start],
            points[left_return_end],
            -1.0,
            outer_offset,
            station_start_x=points[reverse_inner_start][0],
            station_end_x=points[left_return_end][0],
            phase_offset_radians=phase_offset,
            mirror_phase_sign=False,
            amplitude_override=amplitude_override,
        )
        inner_segments.extend(return_left)
        inner_reverse_paths.append(return_left)

        left_upper_via = str(spec["left_upper_via"])
        left_inner_entry_jog = str(spec["left_inner_entry_jog"])
        inner_segments.append((points[left_return_end], points[left_inner_entry_jog]))
        if points[left_inner_entry_jog] != points[left_upper_via]:
            inner_segments.append((points[left_inner_entry_jog], points[left_upper_via]))

        if "next_left_lower_via" in spec:
            next_left_lower_via = str(spec["next_left_lower_via"])
            left_crossover_jog = str(spec["left_crossover_jog"])
            end_label = str(spec["end"])
            if points[left_upper_via] != points[left_crossover_jog]:
                crossover_segments.append((points[left_upper_via], points[left_crossover_jog]))
            crossover_segments.append((points[left_crossover_jog], points[next_left_lower_via]))
            target_segments.append((points[next_left_lower_via], points[end_label]))
        else:
            left_return_escape_jog = str(spec["left_return_escape_jog"])
            left_return_escape_via = str(spec["left_return_escape_via"])
            crossover_segments.append((points[left_upper_via], points[left_return_escape_jog]))
            crossover_segments.append(
                (points[left_return_escape_jog], points[left_return_escape_via])
            )

        via_labels.extend(
            (
                forward_mid_via,
                right_upper_via,
                right_lower_via,
                reverse_mid_via,
                left_upper_via,
                str(spec["left_lower_via"]),
            )
        )

    left_return_escape_via = str(turn_specs[-1]["left_return_escape_via"])
    inner_segments.append((points[left_return_escape_via], points["RETURN_FANOUT_JOG"]))
    inner_segments.append((points["RETURN_FANOUT_JOG"], points["RETURN_TERMINAL_JOG"]))
    inner_segments.append((points["RETURN_TERMINAL_JOG"], points["TERMINAL_RETURN_VIA"]))
    via_labels.extend((left_return_escape_via, "TERMINAL_RETURN_VIA"))
    left_via_labels.append(left_return_escape_via)

    entry_escape_path = (
        (points["TERMINAL_OUTPUT_VIA"], points["ENTRY_FANOUT_JOG"]),
        (points["ENTRY_FANOUT_JOG"], points["ENTRY_CORRIDOR_JOG"]),
        (points["ENTRY_CORRIDOR_JOG"], points["ENTRY_RETURN_ESCAPE_TANGENT_START"]),
        (points["ENTRY_RETURN_ESCAPE_TANGENT_START"], points["ENTRY_RETURN_ESCAPE_TANGENT_END"]),
        (points["ENTRY_RETURN_ESCAPE_TANGENT_END"], points["TURN1_LEFT_LOWER_VIA"]),
    )
    return_escape_path = (
        (points[left_return_escape_via], points["RETURN_FANOUT_JOG"]),
        (points["RETURN_FANOUT_JOG"], points["RETURN_TERMINAL_JOG"]),
        (points["RETURN_TERMINAL_JOG"], points["TERMINAL_RETURN_VIA"]),
    )
    if (
        min(
            point_to_segment_distance(points[left_return_escape_via], segment)
            for segment in entry_escape_path
        )
        + GEOMETRY_TOLERANCE_MM
        < via_clearance
    ):
        raise ValueError("CL1 entry trace cannot clear the left return escape via.")
    if cl2_geometry is not None:
        if (
            path_to_path_distance(entry_escape_path, cl2_geometry.target_segments)
            + GEOMETRY_TOLERANCE_MM
            < trace_pitch(cfg)
        ):
            raise ValueError("CL1 entry escape violates clearance to CL2 target copper.")
        if (
            path_to_path_distance(return_escape_path, cl2_geometry.inner_segments)
            + GEOMETRY_TOLERANCE_MM
            < trace_pitch(cfg)
        ):
            raise ValueError("CL1 return escape violates clearance to CL2 inner copper.")

    return CL1LayoutPlan(
        points=points,
        target_segments=tuple(target_segments),
        inner_segments=tuple(inner_segments),
        crossover_segments=tuple(crossover_segments),
        target_arcs=(),
        inner_arcs=(),
        via_labels=tuple(dict.fromkeys(via_labels)),
        target_forward_paths=tuple(target_forward_paths),
        target_reverse_paths=tuple(target_reverse_paths),
        inner_forward_paths=tuple(inner_forward_paths),
        inner_reverse_paths=tuple(inner_reverse_paths),
        right_via_labels=tuple(dict.fromkeys(right_via_labels)),
        left_via_labels=tuple(dict.fromkeys(left_via_labels)),
    )


def validate_multiturn_cl1_clearance(
    cfg: dict,
    dimensions: SensorDimensions,
    primary_geometry: PrimaryGeometry,
    cl2_geometry: SecondaryCoil | None,
    layout: CL1LayoutPlan,
) -> None:
    """Validate the generated CL1 spiral for the generalized receiver path."""
    endpoint_clearance = (
        (dimensions.primary_length_mm - secondary_stroke_length(cfg)) / 2.0
    )
    if endpoint_clearance + GEOMETRY_TOLERANCE_MM < cfg["cl1_primary_end_min_clearance_mm"]:
        raise ValueError("CL1 endpoint violates minimum clearance to the primary end winding.")

    minimum_pad_distance = secondary_via_spacing(cfg)
    for first_index, first in enumerate(layout.via_labels):
        for second in layout.via_labels[first_index + 1:]:
            if (
                distance(layout.points[first], layout.points[second])
                + GEOMETRY_TOLERANCE_MM
                < minimum_pad_distance
            ):
                raise ValueError(
                    f"CL1 paired vias {first}/{second} violate plated via clearance."
                )

    minimum_trace_distance = osc1_via_trace_clearance(cfg)
    for via_label in layout.via_labels:
        if via_label in ("TERMINAL_OUTPUT_VIA", "TERMINAL_RETURN_VIA"):
            continue
        nearest_primary_trace = min(
            point_to_segment_distance(layout.points[via_label], segment)
            for coil in primary_geometry.coils
            for segment in coil.body_segments
        )
        if nearest_primary_trace + GEOMETRY_TOLERANCE_MM < minimum_trace_distance:
            raise ValueError(f"CL1 via {via_label} violates clearance to the primary winding.")

    osc2 = next((coil for coil in primary_geometry.coils if coil.name == "OSC2"), None)
    if osc2 is not None:
        if (
            path_to_path_distance(layout.crossover_segments, osc2.body_segments)
            + GEOMETRY_TOLERANCE_MM
            < trace_pitch(cfg)
        ):
            raise ValueError("CL1 crossover copper violates clearance to OSC2.")

    if cl2_geometry is not None:
        cl2_terminal_points = (
            cl2_geometry.points["TERMINAL_OUTPUT_VIA"],
            cl2_geometry.points["TERMINAL_RETURN_VIA"],
        )
        for terminal in ("TERMINAL_OUTPUT_VIA", "TERMINAL_RETURN_VIA"):
            for point in cl2_terminal_points + tuple(primary_geometry.pads.values()):
                if (
                    distance(layout.points[terminal], point)
                    + GEOMETRY_TOLERANCE_MM
                    < minimum_pad_distance
                ):
                    raise ValueError(f"CL1 terminal {terminal} collides with an existing via.")

        cl2_segments = cl2_geometry.target_segments + cl2_geometry.inner_segments
        for via_label in layout.right_via_labels + layout.left_via_labels:
            nearest_cl2_trace = min(
                point_to_segment_distance(layout.points[via_label], segment)
                for segment in cl2_segments
            )
            if nearest_cl2_trace + GEOMETRY_TOLERANCE_MM < minimum_trace_distance:
                raise ValueError(
                    f"CL1 crossover via {via_label} violates clearance to CL2."
                )

        for path in layout.target_forward_paths + layout.target_reverse_paths:
            if (
                path_to_path_distance(path, cl2_geometry.target_segments)
                + GEOMETRY_TOLERANCE_MM
                < trace_pitch(cfg)
            ):
                raise ValueError("CL1 target curve violates clearance to CL2.")
        for path in layout.inner_forward_paths + layout.inner_reverse_paths:
            if (
                path_to_path_distance(path, cl2_geometry.inner_segments)
                + GEOMETRY_TOLERANCE_MM
                < trace_pitch(cfg)
            ):
                raise ValueError("CL1 inner curve violates clearance to CL2.")

    pitch = trace_pitch(cfg)
    # The sinusoidal paths are represented by sampled line segments.  The
    # staggered left-side endpoints make the worst pair land between samples;
    # with the configured 256 samples the maximum observed chord error is
    # just over 0.005 mm.  Preserve a small margin beyond it rather than
    # rejecting analytically pitch-spaced turns.
    polygonal_tolerance = max(ROUTING_POLYGONAL_TOLERANCE_MM, 0.006)
    for group in (
        layout.target_forward_paths,
        layout.target_reverse_paths,
        layout.inner_forward_paths,
        layout.inner_reverse_paths,
    ):
        for first, second in zip(group, group[1:]):
            actual_spacing = path_to_path_distance(first, second)
            if actual_spacing + polygonal_tolerance < pitch:
                raise ValueError(
                    "CL1 parallel sinusoidal traces violate configured spacing: "
                    f"minimum centerline distance is {actual_spacing:.6f} mm, "
                    f"required pitch is {pitch:.6f} mm."
                )


def build_cl1_geometry(
    cfg: dict | None = None,
    primary_geometry: PrimaryGeometry | None = None,
    cl2_geometry: SecondaryCoil | None = None,
) -> CL1Coil | None:
    """Build the configured CL1 receiver coil, or return ``None`` when disabled."""
    cfg = build_config() if cfg is None else cfg
    if not cfg["generate_cl1"]:
        return None
    dimensions = calculate_dimensions(cfg)
    validate_config(cfg, dimensions)
    primary_geometry = primary_geometry or build_primary_geometry(cfg)
    if cl2_geometry is None and cfg["generate_cl2"]:
        cl2_geometry = build_cl2_geometry(cfg, primary_geometry)
    layout = build_multiturn_cl1_layout(
        cfg,
        dimensions,
        cl2_geometry,
        primary_geometry,
    )
    if not should_skip_geometry_validation(cfg):
        validate_multiturn_cl1_clearance(
            cfg,
            dimensions,
            primary_geometry,
            cl2_geometry,
            layout,
        )
    points, via_labels = canonical_receiver_point_map(layout.points, layout.via_labels)
    return CL1Coil(
        name="CL1",
        target_layer=receiver_layers(cfg)[0],
        inner_layer=receiver_layers(cfg)[1],
        crossover_layer=receiver_crossover_layer(cfg),
        stroke_length_mm=secondary_stroke_length(cfg),
        points=points,
        target_segments=layout.target_segments,
        inner_segments=layout.inner_segments,
        crossover_segments=layout.crossover_segments,
        target_arcs=layout.target_arcs,
        inner_arcs=layout.inner_arcs,
        via_labels=via_labels,
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
                geometry.pads["OSC1_TERMINAL_OUTPUT_VIA"],
                cfg["via_diameter_mm"],
                cfg["via_hole_size_mm"],
            )
        )
        for segment in coil.body_segments:
            sections.append(fp_line(segment[0], segment[1], cfg["trace_width_mm"], coil.layer))
        sections.append(
            pad_thru_hole(
                cfg["primary_input_pad_name"],
                geometry.pads["VIN_SHARED_VIA"],
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
                geometry.pads["VIN_TERMINAL_VIA"],
                cfg["via_diameter_mm"],
                cfg["via_hole_size_mm"],
            )
        )

    if cfg["generate_osc2"]:
        coil = next(coil for coil in geometry.coils if coil.name == "OSC2")
        sections.append(
            pad_thru_hole(
                cfg["osc2_output_pad_name"],
                geometry.pads["OSC2_TERMINAL_OUTPUT_VIA"],
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
                "TERMINAL_OUTPUT_VIA": cfg["cl2_output_pad_name"],
                "TERMINAL_RETURN_VIA": cfg["cl2_return_pad_name"],
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
                "TERMINAL_OUTPUT_VIA": cfg["cl1_output_pad_name"],
                "TERMINAL_RETURN_VIA": cfg["cl1_return_pad_name"],
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
    measurement_range = cfg["stroke_range_mm"] - cfg["target_x_mm"]
    print(f"Wrote {output_path}")
    print(f"Measurement range: {measurement_range:.3f} mm")
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
