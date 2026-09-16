"""PCB-platform-neutral footprint primitives used for preview and export."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

Point = tuple[float, float]


@dataclass(frozen=True)
class CopperLine:
    layer: str
    start: Point
    end: Point
    width_mm: float


@dataclass(frozen=True)
class CopperArc:
    layer: str
    start: Point
    mid: Point
    end: Point
    width_mm: float


@dataclass(frozen=True)
class ThroughHolePad:
    name: str
    center: Point
    diameter_mm: float
    hole_diameter_mm: float


FootprintItem: TypeAlias = CopperLine | CopperArc | ThroughHolePad


@dataclass(frozen=True)
class FootprintLayout:
    """A rendered-footprint-independent description of a sensor layout."""

    name: str
    reference: str
    items: tuple[FootprintItem, ...]
    primary_length_mm: float
    primary_width_mm: float
    secondary_length_mm: float
    secondary_width_mm: float

    @property
    def layers(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                item.layer for item in self.items if isinstance(item, (CopperLine, CopperArc))
            )
        )

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Return ``min_x, min_y, max_x, max_y`` including copper widths."""
        points: list[Point] = []
        for item in self.items:
            if isinstance(item, CopperLine):
                radius = item.width_mm / 2.0
                points.extend(
                    (
                        (item.start[0] - radius, item.start[1] - radius),
                        (item.start[0] + radius, item.start[1] + radius),
                        (item.end[0] - radius, item.end[1] - radius),
                        (item.end[0] + radius, item.end[1] + radius),
                    )
                )
            elif isinstance(item, CopperArc):
                # The arc midpoint is sufficient for a conservative preview
                # fit; exact circular extrema are not needed for export.
                radius = item.width_mm / 2.0
                for point in (item.start, item.mid, item.end):
                    points.extend(
                        ((point[0] - radius, point[1] - radius),
                         (point[0] + radius, point[1] + radius))
                    )
            else:
                radius = item.diameter_mm / 2.0
                points.extend(
                    ((item.center[0] - radius, item.center[1] - radius),
                     (item.center[0] + radius, item.center[1] + radius))
                )
        if not points:
            return (0.0, 0.0, 0.0, 0.0)
        xs, ys = zip(*points)
        return min(xs), min(ys), max(xs), max(ys)
