"""Qt graphics preview for neutral footprint layouts."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QGraphicsScene, QGraphicsView

from ..layout import CopperArc, CopperLine, FootprintLayout, ThroughHolePad


LAYER_COLORS = {
    "F.Cu": "#dc2626",
    "B.Cu": "#2563eb",
    "In1.Cu": "#7c3aed",
    "In2.Cu": "#059669",
}


class FootprintPreview(QGraphicsView):
    """A lightweight, scalable 2D engineering preview of a footprint layout."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setBackgroundBrush(QColor("#0f172a"))
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self._layout: FootprintLayout | None = None
        self._visible_layers: set[str] = set()

    @property
    def layout(self) -> FootprintLayout | None:
        return self._layout

    def set_layout(self, layout: FootprintLayout | None) -> None:
        self._layout = layout
        if layout is not None:
            self._visible_layers = set(layout.layers)
        self._draw()

    def set_layer_visible(self, layer: str, visible: bool) -> None:
        if visible:
            self._visible_layers.add(layer)
        else:
            self._visible_layers.discard(layer)
        self._draw()

    def _draw(self) -> None:
        self._scene.clear()
        if self._layout is None:
            return
        for item in self._layout.items:
            if isinstance(item, (CopperLine, CopperArc)) and item.layer not in self._visible_layers:
                continue
            if isinstance(item, CopperLine):
                self._draw_line(item)
            elif isinstance(item, CopperArc):
                self._draw_arc(item)
            else:
                self._draw_pad(item)
        self._fit_to_layout()

    @staticmethod
    def _point(point: tuple[float, float]) -> tuple[float, float]:
        # KiCad's footprint viewer presents positive PCB Y downward.  Preserve
        # that convention so the in-app preview has the same orientation.
        return point

    def _pen(self, layer: str, width_mm: float) -> QPen:
        pen = QPen(QColor(LAYER_COLORS.get(layer, "#f59e0b")))
        pen.setWidthF(max(width_mm, 0.02))
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        return pen

    def _draw_line(self, item: CopperLine) -> None:
        start_x, start_y = self._point(item.start)
        end_x, end_y = self._point(item.end)
        self._scene.addLine(start_x, start_y, end_x, end_y, self._pen(item.layer, item.width_mm))

    def _draw_arc(self, item: CopperArc) -> None:
        # KiCad arcs are defined by three points. A quadratic curve is a
        # readable preview approximation; exporters retain exact coordinates.
        start_x, start_y = self._point(item.start)
        mid_x, mid_y = self._point(item.mid)
        end_x, end_y = self._point(item.end)
        path = QPainterPath()
        path.moveTo(start_x, start_y)
        path.quadTo(mid_x, mid_y, end_x, end_y)
        self._scene.addPath(path, self._pen(item.layer, item.width_mm))

    def _draw_pad(self, item: ThroughHolePad) -> None:
        center_x, center_y = self._point(item.center)
        radius = item.diameter_mm / 2.0
        outline = QPen(QColor("#e2e8f0"))
        outline.setWidthF(max(item.diameter_mm * 0.08, 0.03))
        pad_item = self._scene.addEllipse(
            center_x - radius,
            center_y - radius,
            item.diameter_mm,
            item.diameter_mm,
            outline,
            QColor("#475569"),
        )
        pad_item.setToolTip(item.name)

    def _fit_to_layout(self) -> None:
        rect = self._scene.itemsBoundingRect()
        if not rect.isNull():
            self.fitInView(rect.adjusted(-1.0, -1.0, 1.0, 1.0), Qt.AspectRatioMode.KeepAspectRatio)

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt callback name
        super().resizeEvent(event)
        self._fit_to_layout()
