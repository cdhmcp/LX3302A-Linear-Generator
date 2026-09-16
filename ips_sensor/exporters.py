"""Output adapters for neutral sensor-footprint layouts."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from .layout import CopperArc, CopperLine, FootprintLayout, ThroughHolePad
from .models import OutputConfig


class Exporter(Protocol):
    """An output target that can render and persist a footprint layout."""

    id: str
    display_name: str

    def render(self, layout: FootprintLayout) -> str: ...

    def write(self, layout: FootprintLayout, output: OutputConfig) -> Path: ...


class KiCad9Exporter:
    """KiCad 9 footprint writer for a platform-neutral layout."""

    id = "kicad9"
    display_name = "KiCad 9 footprint"

    def render(self, layout: FootprintLayout) -> str:
        sections = [
            f'''(footprint "{layout.name}"
  (version 20240201)
  (generator "linear_sensor_generator")
  (layer "F.Cu")
  (attr smd)
''',
            f'''  (fp_text reference "{layout.reference}" (at 0 0) (layer "F.SilkS") hide
    (effects (font (size 1 1) (thickness 0.15))))
  (fp_text value "{layout.name}" (at 0 0) (layer "F.Fab") hide
    (effects (font (size 1 1) (thickness 0.15))))
''',
        ]
        for item in layout.items:
            if isinstance(item, CopperLine):
                sections.append(
                    f'''  (fp_line (start {item.start[0]:.6f} {item.start[1]:.6f}) (end {item.end[0]:.6f} {item.end[1]:.6f})
    (stroke (width {item.width_mm:.6f}) (type solid)) (layer "{item.layer}"))
'''
                )
            elif isinstance(item, CopperArc):
                sections.append(
                    f'''  (fp_arc (start {item.start[0]:.6f} {item.start[1]:.6f}) (mid {item.mid[0]:.6f} {item.mid[1]:.6f}) (end {item.end[0]:.6f} {item.end[1]:.6f})
    (stroke (width {item.width_mm:.6f}) (type solid)) (layer "{item.layer}"))
'''
                )
            else:
                sections.append(
                    f'''  (pad "{item.name}" thru_hole circle (at {item.center[0]:.6f} {item.center[1]:.6f}) (size {item.diameter_mm:.6f} {item.diameter_mm:.6f}) (drill {item.hole_diameter_mm:.6f})
    (layers "*.Cu" "*.Mask"))
'''
                )
        sections.append(")\n")
        return "".join(sections)

    def write(self, layout: FootprintLayout, output: OutputConfig) -> Path:
        output_dir = Path(output.output_dir)
        if not output_dir.is_absolute():
            output_dir = Path.cwd() / output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{layout.name}.kicad_mod"
        output_path.write_text(self.render(layout), encoding="ascii")
        return output_path


EXPORTERS: dict[str, Exporter] = {KiCad9Exporter.id: KiCad9Exporter()}
