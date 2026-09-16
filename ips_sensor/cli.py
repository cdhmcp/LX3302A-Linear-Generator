"""Small command-line entry point for repeatable project-file generation."""

from __future__ import annotations

import argparse
from pathlib import Path

from .engine import GENERATORS, GenerationBlocked
from .models import default_request
from .project import load_project


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate an inductive-sensor footprint.")
    parser.add_argument("--project", type=Path, help="Versioned .ips-sensor.json project file")
    parser.add_argument("--force", action="store_true", help="Allow export despite geometry errors")
    args = parser.parse_args(argv)
    request = load_project(args.project) if args.project else default_request()
    generator = GENERATORS.get(request.sensor_type)
    if generator is None:
        print(f"Unsupported sensor type: {request.sensor_type}")
        return 2
    try:
        output_path = generator.generate(request, force=args.force)
    except GenerationBlocked as error:
        for item in error.report.diagnostics:
            print(f"{item.severity.value.upper()} {item.code}: {item.message}")
        return 2
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
