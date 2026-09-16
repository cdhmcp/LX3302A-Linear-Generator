"""Portable versioned project files for reproducible sensor configurations."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .models import GenerationRequest, LinearSensorConfig, OutputConfig

PROJECT_SCHEMA_VERSION = 1


class ProjectLoadError(ValueError):
    """Raised when a project document cannot safely be interpreted."""


def save_project(path: str | Path, request: GenerationRequest) -> Path:
    """Write a human-readable, versioned project document."""
    project_path = Path(path)
    payload = {
        "schema_version": PROJECT_SCHEMA_VERSION,
        "sensor_type": request.sensor_type,
        "exporter_id": request.exporter_id,
        "config": asdict(request.config),
        "output": asdict(request.output),
    }
    project_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return project_path


def load_project(path: str | Path) -> GenerationRequest:
    """Load schema v1 documents and normalize older missing-version files."""
    project_path = Path(path)
    try:
        raw: Any = json.loads(project_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProjectLoadError(f"Could not read project file: {error}") from error
    if not isinstance(raw, dict):
        raise ProjectLoadError("Project file must contain a JSON object.")

    schema_version = raw.get("schema_version", 0)
    if not isinstance(schema_version, int) or schema_version < 0:
        raise ProjectLoadError("Project schema version must be a non-negative integer.")
    if schema_version > PROJECT_SCHEMA_VERSION:
        raise ProjectLoadError(
            f"Project schema version {schema_version} is newer than this application supports."
        )
    for key in ("config", "output"):
        if key in raw and not isinstance(raw[key], dict):
            raise ProjectLoadError(f"Project {key} value must be a JSON object.")
    migrated = _migrate(raw, schema_version)
    config = migrated.get("config", {})
    output = migrated.get("output", {})
    if not isinstance(config, dict) or not isinstance(output, dict):
        raise ProjectLoadError("Project config and output values must be JSON objects.")
    return GenerationRequest(
        config=LinearSensorConfig.from_mapping(config),
        output=OutputConfig.from_mapping(output),
        sensor_type=str(migrated.get("sensor_type", "linear-lx3302a")),
        exporter_id=str(migrated.get("exporter_id", "kicad9")),
    )


def _migrate(raw: dict[str, Any], schema_version: int) -> dict[str, Any]:
    """Migrate known legacy shapes; add future migrations here in sequence."""
    if schema_version == 0:
        # Early hand-authored files may have placed output_dir in config.
        config = dict(raw.get("config", {}))
        output = dict(raw.get("output", {}))
        if "output_dir" in config and "output_dir" not in output:
            output["output_dir"] = config.pop("output_dir")
        return {**raw, "schema_version": 1, "config": config, "output": output}
    return raw
