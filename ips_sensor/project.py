"""Portable versioned project files for reproducible sensor configurations."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .models import GenerationRequest, LinearSensorConfig, OutputConfig

PROJECT_SCHEMA_VERSION = 4


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
    """Load current project documents and migrate supported legacy versions."""
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
    migrated = dict(raw)
    version = schema_version
    while version < PROJECT_SCHEMA_VERSION:
        if version == 0:
            # Early hand-authored files may have placed output_dir in config.
            config = dict(migrated.get("config", {}))
            output = dict(migrated.get("output", {}))
            if "output_dir" in config and "output_dir" not in output:
                output["output_dir"] = config.pop("output_dir")
            migrated = {
                **migrated,
                "schema_version": 1,
                "config": config,
                "output": output,
            }
        elif version == 1:
            # In schema v1 this persisted setting had no routing effect. Reset
            # it to Automatic so opening a legacy project preserves its layout.
            config = dict(migrated.get("config", {}))
            config["osc1_vin_exit_offset_mm"] = 0.0
            migrated = {**migrated, "schema_version": 2, "config": config}
        elif version == 2:
            # These were retired after the generalized receiver routes stopped
            # consuming them. Automatic primary extension now derives CL1's
            # actual routing requirement internally.
            config = dict(migrated.get("config", {}))
            for key in (
                "secondary_jump_runup_via_multiplier",
                "secondary_jump_detour_via_multiplier",
                "cl1_transition_column_fraction",
                "cl1_primary_end_min_clearance_mm",
            ):
                config.pop(key, None)
            migrated = {**migrated, "schema_version": 3, "config": config}
        elif version == 3:
            # Before v4 this value was a per-side margin and the primary
            # envelope added it twice. It is now one total extension.
            config = dict(migrated.get("config", {}))
            old_margin = config.get("primary_y_margin_mm", 0.075)
            if isinstance(old_margin, (int, float)) and not isinstance(old_margin, bool):
                config["primary_y_margin_mm"] = old_margin * 2.0
            migrated = {**migrated, "schema_version": 4, "config": config}
        version += 1
    return migrated
