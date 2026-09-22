# Inductive Sensor Footprint Generator

This project generates KiCad 9 footprints for the LX3302A linear inductive position-sensor coils. It now includes a reusable typed core and a Windows-first PySide6 desktop GUI, so sensor settings can be changed without editing the generator source.

## Run the desktop GUI

Install the GUI optional dependency, then start the application:

```powershell
python -m pip install -e .[gui]
ips-sensor-gui
```

The application presents common settings first and exposes all routing and naming controls through **Show advanced**. Every field has a help button. Select **Validate & Update Preview** to run background geometry checks and refresh the preview/diagnostics; changing a setting marks those results as out of date until you validate again. Invalid but renderable geometry requires an explicit **Generate anyway** confirmation before KiCad output is written.

Set **Primary end extension** to `0` to have the generator select the smallest symmetric extension required by the active CL1 and CL2 routing. A positive value is accepted only when it is at least that computed clearance-safe value.

**Primary vertical extension** is the total extra height of the primary envelope: `primary width = secondary width + extension`. It defaults to `0 mm`, which keeps the two envelopes the same height.

## Generate from a saved project

The GUI saves portable, versioned `*.ips-sensor.json` files. They contain the selected sensor definition, exporter, all sensor values, and output folder.

```powershell
ips-sensor-generate --project my-sensor.ips-sensor.json
```

Use `--force` only after reviewing reported geometry errors:

```powershell
ips-sensor-generate --project my-sensor.ips-sensor.json --force
```

The original `linear_sensor_generator.py` remains executable and accepts its existing dictionary configuration, preserving script-based workflows.

## Windows package

On a Windows build machine with Python installed, run:

```powershell
.\scripts\build_windows.ps1
```

It creates an onedir bundle under `dist\IPS-Sensor-GUI`. Code signing and installer wrapping are intentionally release-process steps because no certificate or distribution channel is included in this repository.

## Internal Windows releases

Internal releases are versioned ZIP files made from an annotated Git tag. Normal development commits are not versioned; create a tag only for a build that will be distributed.

1. Update the `version` in `pyproject.toml`, commit the release candidate, and create an annotated tag such as `v0.1.0`.
2. From that clean, tagged commit, run:

   ```powershell
   .\scripts\release_windows.ps1 -Version 0.1.0
   ```

   Pass `-ReleaseNotesPath .\my-release-notes.md` to supply curated Markdown notes; otherwise the script creates notes from Git commit subjects.
3. Push the release commit and tag, then publish the generated `dist\releases\v0.1.0` folder to the internal release location:

   ```powershell
   git push origin gui
   git push origin v0.1.0
   ```

   The folder contains the versioned ZIP, `SHA256SUMS.txt`, `INSTALL.txt`, and release notes.

Users should download the ZIP, extract the whole folder locally, and run `IPS-Sensor-GUI.exe`. Do not run it directly from a network share or from inside the ZIP. Keep published release folders intact for rollback; the script refuses to overwrite an existing release version.

Use `v0.1.1` for a bug-fix release and `v0.2.0` for a new internal feature. Never move a published tag; create a new version instead. Application versions are independent of `PROJECT_SCHEMA_VERSION`, which changes only when the saved sensor-project file format requires migration.

Before publishing a release, test the ZIP on a clean Windows workstation or VM with no source checkout or Python installation. Confirm it launches, validates a sensor, saves and reloads a project, and writes a KiCad footprint. Also compare the ZIP hash with `SHA256SUMS.txt` using `Get-FileHash`.

## Architecture

- `ips_sensor.models`: immutable request, sensor, and output configuration models.
- `ips_sensor.parameters`: the single UI/validation/help metadata catalog for all current properties.
- `ips_sensor.engine`: the sensor-definition registry, strict/relaxed validation policy, and adapter over the verified routing engine.
- `ips_sensor.layout`: neutral lines, arcs, and through-hole pads shared by preview and exporters.
- `ips_sensor.exporters`: KiCad 9 exporter registry; future Altium exporters can consume the same layout.
- `ips_sensor.gui`: standalone PySide6 interface.

The legacy generator is deliberately retained during this migration: the new core converts typed configuration to its established routing routines and converts their output into platform-neutral primitives. This keeps current coil-routing behavior stable while allowing new sensor shapes and exporters to be added through the registry.

## Advanced primary-corridor placement

**Primary corridor top inset** controls the vertical placement of the shared VIN/OSC corridor. `0 mm` means **Automatic** and preserves the verified clearance-safe routing. A positive value is measured downward from the physical outer copper edge of the primary coil's top rail. It is an advanced setting because an overly small manual inset can violate copper clearance; strict generation reports that condition, while **Generate anyway** remains available for renderable layouts after explicit confirmation. Existing schema-v1 project files are migrated to `0 mm`, because their previously stored value did not affect routing. Legacy script overrides of `osc1_vin_exit_offset_mm` now use this active top-inset meaning.
