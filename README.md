# Inductive Sensor Footprint Generator

This project generates KiCad 9 footprints for the LX3302A linear inductive position-sensor coils. It now includes a reusable typed core and a Windows-first PySide6 desktop GUI, so sensor settings can be changed without editing the generator source.

## Run the desktop GUI

Install the GUI optional dependency, then start the application:

```powershell
python -m pip install -e .[gui]
ips-sensor-gui
```

The application presents common settings first and exposes all routing and naming controls through **Show advanced**. Every field has a help button. Geometry checks run in the background; invalid but renderable geometry requires an explicit **Generate anyway** confirmation before KiCad output is written.

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
