[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Push-Location $ProjectRoot
try {
    python -m pip install -e ".[gui,package]"
    python -m PyInstaller --noconfirm --clean --windowed --onedir --name "IPS-Sensor-GUI" --collect-all PySide6 ips_sensor_gui.py
    Write-Host "Windows application bundle: $ProjectRoot\dist\IPS-Sensor-GUI"
}
finally {
    Pop-Location
}
