[CmdletBinding()]
param(
    [switch]$SkipDependencyInstall
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Push-Location $ProjectRoot
try {
    if (-not $SkipDependencyInstall) {
        python -m pip install -e ".[gui,package]"
        if ($LASTEXITCODE -ne 0) {
            throw "Could not install Windows package build dependencies."
        }
    }
    python -m PyInstaller --noconfirm --clean --windowed --onedir --name "IPS-Sensor-GUI" --collect-all PySide6 ips_sensor_gui.py
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller packaging failed."
    }
    Write-Host "Windows application bundle: $ProjectRoot\dist\IPS-Sensor-GUI"
}
finally {
    Pop-Location
}
