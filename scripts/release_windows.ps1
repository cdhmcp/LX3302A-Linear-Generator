[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^\d+\.\d+\.\d+$')]
    [string]$Version,

    [Parameter()]
    [ValidateScript({ Test-Path -LiteralPath $_ -PathType Leaf })]
    [string]$ReleaseNotesPath
)

$ErrorActionPreference = "Stop"

function Invoke-Git {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)

    $result = & git @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "git $($Arguments -join ' ') failed with exit code $LASTEXITCODE."
    }
    return @($result)
}

function Get-ProjectVersion {
    param([Parameter(Mandatory = $true)][string]$Path)

    $contents = Get-Content -LiteralPath $Path -Raw
    $match = [regex]::Match(
        $contents,
        '(?m)^\s*version\s*=\s*"(?<version>[^"]+)"\s*$'
    )
    if (-not $match.Success) {
        throw "Could not find [project] version in $Path."
    }
    return $match.Groups['version'].Value
}

function Get-ReleaseNotes {
    param(
        [Parameter(Mandatory = $true)][string]$Tag,
        [Parameter(Mandatory = $true)][string]$Version,
        [Parameter(Mandatory = $true)][string]$Commit,
        [string]$NotesPath
    )

    if ($NotesPath) {
        return Get-Content -LiteralPath $NotesPath -Raw
    }

    $previousTag = $null
    $candidateTags = Invoke-Git -Arguments @(
        'tag', '--merged', 'HEAD', '--sort=-creatordate', '--format=%(refname:short)'
    )
    foreach ($candidate in $candidateTags) {
        if ($candidate -and $candidate.Trim() -ne $Tag) {
            $previousTag = $candidate.Trim()
            break
        }
    }

    $logRange = if ($previousTag) { "$previousTag..HEAD" } else { 'HEAD' }
    $changes = Invoke-Git -Arguments @(
        'log', '--pretty=format:- %s (%h)', $logRange
    )
    if (-not $changes) {
        $changes = @('- Internal release build; no commit subjects were available.')
    }

    $rangeLabel = if ($previousTag) { "Changes since $previousTag" } else { 'Included changes' }
    return @(
        "# IPS Sensor GUI v$Version",
        '',
        "Released: $(Get-Date -Format 'yyyy-MM-dd')",
        "Git tag: $Tag",
        "Commit: $Commit",
        '',
        "## $rangeLabel",
        $changes,
        '',
        'This is an internal Windows release. Extract the ZIP to a local folder before running the application.'
    ) -join [Environment]::NewLine
}

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Tag = "v$Version"
$ProjectFile = Join-Path $ProjectRoot 'pyproject.toml'
$ReleaseRoot = Join-Path $ProjectRoot (Join-Path 'dist\releases' $Tag)
$StagingParent = Join-Path $ProjectRoot 'dist\release-staging'
$StagingRoot = Join-Path $StagingParent "$Tag-$PID"
$BundleName = "IPS-Sensor-GUI-v$Version-win64"
$BundleDirectory = Join-Path $StagingRoot $BundleName
$ArchivePath = Join-Path $StagingRoot "$BundleName.zip"
$ReleasePublished = $false

if ($ReleaseNotesPath) {
    $ReleaseNotesPath = (Resolve-Path -LiteralPath $ReleaseNotesPath).Path
}

Push-Location $ProjectRoot
try {
    $projectVersion = Get-ProjectVersion -Path $ProjectFile
    if ($projectVersion -ne $Version) {
        throw "Requested version $Version does not match pyproject.toml version $projectVersion."
    }

    $dirtyTrackedFiles = Invoke-Git -Arguments @('status', '--porcelain', '--untracked-files=no')
    if ($dirtyTrackedFiles -and $dirtyTrackedFiles.Count -gt 0) {
        throw "Release builds require no tracked uncommitted changes. Commit or stash them before releasing."
    }

    $headTags = Invoke-Git -Arguments @('tag', '--points-at', 'HEAD', '--format=%(refname:short)')
    if ($headTags -notcontains $Tag) {
        throw "HEAD must have the annotated tag $Tag before this release can be built."
    }
    $tagObjectType = (Invoke-Git -Arguments @('cat-file', '-t', "refs/tags/$Tag") | Select-Object -First 1).Trim()
    if ($tagObjectType -ne 'tag') {
        throw "$Tag must be an annotated Git tag, not a lightweight tag."
    }

    if (Test-Path -LiteralPath $ReleaseRoot) {
        throw "Release directory already exists: $ReleaseRoot. Published releases are never overwritten."
    }
    if (-not [Environment]::Is64BitProcess) {
        throw "The win64 release archive must be built with a 64-bit PowerShell/Python environment."
    }

    $commit = (Invoke-Git -Arguments @('rev-parse', 'HEAD') | Select-Object -First 1).Trim()
    $shortCommit = (Invoke-Git -Arguments @('rev-parse', '--short=12', 'HEAD') | Select-Object -First 1).Trim()

    python -m pip install -e ".[gui,package]"
    if ($LASTEXITCODE -ne 0) {
        throw "Could not install release build dependencies."
    }

    python -m unittest discover -v
    if ($LASTEXITCODE -ne 0) {
        throw "The test suite failed; no release archive was created."
    }

    & (Join-Path $PSScriptRoot 'build_windows.ps1') -SkipDependencyInstall
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller packaging failed."
    }

    $builtBundle = Join-Path $ProjectRoot 'dist\IPS-Sensor-GUI'
    $executable = Join-Path $builtBundle 'IPS-Sensor-GUI.exe'
    if (-not (Test-Path -LiteralPath $executable -PathType Leaf)) {
        throw "Expected packaged executable was not created: $executable"
    }

    New-Item -ItemType Directory -Path $StagingRoot | Out-Null
    Copy-Item -LiteralPath $builtBundle -Destination $BundleDirectory -Recurse

    $manifest = [ordered]@{
        product = 'IPS Sensor GUI'
        version = $Version
        git_tag = $Tag
        git_commit = $commit
        git_commit_short = $shortCommit
        build_utc = (Get-Date).ToUniversalTime().ToString('o')
        platform = 'Windows x64'
    } | ConvertTo-Json
    $manifest | Set-Content -LiteralPath (Join-Path $BundleDirectory 'RELEASE_MANIFEST.json') -Encoding utf8

    $installInstructions = @(
        'IPS Sensor GUI internal installation',
        '',
        '1. Extract the entire ZIP to a local folder. Do not run the application from the ZIP or a network share.',
        '2. Run IPS-Sensor-GUI.exe from the extracted IPS-Sensor-GUI-v<version>-win64 folder.',
        '3. Keep the extracted folder intact; the executable depends on the files beside it.',
        '',
        'If Windows Defender or SmartScreen blocks the application, contact the internal release owner or IT/security with the version and SHA-256 checksum.'
    ) -join [Environment]::NewLine
    $installInstructions | Set-Content -LiteralPath (Join-Path $BundleDirectory 'INSTALL.txt') -Encoding utf8

    Compress-Archive -LiteralPath $BundleDirectory -DestinationPath $ArchivePath -CompressionLevel Optimal
    $hash = Get-FileHash -LiteralPath $ArchivePath -Algorithm SHA256
    "$($hash.Hash.ToLowerInvariant()) *$([IO.Path]::GetFileName($ArchivePath))" |
        Set-Content -LiteralPath (Join-Path $StagingRoot 'SHA256SUMS.txt') -Encoding ascii

    $releaseNotes = Get-ReleaseNotes -Tag $Tag -Version $Version -Commit $commit -NotesPath $ReleaseNotesPath
    $releaseNotes | Set-Content -LiteralPath (Join-Path $StagingRoot "ReleaseNotes-v$Version.md") -Encoding utf8
    $installInstructions | Set-Content -LiteralPath (Join-Path $StagingRoot 'INSTALL.txt') -Encoding utf8

    New-Item -ItemType Directory -Path (Split-Path -Parent $ReleaseRoot) -Force | Out-Null
    Move-Item -LiteralPath $StagingRoot -Destination $ReleaseRoot
    $ReleasePublished = $true

    Write-Host "Release archive: $(Join-Path $ReleaseRoot "$BundleName.zip")"
    Write-Host "SHA-256 file: $(Join-Path $ReleaseRoot 'SHA256SUMS.txt')"
    Write-Host "Release notes: $(Join-Path $ReleaseRoot "ReleaseNotes-v$Version.md")"
}
finally {
    $shouldCleanStaging = (
        (-not $ReleasePublished) -and
        (Test-Path -LiteralPath $StagingRoot) -and
        $StagingRoot.StartsWith($StagingParent, [System.StringComparison]::OrdinalIgnoreCase)
    )
    if ($shouldCleanStaging) {
        Remove-Item -LiteralPath $StagingRoot -Recurse -Force
    }
    Pop-Location
}
