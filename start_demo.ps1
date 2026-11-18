param(
    [switch]$NoBrowser,
    [switch]$WithWorldBackground,
    [double]$SupportPlaneZ = 0.0
)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot

# Load .env into current environment if present
$EnvFile = Join-Path $Root ".env"
if (Test-Path $EnvFile) {
    Get-Content $EnvFile | ForEach-Object {
        $line = $_.Trim()
        if ($line -and -not $line.StartsWith("#") -and $line.Contains("=")) {
            $k, $v = $line.Split("=", 2)
            $k = $k.Trim()
            $v = $v.Trim().Trim('"').Trim("'")
            if ($k -and $v -and -not [Environment]::GetEnvironmentVariable($k)) {
                [Environment]::SetEnvironmentVariable($k, $v, "Process")
            }
        }
    }
}

$Py = Resolve-Path (Join-Path $Root "..\..\work\venv\Scripts\python.exe") -ErrorAction SilentlyContinue
if (-not $Py) {
    $Py = Resolve-Path (Join-Path $Root ".venv\Scripts\python.exe") -ErrorAction SilentlyContinue
}
if (-not $Py) {
    throw "Python simulation environment not found (expected ..\..\work\venv\Scripts\python.exe or .venv\Scripts\python.exe)"
}
$PyPath = $Py.Path

function Test-HttpReady([string]$Url) {
    try {
        $resp = Invoke-WebRequest -Uri $Url -TimeoutSec 2 -UseBasicParsing
        return ($resp.StatusCode -ge 200 -and $resp.StatusCode -lt 400)
    } catch {
        return $false
    }
}

Write-Host "[1/3] Preparing and validating hackathon_demo project..." -ForegroundColor Cyan
$worldArg = if ($WithWorldBackground) { "True" } else { "False" }
& $PyPath -c "from pathlib import Path; from vlab.store import ConfigStore; from vlab.hackathon_demo import setup_hackathon_demo; r = setup_hackathon_demo(Path(r'$Root'), ConfigStore(Path(r'$Root') / 'projects'), with_world_background=$worldArg, support_plane_z=$SupportPlaneZ); print('Demo ready:', r['project_id'], 'v' + str(r['version']), 'boxes=' + str(r['tray_compound_boxes']), 'dynamic=' + str(r['dynamic_nodes']))"

Write-Host "[2/3] Ensuring backend (127.0.0.1:8767) and frontend (127.0.0.1:5173) are running..." -ForegroundColor Cyan
if (-not (Test-HttpReady "http://127.0.0.1:8767/api/v1/capabilities")) {
    # Start uvicorn in a visible desktop console so MuJoCo OpenGL camera rendering works reliably.
    Start-Process -FilePath $PyPath -ArgumentList "-m", "uvicorn", "vlab.api:app", "--host", "127.0.0.1", "--port", "8767" -WorkingDirectory $Root
} else {
    Write-Host "  Backend already listening on http://127.0.0.1:8767" -ForegroundColor DarkGray
}

if (-not (Test-HttpReady "http://127.0.0.1:5173/")) {
    $FrontendDir = Join-Path $Root "frontend"
    Start-Process -FilePath "npm.cmd" -ArgumentList "run", "dev", "--", "--host", "127.0.0.1", "--port", "5173" -WorkingDirectory $FrontendDir
} else {
    Write-Host "  Frontend already listening on http://127.0.0.1:5173" -ForegroundColor DarkGray
}

$deadline = (Get-Date).AddSeconds(35)
while ((Get-Date) -lt $deadline) {
    if ((Test-HttpReady "http://127.0.0.1:8767/api/v1/capabilities") -and (Test-HttpReady "http://127.0.0.1:5173/")) {
        break
    }
    Start-Sleep -Milliseconds 600
}

if (-not (Test-HttpReady "http://127.0.0.1:8767/api/v1/capabilities") -or -not (Test-HttpReady "http://127.0.0.1:5173/")) {
    throw "Timed out waiting for RoboForge backend (8767) or frontend (5173) to become ready."
}

$DemoUrl = "http://127.0.0.1:5173/projects/hackathon_demo/sim"
Write-Host "[3/3] RoboForge x Aholo Hackathon Demo is live!" -ForegroundColor Green
Write-Host "  Demo URL : $DemoUrl"
Write-Host "  API Docs : http://127.0.0.1:8767/docs"

if (-not $NoBrowser) {
    Start-Process $DemoUrl
}
