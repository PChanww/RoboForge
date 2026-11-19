# Start the isolated local SmolVLA service; leave the simulator environment intact.
param(
    [ValidateSet('cuda', 'cpu')][string]$Device = 'cuda',
    [ValidateSet('auto', 'float32', 'bfloat16')][string]$Precision = 'auto'
)
$ErrorActionPreference = 'Stop'
$transportRoot = Split-Path -Parent $PSScriptRoot
$transportPython = Join-Path $transportRoot '.venv-vla\Scripts\python.exe'
$transportWeights = Join-Path $transportRoot 'work\smolvla-libero'
$transportProcessor = Join-Path $transportRoot 'work\smolvlm-processor'
foreach ($transportRequired in @($transportPython, (Join-Path $transportWeights 'model.safetensors'), (Join-Path $transportProcessor 'tokenizer.json'))) {
    if (-not (Test-Path -LiteralPath $transportRequired)) {
        throw "Missing $transportRequired. See frontend/VLA_WORKBENCH.md for setup."
    }
}
$transportHealth = $null
try {
    $transportHealth = Invoke-RestMethod 'http://127.0.0.1:8772/health' -TimeoutSec 2
} catch {
    # A stopped service is expected before the first launch.
}
if ($transportHealth) {
    $transportActiveDevice = $transportHealth.identity.device
    if (-not $transportActiveDevice) { $transportActiveDevice = $transportHealth.requested_device }
    if ($transportActiveDevice -and $transportActiveDevice -notlike "$Device*") {
        throw "Existing VLA service uses $transportActiveDevice; requested $Device. Stop the old service before restarting."
    }
    if ($transportHealth.state -eq 'error') {
        throw "Existing VLA service failed: $($transportHealth.error). Stop it before restarting."
    }
    Write-Output "VLA service already running: $($transportHealth.state) $transportActiveDevice"
    return
}
$transportLog = Join-Path $transportRoot 'work\transport-vla.stdout.log'
$transportErrorLog = Join-Path $transportRoot 'work\transport-vla.stderr.log'
$transportArguments = @('integrations/transport_smolvla_server.py', '--weights', 'work/smolvla-libero', '--processor', 'work/smolvlm-processor', '--device', $Device, '--precision', $Precision, '--port', '8772')
$transportProcess = Start-Process -FilePath $transportPython -ArgumentList $transportArguments -WorkingDirectory $transportRoot -WindowStyle Hidden -RedirectStandardOutput $transportLog -RedirectStandardError $transportErrorLog -PassThru
Write-Output "VLA service loading (PID $($transportProcess.Id)). Check the policy page or http://127.0.0.1:8772/health."
Write-Output "Logs: $transportLog; $transportErrorLog"
