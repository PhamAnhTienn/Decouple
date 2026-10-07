<#
.SYNOPSIS
    Runs TFRNet on Decouple sequentially across 5 seeds and missing rates 0.0 to 0.9.
.EXAMPLE
    .\scripts\run_tfr_net_missing.ps1 -Gpu 0 -Dataset mosi
#>

param(
    [string]$Dataset = "mosi",
    [int]$Gpu = 0,
    [string]$DataDir = "./data",
    [string]$ConfigFile = "",
    [int]$BatchSize = 0,
    [switch]$NoResume,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$DecoupleRoot = Split-Path -Parent $ScriptDir

# Detect Python interpreter
$PythonExec = "python"
if (Test-Path "C:\Users\PC\miniconda3\envs\decouple_env\python.exe") {
    $PythonExec = "C:\Users\PC\miniconda3\envs\decouple_env\python.exe"
}

if (-not $ConfigFile) {
    $ConfigFile = Join-Path $DecoupleRoot "config\tfr_net_${Dataset}_config.json"
    if (-not (Test-Path $ConfigFile)) {
        $ConfigFile = Join-Path $DecoupleRoot "config\tfr_net_mosi_config.json"
    }
}

$Seeds = @(1111, 1112, 1113, 1114, 1115)
$MissingRates = @(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "Starting TFRNet Sequential Benchmark (Seeds: $Seeds | Missing Rates: 0.0 -> 0.9)" -ForegroundColor Cyan
Write-Host "Python: $PythonExec" -ForegroundColor Cyan
Write-Host "Config: $ConfigFile" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

$TotalRuns = $Seeds.Count * $MissingRates.Count
$CurrentRun = 0

foreach ($seed in $Seeds) {
    foreach ($rate in $MissingRates) {
        $CurrentRun++
        $RateStr = $rate.ToString("0.0", [System.Globalization.CultureInfo]::InvariantCulture)
        $ModelSaveDir = Join-Path $DecoupleRoot "pt\seed_$seed"
        $ResSaveDir = Join-Path $DecoupleRoot "result\seed_$seed"
        $LogDir = Join-Path $DecoupleRoot "log\seed_$seed"
        $CkptPath = Join-Path $ModelSaveDir "tfr_net-$Dataset-missing$RateStr.pth"

        Write-Host "`n>>> [$CurrentRun/$TotalRuns] Seed: $seed | Missing Rate: $RateStr" -ForegroundColor Yellow

        if (-not $NoResume -and (Test-Path $CkptPath)) {
            Write-Host "[Skip] Checkpoint exists: $CkptPath" -ForegroundColor Green
            continue
        }

        $ArgsList = @(
            (Join-Path $DecoupleRoot "main.py"),
            "--model", "tfr_net",
            "--dataset", $Dataset,
            "--config_file", $ConfigFile,
            "--data_missing",
            "--missing_rate_eval_test", $RateStr,
            "--seeds", $seed,
            "--gpu_ids", $Gpu,
            "--data_dir", $DataDir,
            "--model_save_dir", $ModelSaveDir,
            "--res_save_dir", $ResSaveDir,
            "--log_dir", $LogDir,
            "--num_workers", 0
        )

        if ($BatchSize -gt 0) {
            $ArgsList += @("--batch_size", $BatchSize)
        }

        Write-Host "$PythonExec $($ArgsList -join ' ')" -ForegroundColor Gray

        if ($DryRun) {
            continue
        }

        Push-Location $DecoupleRoot
        try {
            & $PythonExec @ArgsList
        } catch {
            Write-Host "[Error] Run failed for seed $seed missing rate $RateStr : $_" -ForegroundColor Red
        } finally {
            Pop-Location
        }
    }
}

Write-Host "`nBenchmark completed!" -ForegroundColor Green
