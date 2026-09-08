<#
.SYNOPSIS
    Runs TFRNet on Decouple sequentially for CMU-MOSEI across 5 seeds and missing rates 0.0 to 0.9.
.EXAMPLE
    .\scripts\run_tfr_net_mosei_missing.ps1 -Gpu 0
#>

param(
    [int]$Gpu = 0,
    [string]$DataDir = "./data",
    [string]$ConfigFile = "",
    [switch]$DryRun
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RunnerScript = Join-Path $ScriptDir "run_tfr_net_missing.ps1"

$Params = @{
    Dataset = "mosei"
    Gpu = $Gpu
    DataDir = $DataDir
}

if ($ConfigFile) {
    $Params["ConfigFile"] = $ConfigFile
}
if ($DryRun) {
    $Params["DryRun"] = $true
}

& $RunnerScript @Params
