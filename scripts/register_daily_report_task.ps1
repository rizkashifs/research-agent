param(
    [string]$TaskName = "ResearchAgentDailyAIReport",
    [string]$Time = "11:00",
    [switch]$Online,
    [string]$PythonPath = "",
    [string]$ProjectRoot = ""
)

$ErrorActionPreference = "Stop"

if (-not $ProjectRoot) {
    $ProjectRoot = (Resolve-Path "$PSScriptRoot\..").Path
}

if (-not $PythonPath) {
    $VenvPython = Join-Path $ProjectRoot "venv\Scripts\python.exe"
    if (Test-Path $VenvPython) {
        $PythonPath = $VenvPython
    } else {
        $PythonPath = "python"
    }
}

$ScriptPath = Join-Path $ProjectRoot "scripts\daily_ai_research_report.py"
$Arguments = "`"$ScriptPath`""
if ($Online) {
    $Arguments += " --online"
}

$Action = New-ScheduledTaskAction -Execute $PythonPath -Argument $Arguments -WorkingDirectory $ProjectRoot
$Trigger = New-ScheduledTaskTrigger -Daily -At $Time
$Settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -WakeToRun `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 60) `
    -ExecutionTimeLimit (New-TimeSpan -Hours 1)

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger $Trigger `
    -Settings $Settings `
    -Description "Generate and email the Research Agent daily AI report." `
    -Force

Write-Host "Registered scheduled task '$TaskName' to run daily at $Time."
Write-Host "Project root: $ProjectRoot"
Write-Host "Python: $PythonPath"
Write-Host "Online mode: $($Online.IsPresent)"
