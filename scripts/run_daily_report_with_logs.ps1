param(
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
$LogDir = Join-Path $ProjectRoot "results\daily\logs\task"
New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
$LogPath = Join-Path $LogDir ("{0}_daily_ai_research_report_task.log" -f (Get-Date -Format "yyyy-MM-dd"))

Start-Transcript -Path $LogPath -Append | Out-Null
try {
    Write-Host ("[{0}] Starting daily report task" -f (Get-Date -Format "s"))
    Write-Host ("Project root: {0}" -f $ProjectRoot)
    Write-Host ("Python: {0}" -f $PythonPath)
    Write-Host ("Script: {0}" -f $ScriptPath)
    Write-Host ("Online mode: {0}" -f $Online.IsPresent)

    $arguments = @($ScriptPath)
    if ($Online) {
        $arguments += "--online"
    }

    & $PythonPath @arguments 2>&1 | ForEach-Object {
        $_
    }

    $exitCode = $LASTEXITCODE
    Write-Host ("[{0}] Python exit code: {1}" -f (Get-Date -Format "s"), $exitCode)
    if ($exitCode -ne 0) {
        exit $exitCode
    }

    Write-Host ("[{0}] Daily report task completed successfully" -f (Get-Date -Format "s"))
}
catch {
    Write-Error $_
    throw
}
finally {
    Stop-Transcript | Out-Null
}
