$SCRIPTS_DIR = "C:\Users\bokai\Documents\capstone"
$LOG_DIR = "$SCRIPTS_DIR\run_logs"
$PYTHON = "$SCRIPTS_DIR\.venv\Scripts\python.exe"

$env:TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL = "1"

if (-not (Test-Path $LOG_DIR)) { New-Item -ItemType Directory -Path $LOG_DIR -Force }

function Run-Test {
    param($Name, $Script, $Config, $LogFile)
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$ts] Starting: $Name" | Out-File "$LOG_DIR\run_all.log" -Append
    & $PYTHON "$SCRIPTS_DIR\$Script" --config "$SCRIPTS_DIR\$Config" *> "$LOG_DIR\$LogFile"
    $te = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$te] Finished: $Name (exit: $LASTEXITCODE)" | Out-File "$LOG_DIR\run_all.log" -Append
}

Run-Test "VGGT Windows"   "benchmarks\comprehensive_profile_vggt.py"    "benchmarks\configs\vggt-windows.json"   "vggt_windows.log"
# Run-Test "VGGT WSL"       "benchmarks\comprehensive_profile_vggt.py"    "benchmarks\configs\vggt-wsl.json"       "vggt_wsl.log"

"$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ALL DONE" | Out-File "$LOG_DIR\run_all.log" -Append