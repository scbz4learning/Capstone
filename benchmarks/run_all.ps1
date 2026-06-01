$BENCHMARKS_DIR = $PSScriptRoot
$ROOT_DIR = Split-Path -Parent $BENCHMARKS_DIR
$LOG_DIR = "$BENCHMARKS_DIR\run_logs"
$PYTHON = "$ROOT_DIR\.venv\Scripts\python.exe"

$env:TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL = "1"

if (-not (Test-Path $LOG_DIR)) { New-Item -ItemType Directory -Path $LOG_DIR -Force }

function Run-Test {
    param($Name, $Script, $Config, $LogFile)
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$ts] Starting: $Name" | Out-File "$LOG_DIR\run_all.log" -Append
    & $PYTHON "$BENCHMARKS_DIR\$Script" --config "$BENCHMARKS_DIR\$Config" *> "$LOG_DIR\$LogFile"
    $te = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$te] Finished: $Name (exit: $LASTEXITCODE)" | Out-File "$LOG_DIR\run_all.log" -Append
}

Run-Test "VGGT Windows"   "comprehensive_profile_vggt.py"    "configs\vggt-windows.json"   "vggt_windows.log"
Run-Test "VGGT WSL"       "comprehensive_profile_vggt.py"    "configs\vggt-wsl.json"       "vggt_wsl.log"
Run-Test "SmolVLM Windows" "comprehensive_profile_smolvlm.py" "configs\smolvlm-windows.json" "smolvlm_windows.log"
Run-Test "SmolVLM WSL"     "comprehensive_profile_smolvlm.py" "configs\smolvlm-wsl.json"     "smolvlm_wsl.log"

"$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ALL DONE" | Out-File "$LOG_DIR\run_all.log" -Append