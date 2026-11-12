# 停止占用 5000 端口的本项目进程（只匹配 app.py / run.py）。
$ErrorActionPreference = "SilentlyContinue"

$listeners = Get-NetTCPConnection -LocalPort 5000 -State Listen -ErrorAction SilentlyContinue
foreach ($listener in $listeners) {
    $proc = Get-CimInstance Win32_Process -Filter "ProcessId=$($listener.OwningProcess)"
    if ($proc -and $proc.CommandLine -match "(?i)(app\.py|run\.py)") {
        Stop-Process -Id $proc.ProcessId -Force -ErrorAction SilentlyContinue
    }
}

Start-Sleep -Milliseconds 1000
