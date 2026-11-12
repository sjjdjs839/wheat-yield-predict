param([ValidateSet("start", "stop", "status")]$Action = "status")

# 项目内免安装 MySQL 8 控制脚本（数据物理位置：<项目>/data/mysql）
# 用法：powershell -ExecutionPolicy Bypass -File scripts\mysql_server.ps1 start

$Base = "C:\wheat_mysql\mysql-8.0.24-winx64"
$Data = "C:\wheat_mysql\data"
$Mysqld = Join-Path $Base "bin\mysqld.exe"
$Port = 3307

function Get-PortPid {
    (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
        Select-Object -First 1).OwningProcess
}

switch ($Action) {
    "start" {
        $pid = Get-PortPid
        if ($pid) { "MySQL 已在运行（PID $pid，端口 $Port）"; break }
        Start-Process -FilePath $Mysqld `
            -ArgumentList "--basedir=$Base", "--datadir=$Data", "--port=$Port",
                          "--local-infile=1", "--character-set-server=utf8mb4" `
            -WindowStyle Hidden
        Start-Sleep -Seconds 4
        $pid = Get-PortPid
        if ($pid) { "MySQL 已启动：127.0.0.1:$Port（PID $pid）" } else { "启动失败，请查看 $Data\error.log" }
    }
    "stop" {
        $pid = Get-PortPid
        if (-not $pid) { "MySQL 未在运行"; break }
        Stop-Process -Id $pid -Force
        "MySQL 已停止（PID $pid）"
    }
    default {
        $pid = Get-PortPid
        if ($pid) { "运行中：127.0.0.1:$Port（PID $pid）" } else { "未运行（端口 $Port 无监听）" }
    }
}
