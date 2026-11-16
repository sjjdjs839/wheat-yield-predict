@echo off
setlocal
title 豫麦智测服务
cd /d "%~dp0wheat-ml"

echo ========================================
echo   河南省小麦气象数据分析与产量预测系统
echo ========================================
echo.

set "PY="
python --version >nul 2>nul
if not errorlevel 1 set "PY=python"
if not defined PY (
    py -3 --version >nul 2>nul
    if not errorlevel 1 set "PY=py -3"
)
if not defined PY if exist "D:\python\python.exe" set "PY=D:\python\python.exe"
if not defined PY (
    echo [错误] 未检测到 Python，请先安装 Python 3.12+ 并加入 PATH。
    pause
    exit /b 1
)

%PY% --version >nul 2>nul
if errorlevel 1 (
    echo [错误] Python 无法正常运行：%PY%
    pause
    exit /b 1
)

%PY% -c "import flask, pandas, numpy, sklearn, joblib, yaml, openpyxl, pymysql, matplotlib, lightgbm, reportlab, xlrd" >nul 2>nul
if errorlevel 1 (
    echo [错误] Python 依赖不完整。
    echo 请先运行：%PY% -m pip install -r requirements.txt
    pause
    exit /b 1
)

echo [清理] 检查并结束占用 5000 端口的旧服务...
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\stop_app.ps1 >nul 2>nul

echo [启动] 豫麦智测服务... 地址: http://127.0.0.1:5000
echo [说明] 服务就绪后会自动打开默认浏览器；关闭本窗口即停止系统。
echo.

%PY% run.py

echo.
echo [提示] 服务已停止。若上方有报错信息，请截图保存。
pause
