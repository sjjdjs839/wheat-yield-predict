@echo off
setlocal
title 豫麦智测 · 启动器
cd /d "%~dp0"

echo ========================================
echo   豫麦智测 · 河南省小麦气象数据分析与产量预测系统
echo ========================================
echo.

REM ---------- 1. 定位 Python ----------
set "PY="
python --version >nul 2>nul
if not errorlevel 1 set "PY=python"
if not defined PY (
    py -3 --version >nul 2>nul
    if not errorlevel 1 set "PY=py -3"
)
if not defined PY (
    for %%D in ("C:\Python312\python.exe" "C:\Python313\python.exe" "D:\python\python.exe") do (
        if not defined PY if exist %%D set "PY=%%~D"
    )
)
if not defined PY (
    echo [错误] 没有找到 Python。
    echo        请安装 Python 3.12 或更高版本，安装时务必勾选 "Add Python to PATH"。
    echo        下载地址: https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)
echo [环境] Python 解释器: %PY%

REM ---------- 2. 依赖 ----------
%PY% -c "import flask, pandas, numpy, sklearn, joblib, yaml, openpyxl, pymysql, matplotlib, lightgbm, reportlab, xlrd" >nul 2>nul
if errorlevel 1 (
    echo [依赖] 首次运行需要安装依赖，约 2-5 分钟，请勿关闭本窗口...
    %PY% -m pip install --upgrade pip >nul 2>nul
    %PY% -m pip install -r "%~dp0requirements.txt"
    if errorlevel 1 (
        echo.
        echo [错误] 依赖安装失败。请检查网络后手动执行下面这行:
        echo        %PY% -m pip install -r "%~dp0requirements.txt"
        pause
        exit /b 1
    )
)
echo [依赖] 已就绪

REM ---------- 3. 数据与模型 ----------
cd /d "%~dp0wheat-ml"
if exist "data\processed\weather_clean.csv" if exist "output\models\model_rf.joblib" goto RUN

echo.
echo [数据] 本机缺少清洗数据或模型，将从 data\raw 自动生成一次。
echo        约 1-2 分钟，仅首次执行，之后一直复用。
echo.
%PY% -m scripts.run_pipeline
if errorlevel 1 (
    echo.
    echo [错误] 数据生成失败。请把上面的报错截图发给团队成员。
    pause
    exit /b 1
)

:RUN
echo.
echo [清理] 检查并结束占用 5000 端口的旧服务...
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\stop_app.ps1 >nul 2>nul

echo [启动] 服务地址 http://127.0.0.1:5000
echo        启动后会自动打开默认浏览器；关闭本窗口即停止系统。
echo.
%PY% run.py

echo.
echo [提示] 服务已停止；上方如有报错请截图发团队。
pause
