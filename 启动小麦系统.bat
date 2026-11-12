@echo off
setlocal enabledelayedexpansion
title 豫麦智测服务
cd /d "%~dp0wheat-ml"

echo ========================================
echo   河南省小麦气象数据分析与产量预测系统
echo ========================================
echo.

set "PY="

REM 候选1：本机已装好全部依赖的 D:\python
if exist "D:\python\python.exe" (
    "D:\python\python.exe" -c "import flask,pandas,numpy,sklearn,joblib,yaml,openpyxl,pymysql,matplotlib,lightgbm,reportlab,xlrd" >nul 2>nul
    if not errorlevel 1 set "PY=D:\python\python.exe"
)

REM 候选2：py 启动器指定的 3.x
if not defined PY (
    py -3 -c "import flask,pandas,numpy,sklearn,joblib,yaml,openpyxl,pymysql,matplotlib,lightgbm,reportlab,xlrd" >nul 2>nul
    if not errorlevel 1 set "PY=py -3"
)

REM 候选3：PATH 中的 python
if not defined PY (
    python -c "import flask,pandas,numpy,sklearn,joblib,yaml,openpyxl,pymysql,matplotlib,lightgbm,reportlab,xlrd" >nul 2>nul
    if not errorlevel 1 set "PY=python"
)

if not defined PY (
    echo [提示] 未检测到带完整依赖的 Python，正在尝试自动安装依赖...
    where python >nul 2>nul && (
        python -m pip install -r requirements.txt
        python -c "import flask,pandas,numpy,sklearn,joblib,yaml,openpyxl,pymysql,matplotlib,lightgbm,reportlab,xlrd" >nul 2>nul
        if not errorlevel 1 set "PY=python"
    )
)

if not defined PY (
    echo [错误] 未找到可用的 Python 3.12+ 或依赖不完整。
    echo 请安装 Python 后执行：python -m pip install -r requirements.txt
    pause
    exit /b 1
)

echo [Python] !PY!
!PY! --version
echo.

echo [清理] 检查并结束占用 5000 端口的旧服务...
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\stop_app.ps1 >nul 2>nul

echo [启动] 豫麦智测服务... 地址: http://127.0.0.1:5000
echo [说明] 服务就绪后自动打开浏览器；关闭本窗口即停止系统。
echo.

!PY! run.py

echo.
echo [提示] 服务已停止。若上方有报错信息，请截图保存。
pause
