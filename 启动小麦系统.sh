#!/usr/bin/env bash
# 豫麦智测 · macOS / Linux 启动器
set -euo pipefail
cd "$(dirname "$0")"

echo "========================================"
echo "  豫麦智测 · 河南省小麦气象数据分析与产量预测系统"
echo "========================================"

PY="$(command -v python3 || command -v python || true)"
if [ -z "$PY" ]; then
  echo "[错误] 没有找到 Python，请先安装 Python 3.12 或更高版本。"
  exit 1
fi
echo "[环境] Python: $PY"

if ! "$PY" -c "import flask, pandas, numpy, sklearn, joblib, yaml, openpyxl, pymysql, matplotlib, lightgbm, reportlab, xlrd" >/dev/null 2>&1; then
  echo "[依赖] 首次运行需要安装依赖，约 2-5 分钟..."
  "$PY" -m pip install -r requirements.txt
fi
echo "[依赖] 已就绪"

cd wheat-ml
if [ ! -f "data/processed/weather_clean.csv" ] || [ ! -f "output/models/model_rf.joblib" ]; then
  echo "[数据] 本机缺少清洗数据或模型，将从 data/raw 自动生成一次（约 1-2 分钟）。"
  "$PY" -m scripts.run_pipeline
fi

echo "[启动] 服务地址 http://127.0.0.1:5000 （Ctrl+C 停止）"
exec "$PY" run.py
