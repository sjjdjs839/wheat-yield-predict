# -*- coding: utf-8 -*-
"""豫麦智测启动器。

前台运行 Flask（前端页面与后端接口同一进程），服务就绪后自动打开
系统默认浏览器。关闭控制台窗口会终止本进程，服务随即停止。

用法：
    python run.py                 # 启动并自动打开浏览器
    python run.py --no-browser    # 只启动服务，不打开浏览器
"""

from __future__ import annotations

import os
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
import warnings
import webbrowser
from pathlib import Path

# 模型用 scikit-learn 1.5.2 训练，当前运行库版本更高；忽略已知的版本提示，
# 避免启动日志被三条 InconsistentVersionWarning 淹没。
warnings.filterwarnings("ignore", message="Trying to unpickle estimator.*")
warnings.filterwarnings("ignore", message="X does not have valid feature names.*")

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import app as wheat  # noqa: E402

HOST = str(wheat.CONFIG["web"]["host"])
PORT = int(wheat.CONFIG["web"]["port"])
URL = f"http://{HOST}:{PORT}/"


def port_is_busy() -> bool:
    """检测端口上是否已有服务在监听，避免打开到旧服务。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.3)
        return sock.connect_ex((HOST, PORT)) == 0


def warm_caches() -> None:
    try:
        wheat.builtin_dataset()
        wheat.load_models()
        wheat.feature_importance_items()
    except Exception:
        pass


def open_browser_when_ready() -> None:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(URL, timeout=0.8) as response:
                if response.status == 200:
                    opened = webbrowser.open(URL)
                    if opened:
                        print(f"[浏览器] 已打开 {URL}", flush=True)
                    else:
                        print(f"[浏览器] 未能自动打开，请手动访问 {URL}", flush=True)
                    return
        except (urllib.error.URLError, TimeoutError, OSError):
            pass
        time.sleep(0.25)
    print(f"[浏览器] 等待超时，请手动访问 {URL}", flush=True)


def main() -> int:
    if port_is_busy():
        print(f"[错误] 端口 {PORT} 已被占用。", flush=True)
        print("请确认占用该端口的程序已关闭，或重新双击启动脚本清理旧服务。", flush=True)
        return 1

    no_browser = "--no-browser" in sys.argv or os.environ.get("WHEAT_NO_BROWSER") == "1"

    print("=" * 52, flush=True)
    print("  豫麦智测 · 河南省小麦气象分析与产量预测系统", flush=True)
    print(f"  服务地址：{URL}", flush=True)
    print("  关闭本窗口即停止服务；浏览器标签可以保留。", flush=True)
    print("=" * 52, flush=True)

    threading.Thread(target=warm_caches, daemon=True).start()
    if not no_browser:
        threading.Thread(target=open_browser_when_ready, daemon=True).start()

    try:
        wheat.app.run(
            host=HOST,
            port=PORT,
            debug=False,
            use_reloader=False,
        )
    except OSError as exc:
        print(f"[错误] 服务启动失败：{exc}", flush=True)
        print(f"请检查端口 {PORT} 是否被其他程序占用。", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
