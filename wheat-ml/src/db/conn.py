"""MySQL 连接和可用性探测，结果缓存 5 秒，连不上就自动降级。

参数取值优先级：环境变量 > config/config.yaml 的 database 段。换机器部署
不用改文件，设环境变量就行：

    WHEAT_DB_ENABLED=0        不开库，全程走 CSV（最省事）
    WHEAT_DB_HOST=127.0.0.1
    WHEAT_DB_PORT=3306
    WHEAT_DB_USER=root
    WHEAT_DB_PASSWORD=<你的密码>
    WHEAT_DB_NAME=wheat_ml

连不上或者表是空的，上层自己回退到内置 CSV，页面照常能用。
"""

import os
import time
import threading
import yaml
import pymysql
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
with open(PROJECT_ROOT / "config" / "config.yaml", "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

DB = dict(CONFIG["database"])


def _as_bool(v: str) -> bool:
    return v.strip().lower() not in ("0", "false", "no", "off", "")


_ENV_MAP = {
    "WHEAT_DB_ENABLED": ("enabled", _as_bool),
    "WHEAT_DB_HOST": ("host", str),
    "WHEAT_DB_PORT": ("port", int),
    "WHEAT_DB_USER": ("user", str),
    "WHEAT_DB_PASSWORD": ("password", str),
    "WHEAT_DB_NAME": ("name", str),
}
for _env_key, (_field, _cast) in _ENV_MAP.items():
    _raw = os.environ.get(_env_key)
    if _raw is None or _raw == "":
        continue
    try:
        DB[_field] = _cast(_raw)
    except (TypeError, ValueError):
        pass

DB_SOURCE = "env" if any(os.environ.get(k) for k in _ENV_MAP) else "config"
_TTL = 5.0
_cache = {"ts": 0.0, "ok": False}
_lock = threading.Lock()


def connect(with_database: bool = True):
    """建立连接；with_database=False 用于建库"""
    kwargs = dict(
        host=DB["host"], port=int(DB["port"]), user=DB["user"],
        password=DB["password"], charset=DB.get("charset", "utf8mb4"),
        connect_timeout=int(DB.get("connect_timeout", 3)),
        autocommit=True, local_infile=bool(DB.get("local_infile", False)),
    )
    if with_database:
        kwargs["database"] = DB["name"]
    return pymysql.connect(**kwargs)


def available() -> bool:
    """数据库是否可用（结果缓存 5 秒，避免每次请求都探测）"""
    if not DB.get("enabled"):
        return False
    if not str(DB.get("password") or "").strip():
        # 没配密码就不去撞连接：本机没库时会白等一个超时，页面跟着变慢
        return False
    now = time.time()
    with _lock:
        if now - _cache["ts"] < _TTL:
            return _cache["ok"]
        try:
            conn = connect()
            conn.ping(reconnect=False)
            conn.close()
            ok = True
        except Exception:
            ok = False
        _cache.update(ts=now, ok=ok)
        return ok
