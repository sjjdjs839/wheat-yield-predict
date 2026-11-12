"""MySQL 连接与可用性探测（带 5 秒缓存，连接失败自动降级）"""

import time
import threading
import yaml
import pymysql
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
with open(PROJECT_ROOT / "config" / "config.yaml", "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

DB = CONFIG["database"]
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
