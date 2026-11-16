"""
作业卡执行勾选存储
==================
- 粒度：县区 + 年份 + 生育期 + 卡片 四元组唯一，重复勾选覆盖更新。
- 语义：done=1 表示该卡已执行，可撤销（再点一次变回未执行）。
- 存储链路：MySQL 优先（advice_checks 表，首次使用时自动建表）；
  库不可用或建表失败时回退本地 CSV，保证单机演示不中断。

这样农事建议就从「读一读」变成「派下去 → 打勾 → 留痕」，
页面能显示本期作业完成度，也是对外演示时最能体现落地的一环。
"""

from pathlib import Path

import pandas as pd

from src.db import repository as repo
from src.db.conn import available

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = PROJECT_ROOT / "web" / "uploads" / "advice_checks.csv"
COLS = ["county", "year", "stage_key", "card_key", "done", "updated_at"]

_table_ready = False


def _ensure() -> bool:
    """确保 MySQL 侧表存在；只尝试一次，失败即走 CSV。"""
    global _table_ready
    if _table_ready:
        return True
    if not available():
        return False
    try:
        repo.ensure_check_table()
        _table_ready = True
        return True
    except Exception:
        return False


def load(county: str, year: int) -> dict:
    """返回 {(stage_key, card_key): True}，只含已执行的卡。"""
    done: dict = {}
    if _ensure():
        try:
            df = repo.advice_checks(county, int(year))
            for _, r in df.iterrows():
                if int(r["done"] or 0) == 1:
                    done[(str(r["stage_key"]), str(r["card_key"]))] = True
            return done
        except Exception:
            pass
    df = _csv_read()
    if df.empty:
        return done
    sub = df[(df["county"] == county) & (df["year"] == int(year)) & (df["done"] == 1)]
    for _, r in sub.iterrows():
        done[(str(r["stage_key"]), str(r["card_key"]))] = True
    return done


def toggle(county: str, year: int, stage_key: str, card_key: str, done: bool) -> str:
    """写入勾选状态，返回存储来源 mysql/csv。"""
    row = {"county": county, "year": int(year), "stage_key": stage_key,
           "card_key": card_key, "done": 1 if done else 0}
    if _ensure():
        try:
            repo.upsert_check(row)
            return "mysql"
        except Exception:
            pass
    _csv_upsert(row)
    return "csv"


# ------------------------- CSV 回退 -------------------------

def _csv_read() -> pd.DataFrame:
    if CSV_PATH.exists():
        try:
            return pd.read_csv(CSV_PATH)
        except Exception:
            return pd.DataFrame(columns=COLS)
    return pd.DataFrame(columns=COLS)


def _csv_upsert(row: dict) -> None:
    old = _csv_read()
    new = pd.DataFrame([row])
    if old.empty:
        merged = new
    else:
        mask = ~((old["county"] == row["county"]) & (old["year"] == row["year"])
                 & (old["stage_key"] == row["stage_key"]) & (old["card_key"] == row["card_key"]))
        merged = pd.concat([old[mask], new], ignore_index=True)
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
