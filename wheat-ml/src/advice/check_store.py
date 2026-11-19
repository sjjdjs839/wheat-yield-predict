"""作业卡打勾记录。

县区 + 年份 + 生育期 + 卡片，四元组唯一，重复勾选就覆盖，不会堆出一堆重复行。
done=1 表示这张卡干完了，还能撤——再点一次回到未执行。

存储先走 MySQL（advice_checks 表，第一次用的时候自己建），库连不上或者建表
失败就退回本地 CSV，单机演示不会中途断。

有了这个，农事建议就不只是"读一读"，是能派下去、能打勾、能留痕。页面上
显示本期完成度，对外演示时这段最像真的在用。
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


# --- CSV 回退 ---

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
