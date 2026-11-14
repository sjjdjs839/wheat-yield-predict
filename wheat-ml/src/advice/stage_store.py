"""
分阶段农事观测的累积存储
========================
- 阶段识别：按日期的「月-日」落入返青/拔节/抽穗扬花/灌浆/成熟区间，
  区间外记为 offseason（越冬 / 收获后，仅备注，不进阶段建议）。
- 累积语义：按 县区 + 日期 去重，重复上传同日覆盖更新，多次上传累加保留。
- 存储链路：MySQL 优先；库不可用、表缺失或写入异常时自动回退本地 CSV。
"""

import yaml
import pandas as pd
from pathlib import Path

from src.db import repository as repo
from src.db.conn import available

PROJECT_ROOT = Path(__file__).resolve().parents[2]
with open(PROJECT_ROOT / "config" / "config.yaml", "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

STAGES = CONFIG["growth_stages"]
STAGE_WEATHER = repo.STAGE_WEATHER
CSV_PATH = PROJECT_ROOT / "web" / "uploads" / "stage_observations.csv"


def classify_stage(month_day: str) -> str:
    """按月-日判定生育期；不在主生育期区间返回 offseason。"""
    if not month_day or pd.isna(month_day):
        return "offseason"
    for st in STAGES:
        if st["start"] <= month_day <= st["end"]:
            return st["key"]
    return "offseason"


def _frame(county: str, year: int, df: pd.DataFrame) -> pd.DataFrame:
    """清洗长表 -> 阶段观测记录帧（打 stage_key）。"""
    out = pd.DataFrame({
        "county": county,
        "year": pd.to_numeric(df["year"], errors="coerce").astype("Int64"),
        "date": pd.to_datetime(df["date"]),
        "stage_key": [classify_stage(m) for m in df["month_day"]],
    })
    for c in STAGE_WEATHER:
        out[c] = df[c] if c in df.columns else None
    return out.dropna(subset=["date"]).reset_index(drop=True)


def upsert(county: str, year: int, df: pd.DataFrame) -> tuple[int, str]:
    """累积写入；返回 (行数, 存储来源 mysql/csv/empty)。"""
    if df is None or df.empty:
        return 0, "empty"
    rec = _frame(county, year, df)
    if rec.empty:
        return 0, "empty"

    if available():
        try:
            n = repo.upsert_stage(rec)
            return n, "mysql"
        except Exception:
            pass  # 表缺失 / 连接异常 -> 落 CSV 回退
    _csv_upsert(rec)
    return len(rec), "csv"


def load(county: str, year: int) -> pd.DataFrame:
    """读取某县某年累积观测；MySQL 优先，失败回退 CSV。"""
    if available():
        try:
            df = repo.stage_observations(county, year)
            if not df.empty:
                return df
        except Exception:
            pass
    return _csv_load(county, year)


def reset(county: str, year: int) -> int:
    """清空某县某年累积观测（MySQL + CSV 双清），返回 MySQL 删除行数。"""
    n = 0
    if available():
        try:
            n = repo.delete_stage(county, year)
        except Exception:
            n = 0
    _csv_reset(county, year)
    return n


# ------------------------- CSV 回退 -------------------------

def _read_csv() -> pd.DataFrame:
    cols = ["county", "year", "date", "stage_key"] + STAGE_WEATHER
    if CSV_PATH.exists():
        try:
            return pd.read_csv(CSV_PATH, parse_dates=["date"])
        except Exception:
            return pd.DataFrame(columns=cols)
    return pd.DataFrame(columns=cols)


def _csv_upsert(rec: pd.DataFrame) -> None:
    old = _read_csv()
    merged = rec if old.empty else pd.concat([old, rec], ignore_index=True)
    merged = merged.drop_duplicates(subset=["county", "date"], keep="last")
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")


def _csv_load(county: str, year: int) -> pd.DataFrame:
    df = _read_csv()
    if df.empty:
        return df
    df = df[(df["county"] == county) & (df["year"] == year)].copy()
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    df["month_day"] = df["date"].dt.strftime("%m-%d")
    return df.sort_values("date").reset_index(drop=True)


def _csv_reset(county: str, year: int) -> None:
    df = _read_csv()
    if df.empty:
        return
    df = df[~((df["county"] == county) & (df["year"] == year))]
    df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
