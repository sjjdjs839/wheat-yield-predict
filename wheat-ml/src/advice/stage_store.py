"""分阶段农事观测的累积存储。

阶段就按日期的「月-日」落到返青、拔节、抽穗扬花、灌浆、成熟这几段里。
落不进去的记 offseason——越冬或者收完了，只留个备注，不参与阶段建议。

累积的口径：县区 + 日期去重，同一天重传就覆盖，不同天多次传就累加。

存储先走 MySQL，连不上、表没了、写入报错都退回本地 CSV。
"""

import json
import yaml
import pandas as pd
from datetime import datetime
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
    _safe_snapshot(county, year, "all")
    n = 0
    if available():
        try:
            n = repo.delete_stage(county, year)
        except Exception:
            n = 0
    _csv_reset(county, year)
    return n


# --- CSV 回退 ---

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

def stage_counts(county: str, year: int) -> dict[str, int]:
    """各生育期已累积天数（键为 stage_key；不含 offseason）。"""
    df = load(county, year)
    if df.empty or "stage_key" not in df.columns:
        return {}
    return {str(k): int(v) for k, v in df.groupby("stage_key").size().items()}


def reset_stage(county: str, year: int, stage_key: str) -> int:
    """只清空某一生育期的观测，其余阶段保留。"""
    _safe_snapshot(county, year, stage_key)
    n = 0
    if available():
        try:
            n = repo.delete_stage(county, year, stage_key=stage_key)
        except Exception:
            n = 0
    _csv_reset_stage(county, year, stage_key)
    return n


def _csv_reset_stage(county: str, year: int, stage_key: str) -> None:
    df = _read_csv()
    if df.empty:
        return
    keep = ~((df["county"] == county) & (df["year"] == year) & (df["stage_key"] == stage_key))
    df[keep].to_csv(CSV_PATH, index=False, encoding="utf-8-sig")


# --- 清空回收站 ---
# 清空属于不可逆操作，演示时误点代价高。这里在删除前把待删记录快照一份，
# 并记录"最近一次清空"，页面上可一键撤销。

RECYCLE_DIR = PROJECT_ROOT / "web" / "uploads" / "_recycle"
LAST_PATH = RECYCLE_DIR / "last.json"
SNAP_COLS = ["county", "year", "date", "stage_key"] + list(STAGE_WEATHER)


def _stage_label(tag: str) -> str:
    if tag == "all":
        return "全部生育期"
    for st in STAGES:
        if st["key"] == tag:
            return st["name"]
    return tag


def _safe_snapshot(county: str, year: int, tag: str):
    """删除前快照待删记录；无数据时返回 None，不阻断清空。"""
    try:
        df = load(county, year)
    except Exception:
        return None
    if df is None or df.empty:
        return None
    if tag and tag != "all":
        df = df[df["stage_key"] == tag]
        if df.empty:
            return None
    keep = [c for c in SNAP_COLS if c in df.columns]
    snap = df[keep].copy()
    snap["date"] = pd.to_datetime(snap["date"]).dt.strftime("%Y-%m-%d")
    RECYCLE_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RECYCLE_DIR / (county + "_" + str(year) + "_" + (tag or "all") + "_" + ts + ".csv")
    snap.to_csv(path, index=False, encoding="utf-8-sig")
    info = {
        "county": county, "year": int(year), "stage": tag or "all",
        "label": _stage_label(tag or "all"), "rows": int(len(snap)),
        "path": str(path), "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    LAST_PATH.write_text(json.dumps(info, ensure_ascii=False), encoding="utf-8")
    return info


def last_snapshot():
    """最近一次清空的备份信息；备份文件不在则返回 None。"""
    try:
        if not LAST_PATH.exists():
            return None
        info = json.loads(LAST_PATH.read_text(encoding="utf-8"))
        if not Path(info.get("path", "")).exists():
            return None
        return info
    except Exception:
        return None


def undo_last():
    """把最近一次清空的观测写回；返回备份信息，失败返回 None。"""
    info = last_snapshot()
    if not info:
        return None
    try:
        df = pd.read_csv(info["path"])
        df["date"] = pd.to_datetime(df["date"])
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
        df["stage_key"] = [classify_stage(pd.Timestamp(d).strftime("%m-%d")) for d in df["date"]]
        for c in STAGE_WEATHER:
            if c not in df.columns:
                df[c] = None
        if available():
            try:
                repo.upsert_stage(df)
            except Exception:
                _csv_upsert(df)
        else:
            _csv_upsert(df)
        LAST_PATH.unlink(missing_ok=True)
        return info
    except Exception:
        return None
