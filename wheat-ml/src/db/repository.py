"""应用层查询接口：县列表、单县逐日气象、年度特征、产量标签与库内统计"""

import time

import pandas as pd

from src.db.conn import connect, available  # noqa: F401
from src.db.schema import FEATURE_COLS  # noqa: F401

WEATHER_SQL = """
SELECT c.name AS county, c.city AS city, w.obs_date AS date,
       w.t2m_max, w.t2m_min, w.t2m, w.precip, w.wind, w.radiation, w.dew, w.rh
FROM weather_daily w
JOIN counties c ON c.id = w.county_id
WHERE c.name = %s
ORDER BY w.obs_date
"""


def _query(sql: str, params=None) -> pd.DataFrame:
    conn = connect()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params or ())
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description]
    finally:
        conn.close()
    return pd.DataFrame(rows, columns=cols)


def counties() -> list[str]:
    return _query("SELECT name FROM counties ORDER BY name")["name"].tolist()


def county_weather(county: str) -> pd.DataFrame:
    """单县逐日气象，字段与 CSV 链路一致"""
    df = _query(WEATHER_SQL, (county,))
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["month_day"] = df["date"].dt.strftime("%m-%d")
    df["doy"] = df["date"].dt.day_of_year
    return df.sort_values("date").reset_index(drop=True)


def yield_labels() -> pd.DataFrame:
    """全部产量标签，JOIN 县表返回 county / year / wheat_yield / source。"""
    return _query("""
        SELECT c.name AS county, y.year AS year,
               y.wheat_yield AS wheat_yield, y.source AS source
        FROM yield_labels y
        JOIN counties c ON c.id = y.county_id
        ORDER BY c.name, y.year
    """)


def feature_rows() -> pd.DataFrame:
    """年度特征表，JOIN 县表返回 county / year 及各生育期特征列。"""
    cols = ", ".join(f"f.{c}" for c in FEATURE_COLS)
    return _query(f"""
        SELECT c.name AS county, f.year AS year, {cols}
        FROM feature_table f
        JOIN counties c ON c.id = f.county_id
        ORDER BY c.name, f.year
    """)


_STATS_TTL = 120.0
_STATS_CACHE = {"value": None, "ts": 0.0}


def clear_stats_cache() -> None:
    """数据重建或重新入库后主动失效概览统计缓存。"""
    _STATS_CACHE["value"] = None
    _STATS_CACHE["ts"] = 0.0


def stats(force: bool = False) -> dict:
    """库表规模统计。

    weather_daily 约 82 万行，InnoDB 下 COUNT(*) 需全表扫描（实测约 0.9 秒），
    首页每次渲染都查会明显拖慢首屏，故缓存 120 秒；数据重建后可调用
    clear_stats_cache() 或 stats(force=True) 立即刷新。
    """
    now = time.time()
    if not force and _STATS_CACHE["value"] is not None and now - _STATS_CACHE["ts"] < _STATS_TTL:
        return _STATS_CACHE["value"]

    df = _query("""
        SELECT (SELECT COUNT(*) FROM weather_daily) AS weather_rows,
               (SELECT COUNT(*) FROM counties) AS counties,
               (SELECT MIN(obs_date) FROM weather_daily) AS dmin,
               (SELECT MAX(obs_date) FROM weather_daily) AS dmax,
               (SELECT COUNT(*) FROM feature_table) AS feature_rows,
               (SELECT COUNT(*) FROM yield_labels) AS yield_rows
    """)
    r = df.iloc[0]
    result = {
        "weather_rows": int(r["weather_rows"]),
        "counties": int(r["counties"]),
        "year_min": int(pd.Timestamp(r["dmin"]).year) if r["dmin"] else 2010,
        "year_max": int(pd.Timestamp(r["dmax"]).year) if r["dmax"] else 2024,
        "feature_rows": int(r["feature_rows"]),
        "yield_rows": int(r["yield_rows"]),
    }
    _STATS_CACHE["value"] = result
    _STATS_CACHE["ts"] = now
    return result

# ---- 分阶段农事观测（运行时累积，MySQL 为主、CSV 回退由 stage_store 处理）----

STAGE_WEATHER = ["t2m_max", "t2m_min", "t2m", "precip",
                 "wind", "radiation", "dew", "rh"]


def _clean(v):
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return v


def stage_observations(county: str, year: int) -> pd.DataFrame:
    """某县某年累积的阶段观测逐日记录。"""
    df = _query("""
        SELECT county_name AS county, year, obs_date AS date, stage_key,
               t2m_max, t2m_min, t2m, precip, wind, radiation, dew, rh
        FROM stage_observations
        WHERE county_name = %s AND year = %s
        ORDER BY obs_date
    """, (county, year))
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["month_day"] = df["date"].dt.strftime("%m-%d")
    return df.reset_index(drop=True)


def upsert_stage(df: pd.DataFrame) -> int:
    """按 (县区, 日期) upsert；重复日期覆盖更新。返回写入行数。"""
    if df.empty:
        return 0
    conn = connect()
    cols = ["county_name", "year", "obs_date", "stage_key"] + STAGE_WEATHER + ["source"]
    placeholders = ", ".join(["%s"] * len(cols))
    update_cols = ["year", "stage_key"] + STAGE_WEATHER + ["source"]
    update = ", ".join(f"{c}=VALUES({c})" for c in update_cols)
    sql = (f"INSERT INTO stage_observations ({', '.join(cols)}) "
           f"VALUES ({placeholders}) ON DUPLICATE KEY UPDATE {update}")
    rows = []
    for _, r in df.iterrows():
        obs = r["date"].strftime("%Y-%m-%d") if hasattr(r["date"], "strftime") else str(r["date"])
        rows.append((
            r["county"], int(r["year"]), obs, r["stage_key"],
            *[_clean(r.get(c)) for c in STAGE_WEATHER], "stage_upload",
        ))
    with conn.cursor() as cur:
        for i in range(0, len(rows), 2000):
            cur.executemany(sql, rows[i:i + 2000])
    conn.commit()
    conn.close()
    return len(rows)


def delete_stage(county: str, year: int, stage_key: str | None = None) -> int:
    """删除某县某年阶段观测，返回删除行数；给出 stage_key 时只删该生育期。"""
    conn = connect()
    with conn.cursor() as cur:
        if stage_key:
            cur.execute(
                "DELETE FROM stage_observations "
                "WHERE county_name = %s AND year = %s AND stage_key = %s",
                (county, year, stage_key))
        else:
            cur.execute(
                "DELETE FROM stage_observations WHERE county_name = %s AND year = %s",
                (county, year))
        n = cur.rowcount
    conn.commit()
    conn.close()
    return n


def ensure_check_table() -> None:
    """确保作业卡勾选表存在（首次使用时建表）。"""
    conn = connect()
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS advice_checks (
                county VARCHAR(64) NOT NULL,
                year SMALLINT NOT NULL,
                stage_key VARCHAR(32) NOT NULL,
                card_key VARCHAR(64) NOT NULL,
                done TINYINT NOT NULL DEFAULT 0,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                            ON UPDATE CURRENT_TIMESTAMP,
                PRIMARY KEY (county, year, stage_key, card_key)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
    conn.commit()
    conn.close()


def advice_checks(county: str, year: int) -> pd.DataFrame:
    """某县某年的作业卡勾选记录。"""
    return _query("""
        SELECT county, year, stage_key, card_key, done, updated_at
        FROM advice_checks
        WHERE county = %s AND year = %s
    """, (county, int(year)))


def upsert_check(row: dict) -> None:
    """按四元组 upsert 勾选状态。"""
    conn = connect()
    sql = ("INSERT INTO advice_checks (county, year, stage_key, card_key, done) "
           "VALUES (%s, %s, %s, %s, %s) "
           "ON DUPLICATE KEY UPDATE done = VALUES(done), updated_at = CURRENT_TIMESTAMP")
    with conn.cursor() as cur:
        cur.execute(sql, (row["county"], int(row["year"]),
                          row["stage_key"], row["card_key"], int(row["done"])))
    conn.commit()
    conn.close()

