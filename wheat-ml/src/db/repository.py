"""应用层查询接口：县列表、单县逐日气象、库内统计"""

import pandas as pd

from src.db.conn import connect, available  # noqa: F401  (available 供 app 层调用)

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


def stats() -> dict:
    df = _query("""
        SELECT (SELECT COUNT(*) FROM weather_daily) AS weather_rows,
               (SELECT COUNT(*) FROM counties) AS counties,
               (SELECT MIN(date) FROM weather_daily) AS dmin,
               (SELECT MAX(date) FROM weather_daily) AS dmax,
               (SELECT COUNT(*) FROM feature_table) AS feature_rows,
               (SELECT COUNT(*) FROM yield_labels) AS yield_rows
    """)
    r = df.iloc[0]
    return {
        "weather_rows": int(r["weather_rows"]),
        "counties": int(r["counties"]),
        "year_min": int(pd.Timestamp(r["dmin"]).year) if r["dmin"] else 2010,
        "year_max": int(pd.Timestamp(r["dmax"]).year) if r["dmax"] else 2024,
        "feature_rows": int(r["feature_rows"]),
        "yield_rows": int(r["yield_rows"]),
    }
