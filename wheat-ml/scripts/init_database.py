"""建库建表，把 CSV 那套成果全量灌进 MySQL（默认 127.0.0.1:3306 / wheat_ml）。

四个表：counties 县和所属市；weather_daily 逐日气象八参数；feature_table
年度生育期特征；yield_labels 分县分年产量（真实优先，没有才用演示值）。

写入走分批 executemany，不用 LOAD DATA LOCAL INFILE。这样就不必去开
local_infile，也不吃 secure_file_priv 那个目录限制。

    cd wheat-ml
    python -m scripts.init_database
"""

import sys
import yaml
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.db.conn import connect, DB          # noqa: E402
from src.db.schema import DDL, FEATURE_COLS  # noqa: E402

with open(PROJECT_ROOT / "config" / "config.yaml", "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

PROCESSED = PROJECT_ROOT / CONFIG["paths"]["processed"]
WEATHER_COLS = ["t2m_max", "t2m_min", "t2m", "precip", "wind", "radiation", "dew", "rh"]
BATCH = 2000


def region_type(name: str) -> str:
    if name.endswith("区"):
        return "区"
    if name.endswith("市"):
        return "市"
    if name.endswith("县"):
        return "县"
    return "其他"


def _clean(v):
    """NaN/NaT 转 None，便于写入 SQL NULL。"""
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return v


def bulk_insert(conn, table: str, columns: list[str], rows: list) -> int:
    """分批 executemany 写入；rows 为已对齐 columns 的序列。"""
    if not rows:
        return 0
    cols = ", ".join(columns)
    placeholders = ", ".join(["%s"] * len(columns))
    sql = f"INSERT INTO {table} ({cols}) VALUES ({placeholders})"
    total = 0
    with conn.cursor() as cur:
        for i in range(0, len(rows), BATCH):
            chunk = rows[i:i + BATCH]
            cur.executemany(sql, chunk)
            total += len(chunk)
    conn.commit()
    return total


def create_schema() -> None:
    conn = connect(with_database=False)
    with conn.cursor() as cur:
        cur.execute(f"CREATE DATABASE IF NOT EXISTS `{DB['name']}` "
                    "DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci")
    conn.close()

    conn = connect()
    with conn.cursor() as cur:
        cur.execute("SET FOREIGN_KEY_CHECKS=0")
        # 旧表结构可能与当前 schema 不一致，先删除再按 DDL 重建
        for table in ["weather_daily", "feature_table", "yield_labels", "counties"]:
            cur.execute(f"DROP TABLE IF EXISTS {table}")
        for ddl in DDL:
            cur.execute(ddl)
        cur.execute("SET FOREIGN_KEY_CHECKS=1")
    conn.close()


def import_counties() -> dict[str, int]:
    rows = []
    for city, mapping in CONFIG["counties"].items():
        for name in mapping.values():
            rows.append((name, city, region_type(name)))
    conn = connect()
    bulk_insert(conn, "counties", ["name", "city", "region_type"], rows)
    with conn.cursor() as cur:
        cur.execute("SELECT name, id FROM counties")
        ids = dict(cur.fetchall())
    conn.close()
    print(f"counties: {len(rows)} 个地区")
    return ids


def import_weather(ids: dict[str, int]) -> None:
    df = pd.read_csv(PROCESSED / "weather_clean.csv", parse_dates=["date"])
    df["county_id"] = df["county"].map(ids)
    df = df.dropna(subset=["county_id"])
    df["county_id"] = df["county_id"].astype(int)
    df["obs_date"] = df["date"].dt.strftime("%Y-%m-%d")
    cols = ["county_id", "obs_date"] + WEATHER_COLS
    rows = df[cols].where(pd.notna(df[cols]), None).values.tolist()
    conn = connect()
    n = bulk_insert(conn, "weather_daily", cols, rows)
    conn.close()
    print(f"weather_daily: {n} 行")


def import_features(ids: dict[str, int]) -> None:
    df = pd.read_csv(PROCESSED / "feature_table.csv")
    df["county_id"] = df["county"].map(ids)
    df = df.dropna(subset=["county_id"])
    df["county_id"] = df["county_id"].astype(int)
    cols = ["county_id", "year"] + FEATURE_COLS
    rows = df[cols].where(pd.notna(df[cols]), None).values.tolist()
    conn = connect()
    n = bulk_insert(conn, "feature_table", cols, rows)
    conn.close()
    print(f"feature_table: {n} 行")


def import_yields(ids: dict[str, int]) -> None:
    src, is_real = None, False
    for key, real in [("raw_yield", True), ("raw_yield_demo", False)]:
        p = PROJECT_ROOT / CONFIG["paths"][key]
        if p.exists():
            src, is_real = p, real
            break
    if src is None:
        print("yield_labels: 无产量标签文件，跳过")
        return

    y = pd.read_csv(src)
    y.columns = [c.strip().lower() for c in y.columns]
    ycol = "wheat_yield" if "wheat_yield" in y.columns else \
        [c for c in y.columns if "yield" in c][0]
    y["county_id"] = y["county"].map(ids)
    y = y.dropna(subset=["county_id"])
    y["county_id"] = y["county_id"].astype(int)
    y["source"] = "real" if is_real else "demo"
    out = y[["county_id", "year", ycol, "source"]].rename(columns={ycol: "wheat_yield"})
    rows = out.where(pd.notna(out), None).values.tolist()
    conn = connect()
    n = bulk_insert(conn, "yield_labels",
                     ["county_id", "year", "wheat_yield", "source"], rows)
    conn.close()
    print(f"yield_labels: {n} 行（{'真实产量' if is_real else '演示标签'}）")


def main() -> None:
    create_schema()
    ids = import_counties()
    import_weather(ids)
    import_features(ids)
    import_yields(ids)

    conn = connect()
    with conn.cursor() as cur:
        for t in ["counties", "weather_daily", "feature_table", "yield_labels"]:
            cur.execute(f"SELECT COUNT(*) FROM {t}")
            print(f"  校验 {t}: {cur.fetchone()[0]} 行")
    conn.close()
    print("数据库导入完成 ->", f"{DB['host']}:{DB['port']}/{DB['name']}")


if __name__ == "__main__":
    main()
