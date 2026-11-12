"""
MySQL 建库建表与全量导入
==========================
将 CSV 链路的成果写入项目内 MySQL（默认 127.0.0.1:3307 / wheat_ml）：
  counties      151 个县（市、区）
  weather_daily 827,329 条逐日气象
  feature_table 2,265 条年度特征
  yield_labels  2,265 条产量标签（真实优先，其次演示）

用法：python -m scripts.init_database
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
STAGING = Path(r"C:\wheat_mysql\staging")
if not STAGING.parent.exists():
    STAGING = PROCESSED / "_db_staging"


def region_type(name: str) -> str:
    if name.endswith("区"):
        return "区"
    if name.endswith("市"):
        return "市"
    if name.endswith("县"):
        return "县"
    return "其他"


def create_schema(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(f"CREATE DATABASE IF NOT EXISTS `{DB['name']}` "
                    "DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci")
    conn.close()

    conn = connect()
    with conn.cursor() as cur:
        for ddl in DDL:
            cur.execute(ddl)
        cur.execute("SET FOREIGN_KEY_CHECKS=0")
        for table in ["weather_daily", "feature_table", "yield_labels", "counties"]:
            cur.execute(f"TRUNCATE TABLE {table}")
        cur.execute("SET FOREIGN_KEY_CHECKS=1")
    conn.close()


def import_counties() -> dict[str, int]:
    rows = []
    for city, mapping in CONFIG["counties"].items():
        for name in mapping.values():
            rows.append((name, city, region_type(name)))
    conn = connect()
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO counties (name, city, region_type) VALUES (%s, %s, %s)", rows)
        cur.execute("SELECT name, id FROM counties")
        ids = dict(cur.fetchall())
    conn.close()
    print(f"counties: {len(rows)} 个地区")
    return ids


def load_table(conn, table: str, path: Path, columns: list[str]) -> None:
    col_sql = ", ".join(columns)
    sql = (f"LOAD DATA LOCAL INFILE %s INTO TABLE {table} "
           "CHARACTER SET utf8mb4 FIELDS TERMINATED BY ',' "
           f"LINES TERMINATED BY '\\n' ({col_sql})")
    with conn.cursor() as cur:
        cur.execute(sql, (str(path),))


def import_weather(ids: dict[str, int]) -> None:
    df = pd.read_csv(PROCESSED / "weather_clean.csv", parse_dates=["date"])
    df["county_id"] = df["county"].map(ids)
    df["obs_date"] = df["date"].dt.strftime("%Y-%m-%d")
    cols = ["county_id", "obs_date", "t2m_max", "t2m_min", "t2m",
            "precip", "wind", "radiation", "dew", "rh"]
    STAGING.mkdir(parents=True, exist_ok=True)
    p = STAGING / "weather_daily.csv"
    df[cols].to_csv(p, index=False, header=False, float_format="%.4f")
    conn = connect()
    load_table(conn, "weather_daily", p, cols)
    conn.close()
    print(f"weather_daily: {len(df)} 行")


def import_features(ids: dict[str, int]) -> None:
    df = pd.read_csv(PROCESSED / "feature_table.csv")
    df["county_id"] = df["county"].map(ids)
    cols = ["county_id", "year"] + FEATURE_COLS
    p = STAGING / "feature_table.csv"
    df[cols].to_csv(p, index=False, header=False, float_format="%.4f")
    conn = connect()
    load_table(conn, "feature_table", p, cols)
    conn.close()
    print(f"feature_table: {len(df)} 行")


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
    y["source"] = "real" if is_real else "demo"
    out = y[["county_id", "year", ycol, "source"]].rename(columns={ycol: "wheat_yield"})
    out["county_id"] = out["county_id"].astype(int)
    p = STAGING / "yield_labels.csv"
    out.to_csv(p, index=False, header=False, float_format="%.2f")
    conn = connect()
    load_table(conn, "yield_labels", p, ["county_id", "year", "wheat_yield", "source"])
    conn.close()
    print(f"yield_labels: {len(out)} 行（{'真实产量' if is_real else '演示标签'}）")


def main() -> None:
    conn = connect(with_database=False)
    create_schema(conn)
    ids = import_counties()

    conn = connect()
    import_weather(ids)
    import_features(ids)
    import_yields(ids)

    with conn.cursor() as cur:
        for t in ["counties", "weather_daily", "feature_table", "yield_labels"]:
            cur.execute(f"SELECT COUNT(*) FROM {t}")
            print(f"  校验 {t}: {cur.fetchone()[0]} 行")
    conn.close()
    print("数据库导入完成 ->", f"{DB['host']}:{DB['port']}/{DB['name']}")


if __name__ == "__main__":
    main()
