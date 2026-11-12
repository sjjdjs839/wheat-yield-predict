"""
数据清洗模块（对齐项目方案第三章）
====================================
职责：
  1. 按 config 白名单遍历 data/raw/weather/{城市}/{地区}.csv
  2. 兼容中文表头（年份 + 日序）与 NASA POWER 英文表头两种格式
  3. 统一字段名、日期、缺失与异常处理（复用 src.data.validate）
  4. 输出清洗后的长表 data/processed/weather_clean.csv
  5. 输出数据审计报告 data/processed/data_audit.md

输入：data/raw/weather/{城市}/{地区}.csv
输出：data/processed/weather_clean.csv
      （county, city, date, year, month, month_day, doy + 8 个标准气象字段）
"""

import yaml
import pandas as pd
from pathlib import Path

from src.data.validate import load_weather_file

PROJECT_ROOT = Path(__file__).resolve().parents[2]
with open(PROJECT_ROOT / "config" / "config.yaml", "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

WEATHER_DIR = PROJECT_ROOT / CONFIG["paths"]["raw_weather"]
PROCESSED_DIR = PROJECT_ROOT / CONFIG["paths"]["processed"]

PARAMS = ["t2m_max", "t2m_min", "t2m", "precip", "wind", "radiation", "dew", "rh"]


def resolve_county_files() -> tuple[list[dict], list[str]]:
    """按 config 白名单解析 {城市}/{文件名} -> 标准地名"""
    found, missing = [], []
    for city, mapping in CONFIG["counties"].items():
        city_dir = WEATHER_DIR / city
        for fname, county in mapping.items():
            name = fname if fname.endswith(".csv") else f"{fname}.csv"
            path = city_dir / name
            if path.exists():
                found.append({"city": city, "filename": name,
                              "filepath": path, "county": county})
            else:
                missing.append(f"{city}/{name} -> {county}")
    return found, missing


def run_clean() -> pd.DataFrame:
    """清洗主入口：全部白名单地区合并为一张长表"""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    found, missing = resolve_county_files()
    total = sum(len(v) for v in CONFIG["counties"].values())

    print(f"白名单命中: {len(found)} / {total}")
    if missing:
        print("缺失文件:")
        for m in missing:
            print("  -", m)

    frames, stats = [], []
    for item in found:
        df, report = load_weather_file(item["filepath"], item["county"])
        df["city"] = item["city"]
        df["doy"] = df["date"].dt.day_of_year
        cols = ["county", "city", "date", "year", "month", "month_day", "doy"] + PARAMS
        df = df.reindex(columns=cols)
        frames.append(df)
        stats.append({
            "county": item["county"], "city": item["city"], "rows": len(df),
            "missing": int(df[PARAMS].isna().sum().sum()),
            "issues": len(report["issues"]),
            "start": df["date"].min().date(), "end": df["date"].max().date(),
        })

    if not frames:
        raise FileNotFoundError(f"未在 {WEATHER_DIR} 找到任何白名单气象文件")

    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df.sort_values(["county", "date"]).reset_index(drop=True)
    out = PROCESSED_DIR / "weather_clean.csv"
    all_df.to_csv(out, index=False, encoding="utf-8")

    stat_df = pd.DataFrame(stats)
    print(f"\n清洗完成: {len(stat_df)} 个地区, {len(all_df)} 行 -> {out}")
    print(f"缺失值总数: {int(stat_df['missing'].sum())}")
    print(f"日期范围: {all_df['date'].min().date()} ~ {all_df['date'].max().date()}")

    write_audit(stat_df, all_df, missing, total)
    return all_df


def write_audit(stat_df: pd.DataFrame, all_df: pd.DataFrame,
                missing: list[str], total: int) -> None:
    """输出 markdown 审计报告"""
    lines = ["# 数据审计报告", ""]
    lines.append(f"- 白名单地区数: {len(stat_df)} / {total}")
    lines.append(f"- 缺失文件: {len(missing)}")
    if missing:
        lines += [f"  - {m}" for m in missing]
    lines.append(f"- 总行数: {len(all_df)}")
    lines.append(f"- 日期范围: {all_df['date'].min().date()} ~ {all_df['date'].max().date()}")
    lines.append(f"- 缺失值总数: {int(stat_df['missing'].sum())}")
    lines.append("")
    lines.append("| 城市 | 地区 | 行数 | 缺失值 | 校验提示 | 起始 | 结束 |")
    lines.append("|---|---|---|---|---|---|---|")
    for _, r in stat_df.sort_values(["city", "county"]).iterrows():
        lines.append(f"| {r['city']} | {r['county']} | {r['rows']} | "
                     f"{r['missing']} | {r['issues']} | {r['start']} | {r['end']} |")

    p = PROCESSED_DIR / "data_audit.md"
    p.write_text("\n".join(lines), encoding="utf-8")
    print(f"审计报告 -> {p}")


if __name__ == "__main__":
    run_clean()
