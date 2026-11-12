"""
数据上传与校验模块（对齐项目方案 2.1 / 第三章）
=================================================
职责：
  1. 读取 CSV / Excel（自动识别 UTF-8 / GBK 编码，兼容 NASA POWER 注释头）
  2. 中英文字段名自动映射到标准字段
  3. 检查缺失值、重复记录、异常值（最高温<最低温、湿度超范围等）
  4. 校验「日」字段口径：YYYY-MM-DD 或年内日序，不可混用
  5. 校验短波辐射单位（W/m² → MJ/m²/day 自动提示换算）
  6. 输出结构化校验报告 + 清洗后的长表，供分析与建模使用

对外接口：
  validate_file(path) -> dict     # {"ok":bool, "issues":[...], "preview":[...], ...}
  load_weather_file(path) -> (df, report)   # 已清洗的 DataFrame + 校验报告
"""

import re
import io
import yaml
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
with open(PROJECT_ROOT / "config" / "config.yaml", "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

UP = CONFIG["upload"]

# 标准字段 -> 可识别的列名别名（已做归一化处理，均为小写）
FIELD_ALIASES = {
    "year":      ["年份", "年", "year", "yr"],
    "doy":       ["日", "日序", "年日序", "day", "doy", "dayofyear", "day_of_year"],
    "date":      ["日期", "date", "time", "datetime", "记录日期"],
    "t2m_max":   ["最高温", "最高温度", "最高气温", "日最高温", "日最高气温",
                  "t2m_max", "tmax", "max_temp", "tempmax"],
    "t2m_min":   ["最低温", "最低温度", "最低气温", "日最低温", "日最低气温",
                  "t2m_min", "tmin", "min_temp", "tempmin"],
    "t2m":       ["平均温", "平均温度", "平均气温", "日均温", "气温",
                  "t2m", "tavg", "temp", "mean_temp"],
    "precip":    ["日降水量", "降水量", "降水", "降雨量",
                  "prectotcorr", "precip", "prcp", "rain"],
    "wind":      ["2m风速", "风速", "ws2m", "wind", "wind_speed", "ws"],
    "radiation": ["地表短波辐射", "短波辐射", "辐射", "太阳辐射",
                  "allsky_sfc_sw_dwn", "radiation", "solar", "sw_dwn"],
    "dew":       ["露点温度", "露点", "t2mdew", "dew", "dew_point", "td2m"],
    "rh":        ["相对湿度", "湿度", "rh2m", "rh", "humidity", "relative_humidity"],
}

REQUIRED_FIELDS = ["t2m_max", "t2m_min", "t2m", "precip"]

FIELD_CN = {
    "year": "年份", "doy": "日", "date": "日期", "t2m_max": "最高温",
    "t2m_min": "最低温", "t2m": "平均温", "precip": "日降水量", "wind": "2m风速",
    "radiation": "地表短波辐射", "dew": "露点温度", "rh": "相对湿度",
}

# 视为缺失的标记
MISSING_TOKENS = {"", "na", "nan", "null", "none", "-999", "-9999", "999", "—", "-"}


def _norm_col(col) -> str:
    """列名归一化：去 BOM、全角括号转半角、去掉单位后缀、转小写"""
    s = str(col).strip().replace("\ufeff", "").replace("（", "(").replace("）", ")")
    s = re.sub(r"\([^)]*\)", "", s)          # 去掉 (℃) (mm) 等单位
    s = re.sub(r"[℃%·\s]", "", s)
    return s.lower()


def map_columns(raw_cols) -> tuple[dict, list]:
    """原始列名 -> 标准字段名；返回 (映射表, 未识别列)"""
    mapping, unknown = {}, []
    for raw in raw_cols:
        n = _norm_col(raw)
        hit = None
        for field, aliases in FIELD_ALIASES.items():
            if n == _norm_col(field) or n in [_norm_col(a) for a in aliases]:
                hit = field
                break
        if hit is None:                       # 退化为包含匹配
            for field, aliases in FIELD_ALIASES.items():
                if any(_norm_col(a) in n for a in aliases if len(_norm_col(a)) >= 2):
                    hit = field
                    break
        if hit and hit not in mapping.values():
            mapping[raw] = hit
        elif hit:
            unknown.append(f"{raw}（与已有 {FIELD_CN[hit]} 重复）")
        else:
            unknown.append(str(raw))
    return mapping, unknown


def _decode_text(raw: bytes) -> str:
    for enc in ("utf-8-sig", "utf-8", "gbk", "gb18030"):
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    raise ValueError("无法识别文件编码，请另存为 UTF-8 编码的 CSV")


def _read_csv(path: Path) -> pd.DataFrame:
    """读取 CSV：自动跳 NASA POWER 注释头（前 N 行 + 第 N+1 行列名）"""
    text = _decode_text(path.read_bytes())
    lines = text.splitlines()
    header_idx = 0
    for i, line in enumerate(lines[:40]):
        low = line.lower()
        if ("year" in low and "doy" in low) or ("年份" in line and ("日" in line or "日序" in line)):
            header_idx = i
            break
    return pd.read_csv(io.StringIO("\n".join(lines[header_idx:])))


def _read_excel(path: Path) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name=0)


def load_raw(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext == ".csv":
        return _read_csv(path)
    if ext in (".xlsx", ".xls"):
        return _read_excel(path)
    raise ValueError(f"不支持的文件类型: {ext}")


def _to_numeric(series: pd.Series) -> pd.Series:
    """字符串缺失标记 -> NaN，再转数值"""
    if series.dtype == object:
        s = series.astype(str).str.strip().str.lower()
        series = series.where(~s.isin(MISSING_TOKENS))
    return pd.to_numeric(series, errors="coerce")


def _build_dates(df: pd.DataFrame, issues: list) -> pd.DataFrame:
    """
    统一「日」字段口径：
      - 有 date 列 -> 用完整日期
      - 只有 doy   -> 用 年份 + 年内日序
      - 两者都有   -> 报错（不可混用）
    """
    has_date, has_doy = "date" in df.columns, "doy" in df.columns

    if has_date and has_doy:
        issues.append({"level": "error", "field": "日",
                       "title": "日期口径混用",
                       "detail": "同时存在「日期」与「日序」两列，请只保留一种口径。"})
        has_doy = False

    if has_date:
        d = pd.to_datetime(df["date"], errors="coerce")
        bad = int(d.isna().sum())
        if bad:
            issues.append({"level": "error", "field": "日期",
                           "title": f"{bad} 行日期无法解析",
                           "detail": "需为 YYYY-MM-DD 格式。"})
        df["date"] = d
        if "year" in df.columns:
            mismatch = int((pd.to_numeric(df["year"], errors="coerce") != d.dt.year).sum())
            if mismatch:
                issues.append({"level": "warning", "field": "年份",
                               "title": f"{mismatch} 行年份与日期不一致",
                               "detail": "将以「日期」中的年份为准。"})
        df["year"] = d.dt.year
    elif has_doy:
        sample = df["doy"].dropna().astype(str).str.strip()
        date_like = (sample.str.match(r"^\d{4}[-/.]\d{1,2}[-/.]\d{1,2}$").mean()
                     if len(sample) else 0)
        if date_like > 0.8:
            # 「日」列写成了完整日期（YYYY-MM-DD），按日期口径处理
            d = pd.to_datetime(df["doy"], errors="coerce")
            bad = int(d.isna().sum())
            if bad:
                issues.append({"level": "error", "field": "日",
                               "title": f"{bad} 行日期无法解析",
                               "detail": "需为 YYYY-MM-DD 格式。"})
            df["date"] = d
            if "year" in df.columns:
                mismatch = int((pd.to_numeric(df["year"], errors="coerce") != d.dt.year).sum())
                if mismatch:
                    issues.append({"level": "warning", "field": "年份",
                                   "title": f"{mismatch} 行年份与日期不一致",
                                   "detail": "将以「日」中的日期年份为准。"})
            df["year"] = d.dt.year
        else:
            doy = pd.to_numeric(df["doy"], errors="coerce")
            if "year" not in df.columns:
                issues.append({"level": "error", "field": "年份",
                               "title": "缺少「年份」字段",
                               "detail": "使用年内日序时必须提供年份。"})
                df["date"] = pd.NaT
            else:
                year = pd.to_numeric(df["year"], errors="coerce")
                out_of_range = int(((doy < 1) | (doy > 366)).sum())
                if out_of_range:
                    issues.append({"level": "error", "field": "日",
                                   "title": f"{out_of_range} 行日序超出 1—366",
                                   "detail": "日序应为 1—365（闰年 366）。"})
                # 闰年 366 天检查：非闰年出现 366 视为异常
                leap = year.mod(4).eq(0) & (year.mod(100).ne(0) | year.mod(400).eq(0))
                bad366 = int(((doy == 366) & ~leap).sum())
                if bad366:
                    issues.append({"level": "error", "field": "日",
                                   "title": f"{bad366} 行非闰年出现日序 366",
                                   "detail": "请检查年份与日序是否对应。"})
                df["date"] = (pd.to_datetime(year.astype("Int64").astype(str), format="%Y",
                                             errors="coerce")
                              + pd.to_timedelta(doy - 1, unit="D"))
                df["year"] = df["date"].dt.year
    else:
        issues.append({"level": "error", "field": "日",
                       "title": "缺少日期字段",
                       "detail": "需要「日期(YYYY-MM-DD)」或「年份 + 日」两种口径之一。"})
        df["date"] = pd.NaT

    return df


def _check_constraints(df: pd.DataFrame, issues: list) -> pd.DataFrame:
    """缺失值 / 重复 / 异常值 / 单位检查"""
    n = len(df)

    # 1) 缺失值
    miss = df.isna().sum()
    miss = miss[miss > 0]
    for field, cnt in miss.items():
        cn = FIELD_CN.get(field, field)
        issues.append({
            "level": "warning" if cnt / max(n, 1) < 0.05 else "error",
            "field": cn,
            "title": f"{cn} 缺失 {cnt} 行（{cnt / n * 100:.1f}%）",
            "detail": "缺失值不会被当作 0，参与统计前会剔除或插值。",
        })

    # 2) 重复记录
    if "date" in df.columns and df["date"].notna().any():
        dup = int(df.duplicated(subset=["date"]).sum())
        if dup:
            issues.append({"level": "warning", "field": "日期",
                           "title": f"{dup} 条重复日期记录",
                           "detail": "将按日期去重，保留首条。"})
            df = df.drop_duplicates(subset=["date"], keep="first")

    # 3) 最高温 < 最低温
    if {"t2m_max", "t2m_min"} <= set(df.columns):
        bad = df["t2m_max"] < df["t2m_min"] - UP["tmax_tmin_tolerance"]
        bad = bad & df[["t2m_max", "t2m_min"]].notna().all(axis=1)
        if int(bad.sum()):
            issues.append({"level": "error", "field": "最高温/最低温",
                           "title": f"{int(bad.sum())} 行最高温低于最低温",
                           "detail": "请核对原始数据；这些行会在建模时剔除。"})
            df.loc[bad, ["t2m_max", "t2m_min"]] = np.nan

    # 4) 相对湿度合理范围
    if "rh" in df.columns:
        lo, hi = UP["rh_range"]
        out = df["rh"].notna() & ((df["rh"] < lo) | (df["rh"] > hi))
        if int(out.sum()):
            issues.append({"level": "error", "field": "相对湿度",
                           "title": f"{int(out.sum())} 行湿度超出 {lo}—{hi}%",
                           "detail": "超出物理合理范围，已置为缺失。"})
            df.loc[out, "rh"] = np.nan

    # 5) 降水负值
    if "precip" in df.columns:
        neg = df["precip"].notna() & (df["precip"] < 0)
        if int(neg.sum()):
            issues.append({"level": "error", "field": "日降水量",
                           "title": f"{int(neg.sum())} 行降水为负值",
                           "detail": "降水量不能为负，已置为缺失。"})
            df.loc[neg, "precip"] = np.nan

    # 6) 辐射单位：以中位数判断整列单位，避免个别异常值误触发整列换算
    if "radiation" in df.columns and df["radiation"].notna().any():
        factor = UP["radiation_unit_factor"]
        mid = float(df["radiation"].median())
        if mid > UP["radiation_unit_max"]:
            issues.append({
                "level": "warning", "field": "地表短波辐射",
                "title": f"辐射中位数 {mid:.1f}，判断单位为 W/m²",
                "detail": f"已按 ×{factor} 换算为 MJ/m²/day（换算后中位数 "
                          f"{mid * factor:.2f}）。",
            })
            df["radiation"] = df["radiation"] * factor
        else:
            odd = df["radiation"].notna() & (df["radiation"] > UP["radiation_unit_max"])
            if int(odd.sum()):
                issues.append({
                    "level": "warning", "field": "地表短波辐射",
                    "title": f"{int(odd.sum())} 行辐射值超过 {UP['radiation_unit_max']} MJ/m²/day",
                    "detail": "与全列单位不一致，可能是个别行混入了 W/m² 数值，请核对。",
                })
                df.loc[odd, "radiation"] = np.nan

    # 7) 年份范围
    if "year" in df.columns and df["year"].notna().any():
        lo_y, hi_y = UP["year_min"], UP["year_max"]
        out_y = df["year"].notna() & ((df["year"] < lo_y) | (df["year"] > hi_y))
        if int(out_y.sum()):
            issues.append({"level": "warning", "field": "年份",
                           "title": f"{int(out_y.sum())} 行年份不在 {lo_y}—{hi_y}",
                           "detail": "方案要求 2010—2024 年历史气象数据。"})

    # 8) 闰年 2 月 29 日
    if "date" in df.columns and df["date"].notna().any():
        leap_years = [y for y in range(UP["year_min"], UP["year_max"] + 1)
                      if (y % 4 == 0 and y % 100 != 0) or y % 400 == 0]
        present = set(df.loc[df["date"].dt.strftime("%m-%d") == "02-29", "year"].tolist())
        missing_leap = [y for y in leap_years if y in set(df["year"].unique()) and y not in present]
        if missing_leap:
            issues.append({"level": "warning", "field": "日期",
                           "title": f"闰年缺少 2 月 29 日：{missing_leap}",
                           "detail": "闰年应保留 2 月 29 日记录，否则年度特征会偏差。"})

    # 9) 单年记录完整度
    if "year" in df.columns and df["year"].notna().any():
        counts = df.groupby("year").size()
        thin = counts[counts < UP["min_days_per_year"]]
        if len(thin):
            issues.append({"level": "warning", "field": "记录完整度",
                           "title": f"{len(thin)} 个年份记录不足 "
                                    f"{UP['min_days_per_year']} 天",
                           "detail": "、".join(f"{int(y)}年{int(c)}天"
                                              for y, c in thin.items())})

    return df


def _summary(df: pd.DataFrame) -> dict:
    years = sorted(int(y) for y in df["year"].dropna().unique()) if "year" in df.columns else []
    return {
        "rows": int(len(df)),
        "cols": int(df.shape[1]),
        "years": years,
        "year_range": f"{years[0]}—{years[-1]}" if years else "—",
        "date_range": (f"{df['date'].min():%Y-%m-%d} ~ {df['date'].max():%Y-%m-%d}"
                       if "date" in df.columns and df["date"].notna().any() else "—"),
        "counties": (sorted(df["county"].dropna().unique().tolist())
                     if "county" in df.columns else []),
    }


def load_weather_file(path: str | Path, county: str | None = None) -> tuple[pd.DataFrame, dict]:
    """
    读取并清洗任意上传的气象文件
    Returns: (清洗后的长表, 校验报告)
    """
    path = Path(path)
    report = {"filename": path.name, "ok": True, "issues": [], "summary": {},
              "columns": {}, "preview": []}

    raw = load_raw(path)
    mapping, unknown = map_columns(raw.columns)
    report["columns"] = {str(k): FIELD_CN.get(v, v) for k, v in mapping.items()}
    if unknown:
        report["issues"].append({
            "level": "warning", "field": "列名",
            "title": f"{len(unknown)} 个列未能识别",
            "detail": "、".join(unknown[:8]) + ("…" if len(unknown) > 8 else ""),
        })

    df = raw.rename(columns=mapping)

    # 「日」列写成完整日期时转为 date 口径，避免后续数值化把日期清空
    if "doy" in df.columns and "date" not in df.columns:
        sample = df["doy"].dropna().astype(str).str.strip()
        if len(sample) and sample.str.match(r"^\d{4}[-/.]\d{1,2}[-/.]\d{1,2}$").mean() > 0.8:
            df = df.rename(columns={"doy": "date"})

    # 数值化
    for f in ["year", "doy", "t2m_max", "t2m_min", "t2m", "precip",
              "wind", "radiation", "dew", "rh"]:
        if f in df.columns:
            df[f] = _to_numeric(df[f])

    missing_req = [FIELD_CN[f] for f in REQUIRED_FIELDS if f not in df.columns]
    if missing_req:
        report["ok"] = False
        report["issues"].append({
            "level": "error", "field": "必需字段",
            "title": f"缺少必需字段：{'、'.join(missing_req)}",
            "detail": "方案表 2 要求至少包含年份、日、最高温、最低温、平均温、日降水量。",
        })

    df = _build_dates(df, report["issues"])
    df = _check_constraints(df, report["issues"])

    # 县级标识：上传文件未带县名时用参数补齐
    if "county" not in df.columns:
        df["county"] = county or "未标注"
    elif county:
        df["county"] = county

    df["month"] = df["date"].dt.month
    df["month_day"] = df["date"].dt.strftime("%m-%d")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    if any(i["level"] == "error" for i in report["issues"]):
        report["ok"] = False
    report["summary"] = _summary(df)

    preview_cols = ["date", "year"] + [f for f in
                   ["t2m_max", "t2m_min", "t2m", "precip", "wind", "radiation", "dew", "rh"]
                   if f in df.columns]
    preview = df[preview_cols].head(10).copy()
    preview["date"] = preview["date"].dt.strftime("%Y-%m-%d")
    report["preview"] = preview.round(2).to_dict("records")

    return df, report


def validate_file(path: str | Path) -> dict:
    """只跑校验，返回报告（供 Web 上传页直接使用）"""
    ext = Path(path).suffix.lower()
    if ext not in UP["allowed_ext"]:
        return {"filename": Path(path).name, "ok": False,
                "issues": [{"level": "error", "field": "文件类型",
                            "title": f"不支持的文件类型 {ext}",
                            "detail": f"仅支持 {'、'.join(UP['allowed_ext'])}"}],
                "summary": {}, "columns": {}, "preview": []}
    try:
        _, report = load_weather_file(path)
        return report
    except Exception as e:                      # 文件损坏 / 编码异常
        return {"filename": Path(path).name, "ok": False,
                "issues": [{"level": "error", "field": "文件读取",
                            "title": "文件读取失败", "detail": str(e)}],
                "summary": {}, "columns": {}, "preview": []}


if __name__ == "__main__":
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else "data/raw/weather/郑州市/登封气象.csv"
    rep = validate_file(target)
    print(f"文件: {rep['filename']}  通过: {rep['ok']}")
    print("摘要:", rep["summary"])
    for i in rep["issues"]:
        print(f"  [{i['level']}] {i['title']} — {i['detail']}")
    print("预览:")
    for row in rep["preview"][:3]:
        print("  ", row)
