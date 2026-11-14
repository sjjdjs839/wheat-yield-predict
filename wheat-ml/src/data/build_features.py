"""
特征工程模块（对齐任务书步骤 6 + 表 4 的 20 个特征）
====================================================
输入：data/processed/weather_clean.csv（clean.py 产出）
      data/raw/county_yield.csv（产量标签，可能尚未交付）
输出：data/processed/feature_table.csv
      行为「县 × 年」，列为 20 特征 + wheat_yield（若产量已交付）
"""

import yaml
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
with open(PROJECT_ROOT / "config" / "config.yaml", "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

PROCESSED_DIR = PROJECT_ROOT / CONFIG["paths"]["processed"]
YIELD_PATH = PROJECT_ROOT / CONFIG["paths"]["raw_yield"]

FULL = CONFIG["stages"]["full_season"]     # [3,4,5,6]
S34 = CONFIG["stages"]["stage_34"]         # [3,4]
S56 = CONFIG["stages"]["stage_56"]         # [5,6]
BASE_T = CONFIG["gdd_base_temp"]           # 10

# 兼容标准小写字段名（上传校验 / Web 应用产出的数据）与 NASA POWER 原始列名
CANONICAL_TO_NASA = {
    "t2m_max": "T2M_MAX", "t2m_min": "T2M_MIN", "t2m": "T2M",
    "precip": "PRECTOTCORR", "wind": "WS2M",
    "radiation": "ALLSKY_SFC_SW_DWN", "dew": "T2MDEW", "rh": "RH2M",
}


def gdd(series: pd.Series) -> float:
    """生长度日：Σ max(T2M - 10, 0)"""
    return float(np.clip(series - BASE_T, 0, None).sum())


def build_county(df_county: pd.DataFrame) -> pd.DataFrame:
    """
    单县逐年计算 20 个特征

    口径说明：spi_12 采用「生育期降水距平的县内标准化」（z-score），
    与参考实现的 spi_12 = precip_anom 行为一致；这样 2010 年不会因缺
    上一年数据而丢样本。
    """
    recs = []

    # 空输入保护：缺记录或缺年份列时直接返回空表，避免下游取列崩溃
    if df_county is None or len(df_county) == 0 or "year" not in df_county.columns:
        return pd.DataFrame(columns=["county", "city", "year"])

    # 统一到 NASA POWER 列名，兼容传入标准字段名的数据
    rename = {k: v for k, v in CANONICAL_TO_NASA.items()
              if k in df_county.columns and v not in df_county.columns}
    if rename:
        df_county = df_county.rename(columns=rename)

    for year in sorted(df_county["year"].unique()):
        d = df_county[df_county["year"] == year]
        full = d[d["month"].isin(FULL)]
        d34 = d[d["month"].isin(S34)]
        d56 = d[d["month"].isin(S56)]

        if len(full) < 100:
            continue

        season_precip = float(full["PRECTOTCORR"].sum())

        rec = {
            "county": d["county"].iloc[0],
            "city": d["city"].iloc[0],
            "year": int(year),
            "year_index": int(year) - 2009,  # 2010=1, 2011=2, ... 品种改良趋势项
            "season_precip": season_precip,
            "hot_days": int((full["T2M_MAX"] >= 35).sum()),
            "cold_days": int((full["T2M_MIN"] < 0).sum()),
            "gdd": gdd(full["T2M"]),
            "season_tmax": float(full["T2M_MAX"].mean()),
            "season_tmin": float(full["T2M_MIN"].mean()),
            "season_wind": float(full["WS2M"].mean()),
            "season_sunshine": float(full["ALLSKY_SFC_SW_DWN"].mean()),
            "season_tmax_34": float(d34["T2M_MAX"].mean()),
            "season_precip_34": float(d34["PRECTOTCORR"].sum()),
            "gdd_34": gdd(d34["T2M"]),
            "cold_days_34": int((d34["T2M_MIN"] < 0).sum()),
            "season_tmax_56": float(d56["T2M_MAX"].mean()),
            "season_precip_56": float(d56["PRECTOTCORR"].sum()),
            "gdd_56": gdd(d56["T2M"]),
            "hot_days_56": int((d56["T2M_MAX"] >= 35).sum()),
            "hot_dry_days": int(((d56["T2M_MAX"] >= 35) & (d56["RH2M"] <= 40)).sum()),
        }
        recs.append(rec)

    out = pd.DataFrame(recs)
    if out.empty:
        return out

    # 距平类特征：以该县多年均值为常年值
    mean_p = out["season_precip"].mean()
    std_p = out["season_precip"].std()
    out["precip_anom"] = (out["season_precip"] - mean_p) / mean_p
    out["season_precip_norm"] = (out["season_precip"] - mean_p) / mean_p
    out["spi_12"] = (out["season_precip"] - mean_p) / std_p

    return out


def match_county(raw_name: str, canonicals: set[str]) -> str | None:
    """产量表县名 -> 标准县名（容忍 县/市/区 后缀）"""
    raw_name = str(raw_name).strip()
    if raw_name in canonicals:
        return raw_name
    for suffix in ("县", "市", "区"):
        if raw_name.endswith(suffix) and raw_name[:-1] in canonicals:
            return raw_name[:-1]
    for c in canonicals:
        if c in raw_name:
            return c
    return None


def attach_yield(feat: pd.DataFrame) -> pd.DataFrame:
    """左连接产量标签（若已交付）"""
    if not YIELD_PATH.exists():
        print(f"[待交付] 产量标签不存在: {YIELD_PATH}")
        print("           特征表先不含 wheat_yield 列，产量到位后重跑即可")
        return feat

    y = pd.read_csv(YIELD_PATH, encoding="utf-8")
    y.columns = [c.strip().lower() for c in y.columns]
    col = "wheat_yield" if "wheat_yield" in y.columns else \
          [c for c in y.columns if "yield" in c][0]

    canon = set(feat["county"].unique())
    y["county_std"] = y["county"].map(lambda x: match_county(x, canon))
    unmatched = y[y["county_std"].isna()]["county"].unique()
    if len(unmatched):
        print(f"[警告] 以下产量县名未能匹配: {list(unmatched)}")

    y = y.dropna(subset=["county_std"])
    merged = feat.merge(
        y[["county_std", "year", col]].rename(
            columns={"county_std": "county", col: "wheat_yield"}),
        on=["county", "year"], how="left"
    )
    hit = merged["wheat_yield"].notna().sum()
    print(f"产量标签匹配: {hit} / {len(merged)} 行")
    return merged


def run_build() -> pd.DataFrame:
    clean_path = PROCESSED_DIR / "weather_clean.csv"
    if not clean_path.exists():
        raise FileNotFoundError(f"缺少 {clean_path}，请先运行 src.data.clean")

    df = pd.read_csv(clean_path, parse_dates=["date"])

    frames = []
    for county, d in df.groupby("county"):
        frames.append(build_county(d))
    feat = pd.concat(frames, ignore_index=True)

    # 调整列顺序：标识列 + 20 特征 + 产量
    feats = CONFIG["feature_cols"]
    feat = feat[["county", "city", "year"] + feats]
    feat = attach_yield(feat)

    out = PROCESSED_DIR / "feature_table.csv"
    feat.to_csv(out, index=False, encoding="utf-8")
    print(f"\n特征表: {feat.shape[0]} 行 × {feat.shape[1]} 列 -> {out}")
    print(f"县数: {feat['county'].nunique()}, 年份: {feat['year'].min()}-{feat['year'].max()}")
    return feat


if __name__ == "__main__":
    run_build()
