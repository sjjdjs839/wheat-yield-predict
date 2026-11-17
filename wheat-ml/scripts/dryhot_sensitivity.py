# -*- coding: utf-8 -*-
"""干热风判据敏感性：三种阈值口径 × 双向固定效应。

用法：在 wheat-ml 目录下执行
    python -m scripts.dryhot_sensitivity
输出：output/reports/dryhot_sensitivity.json
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from panel_impact import demean_two_way, cluster_ols, fmt_model  # noqa: E402

WEATHER = ROOT / "data/processed/weather_clean.csv"
FEAT = ROOT / "data/processed/feature_table.csv"
STATS = ROOT / "output/models/county_stats.json"
OUT = ROOT / "output/reports/dryhot_sensitivity.json"

# 三套判据：项目自定、国标轻干热风近似、国标重干热风近似
CRITERIA = {
    "自定义 35℃/40%（现行）": (35, 40),
    "国标轻干热风近似 32℃/30%": (32, 30),
    "国标重干热风近似 35℃/25%": (35, 25),
}


def build_may_days() -> pd.DataFrame:
    w = pd.read_csv(WEATHER, encoding="utf-8-sig",
                    usecols=["county", "year", "month", "t2m_max", "rh"])
    may = w[w["month"] == 5]
    keys = ["county", "year"]
    out = may.groupby(keys).size().rename("may_days").reset_index()
    for tt, rr in CRITERIA.values():
        col = "hd_%d_%d" % (tt, rr)
        hit = ((may["t2m_max"] >= tt) & (may["rh"] <= rr)).astype(int)
        s = may.assign(_hit=hit).groupby(keys)["_hit"].sum().rename(col).reset_index()
        out = out.merge(s, on=keys, how="left")
        out[col] = out[col].fillna(0).astype(int)
    return out


def main() -> None:
    base = build_may_days()
    ft = pd.read_csv(FEAT, encoding="utf-8-sig")
    raw = json.load(io.open(STATS, encoding="utf-8"))
    ft["base"] = ft["county"].map(lambda c: raw[c]["mean_yield"] if c in raw else None)
    df = ft[ft["wheat_yield"].notna() & ft["base"].notna()].copy()
    df["y_norm"] = df["wheat_yield"] / df["base"]
    df["county"] = df["county"].astype(str)
    df["year"] = df["year"].astype(int)
    df = df.merge(base, on=["county", "year"], how="left")

    cols = ["hd_%d_%d" % (t, r) for t, r in CRITERIA.values()]
    df = df.dropna(subset=["y_norm"] + cols)
    print("回归样本：%d 条 / %d 县 / %d 年"
          % (len(df), df["county"].nunique(), df["year"].nunique()))

    out = {}
    print("\n%-28s %-9s %-10s %-8s %-6s %-18s %s"
          % ("判据", "均值天数", "系数(%/天)", "p值", "n", "95%CI", "withinR2"))
    for name, (tt, rr) in CRITERIA.items():
        col = "hd_%d_%d" % (tt, rr)
        ent = df["county"].reset_index(drop=True)
        tim = df["year"].reset_index(drop=True)
        dm = demean_two_way(df[["y_norm", col]].reset_index(drop=True), ent, tim)
        res = cluster_ols(dm["y_norm"].values, dm[[col]].values, ent, [col])
        r = fmt_model(res)[0]
        out[name] = {
            "threshold": "%d\u2103/%d%%" % (tt, rr),
            "mean_days": round(float(df[col].mean()), 2),
            "beta_pct": round(float(r["系数"]) * 100, 2),
            "se_pct": round(float(r["标准误"]) * 100, 2),
            "p": float(r["p值"]),
            "ci_low_pct": round(float(r["95%CI下限"]) * 100, 2),
            "ci_high_pct": round(float(r["95%CI上限"]) * 100, 2),
            "n": int(res["n"]),
            "within_r2": round(float(res["within_r2"]), 4),
            "significant": bool(r["p值"] < 0.05),
        }
        print("%-28s %-9.2f %+-10.2f %-8.4f %-6d [%+.2f, %+.2f]   %.4f"
              % (name, out[name]["mean_days"], out[name]["beta_pct"], out[name]["p"],
                 out[name]["n"], out[name]["ci_low_pct"], out[name]["ci_high_pct"],
                 out[name]["within_r2"]))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    io.open(OUT, "w", encoding="utf-8").write(json.dumps(out, ensure_ascii=False, indent=2))
    print("\n已写 %s" % OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
