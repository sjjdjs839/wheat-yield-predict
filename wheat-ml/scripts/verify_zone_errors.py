# -*- coding: utf-8 -*-
"""分象限看误差。县均单产和干热风暴露各按中位数切一刀，四象限比一比谁的误差大。

    python -m scripts.verify_zone_errors
结果在 output/reports/zone_errors.json
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FT = ROOT / "data/processed/feature_table.csv"
STATS = ROOT / "output/models/county_stats.json"
MODEL = ROOT / "output/models/model_lgbm.joblib"
OUT = ROOT / "output/reports/zone_errors.json"


def zone_of(yield_mean: float, dry_days: float, med_y: float, med_d: float) -> str:
    if yield_mean >= med_y and dry_days < med_d:
        return "A 稳产核心区"
    if yield_mean >= med_y and dry_days >= med_d:
        return "B 高产风险区"
    if yield_mean < med_y and dry_days < med_d:
        return "C 潜力提升区"
    return "D 脆弱敏感区"


def main() -> None:
    stats = json.load(io.open(STATS, encoding="utf-8"))
    prior = {k: v["features"]["county_prior"] for k, v in stats.items()
             if "county_prior" in v["features"]}
    dry = {k: v["features"]["hot_dry_days"] for k, v in stats.items()
           if "hot_dry_days" in v["features"]}

    ft = pd.read_csv(FT, encoding="utf-8-sig")
    model = joblib.load(MODEL)
    feats = list(model.feature_names_in_)
    ft["county_prior"] = ft["county"].map(prior)
    ft["pred"] = model.predict(
        ft[feats].fillna(ft[feats].median(numeric_only=True))) * ft["county_prior"]

    med_y = float(np.median(list(prior.values())))
    med_d = float(np.median(list(dry.values())))
    print("切分阈值：县均单产 %.1f kg/亩 | 干热风 %.2f 天/年" % (med_y, med_d))

    ft["zone"] = [zone_of(prior.get(c, np.nan), dry.get(c, np.nan), med_y, med_d)
                  if (c in prior and c in dry) else None for c in ft["county"]]

    test = ft[ft["year"].between(2022, 2024)].dropna(subset=["wheat_yield", "zone"])
    test["err"] = test["pred"] - test["wheat_yield"]
    test["ape"] = test["err"].abs() / test["wheat_yield"] * 100

    out = {}
    print("\n%-14s %-6s %-8s %-10s %s"
          % ("象限", "县数", "MAE", "MAPE中位", "平均偏差"))
    for z, g in test.groupby("zone"):
        out[z] = {"counties": int(g["county"].nunique()), "n": int(len(g)),
                  "mae": round(float(g["err"].abs().mean()), 2),
                  "mape_median": round(float(g["ape"].median()), 2),
                  "bias": round(float(g["err"].mean()), 2)}
        print("%-14s %-6d %-8.1f %-9.2f%% %+.1f"
              % (z, out[z]["counties"], out[z]["mae"], out[z]["mape_median"], out[z]["bias"]))

    out["_all"] = {"counties": int(test["county"].nunique()), "n": int(len(test)),
                   "mae": round(float(test["err"].abs().mean()), 2),
                   "mape_median": round(float(test["ape"].median()), 2),
                   "bias": round(float(test["err"].mean()), 2)}
    print("全体: MAE %.1f | MAPE 中位 %.2f%% | 平均偏差 %+.1f"
          % (out["_all"]["mae"], out["_all"]["mape_median"], out["_all"]["bias"]))

    # 县 -> 象限，供页面按县给出已知局限提示
    zmap = {}
    for c, z in zip(ft["county"], ft["zone"]):
        if z:
            zmap[str(c)] = z
    out["_county_zone"] = zmap

    OUT.parent.mkdir(parents=True, exist_ok=True)
    io.open(OUT, "w", encoding="utf-8").write(json.dumps(out, ensure_ascii=False, indent=2))
    print("\n已写 %s" % OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
