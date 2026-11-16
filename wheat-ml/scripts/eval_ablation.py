# -*- coding: utf-8 -*-
"""
按 3—5 月新口径重算消融与滚动验证，替换旧（3—6 月）报告。
输出：output/reports/feature_ablation.json、rolling_validation.json
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

FEAT = ROOT / "data" / "processed" / "feature_table.csv"
OUT = ROOT / "output" / "reports"

WEATHER = ["spi_12", "precip_anom", "hot_days", "cold_days", "gdd",
           "season_tmax", "season_tmin", "season_precip", "season_wind", "season_sunshine",
           "season_tmax_34", "season_precip_34", "gdd_34",
           "season_tmax_56", "season_precip_56", "gdd_56", "hot_dry_days"]
GROUPS = {
    "weather_only": WEATHER,
    "weather_trend": WEATHER + ["year_index"],
}

df = pd.read_csv(FEAT)
df = df[df["wheat_yield"].notna()].reset_index(drop=True)


def metrics(true, pred):
    return {"r2": round(float(r2_score(true, pred)), 4),
            "rmse": round(float(mean_squared_error(true, pred) ** 0.5), 4),
            "mae": round(float(mean_absolute_error(true, pred)), 4)}


def evaluate(tr, te, cols, label):
    cm = tr.groupby("county")["wheat_yield"].mean()
    gmean = tr["wheat_yield"].mean()
    tr = tr.assign(prior=tr["county"].map(cm).fillna(gmean),
                   y=tr["wheat_yield"] / tr["county"].map(cm).fillna(gmean))
    te = te.assign(prior=te["county"].map(cm).fillna(gmean),
                   y=te["wheat_yield"] / te["county"].map(cm).fillna(gmean))
    if len(cols) == 0:      # 纯基线：直接用地县常年均值
        pred_kg = te["prior"].to_numpy()
        return {"label": label, "n_features": 0, "metrics": metrics(te["wheat_yield"], pred_kg),
                "metrics_on_deviation": None}
    tr_x = tr[cols].fillna(0)
    if "prior" in cols:
        tr_x = tr_x.assign(prior=tr["prior"])
        te_x = te[cols].fillna(0).assign(prior=te["prior"])
    else:
        te_x = te[cols].fillna(0)
    m = RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=4)
    m.fit(tr_x, tr["y"])
    p = m.predict(te_x)
    return {"label": label, "n_features": len(cols),
            "metrics": metrics(te["wheat_yield"], p * te["prior"].to_numpy()),
            "metrics_on_deviation": metrics(te["y"], p)}


train = df[df["year"] <= 2021]
test = df[df["year"] >= 2022]

out = {
    "window": "3—5 月（截至 5 月 31 日灌浆末期）",
    "split": {"train_years": "2010—2021", "test_years": "2022—2024",
              "n_train": int(len(train)), "n_test": int(len(test)),
              "n_counties": int(df["county"].nunique())},
    "groups": {
        "baseline_county_mean": evaluate(train, test, [], "直接用县常年均值（不训练）"),
        "weather_only": evaluate(train, test, WEATHER, "纯气象 17 项"),
        "weather_trend": evaluate(train, test, WEATHER + ["year_index"], "气象 + 年份趋势"),
        "full": evaluate(train, test, WEATHER + ["year_index", "prior"], "气象 + 趋势 + 县先验（当前模型）"),
    },
    "conclusion": ("绝对精度的主体来自「该县常年单产」基线（R²≈0.906）；模型增益体现为对当代偏离方向的判断。"
                   "纯气象特征对「相对常年偏离」的解释力有限，故页面同时给出基线、区间与预警概率，"
                   "不将点估计作为精确结论。"),
}
(OUT / "feature_ablation.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
print("=== 消融（新口径 3—5 月）===")
for k, v in out["groups"].items():
    dev = v["metrics_on_deviation"]
    print(f"  {k:22} 绝对 R²={v['metrics']['r2']:.4f} MAE={v['metrics']['mae']:6.2f}"
          + (f" | 偏离 R²={dev['r2']:.4f} MAE={dev['mae']*100:.1f}%" if dev else ""))

# ---------------- 滚动验证 ----------------
roll = {"window": out["window"], "folds": {}}
cols_full = WEATHER + ["year_index", "prior"]
for test_year in (2020, 2021, 2022):
    tr = df[df["year"] < test_year]
    te = df[df["year"] == test_year]
    r = evaluate(tr, te, cols_full, f"train<{test_year}")
    roll["folds"][f"fold-{test_year}"] = {
        "n_train": int(len(tr)), "n_test": int(len(te)),
        "r2": r["metrics"]["r2"], "mae": r["metrics"]["mae"],
        "mae_deviation_pct": round(r["metrics_on_deviation"]["mae"] * 100, 1),
    }
    print(f"  滚动 {test_year}: 训练 {len(tr)} 测试 {len(te)} → R²={r['metrics']['r2']:.4f} "
          f"MAE={r['metrics']['mae']:.2f}")

(OUT / "rolling_validation.json").write_text(json.dumps(roll, ensure_ascii=False, indent=2),
                                             encoding="utf-8")
print("\n已写入 output/reports/feature_ablation.json 与 rolling_validation.json")
