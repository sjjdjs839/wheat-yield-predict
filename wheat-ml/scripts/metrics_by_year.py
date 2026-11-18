# -*- coding: utf-8 -*-
"""分年指标：按测试年（2022/2023/2024）分别报 R²、MAE 与系统偏差。

完全复刻 src/models/train_model.run_train 的训练与还原口径
（县域均值只用训练期年份计算，不泄露测试年产量）。
运行：python -m scripts.metrics_by_year
输出：output/reports/metrics_lgbm_by_year.json
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.models.train_model import CONFIG, FEATS, load_matrix, split_by_year

OUT = ROOT / "output" / "reports"


def main():
    df, _X, meta = load_matrix()
    rs = CONFIG["model"]["random_state"]
    test_idx, train_idx = split_by_year(df)

    train_df = df.loc[train_idx]
    county_mean = train_df.groupby("county")["wheat_yield"].mean()
    global_mean = train_df["wheat_yield"].mean()
    df["county_mean_yield"] = df["county"].map(county_mean).fillna(global_mean)
    df["county_prior"] = df["county_mean_yield"]
    y_norm = (df["wheat_yield"] / df["county_mean_yield"]).values
    X = df[FEATS].fillna(df[FEATS].median(numeric_only=True))

    model = LGBMRegressor(n_estimators=500, max_depth=8, learning_rate=0.05,
                          random_state=rs, verbose=-1)
    model.fit(X.loc[train_idx], y_norm[train_idx])

    te = df.loc[test_idx].copy()
    m_te = df.loc[test_idx, "county_mean_yield"].values
    te["pred"] = model.predict(X.loc[test_idx]) * m_te
    te["bias"] = te["pred"] - te["wheat_yield"]

    def block(sub):
        if len(sub) < 3:
            return {"n": int(len(sub))}
        return {
            "n": int(len(sub)),
            "r2": round(float(r2_score(sub["wheat_yield"], sub["pred"])), 4),
            "rmse": round(float(np.sqrt(mean_squared_error(sub["wheat_yield"], sub["pred"]))), 4),
            "mae": round(float(mean_absolute_error(sub["wheat_yield"], sub["pred"])), 4),
            "bias": round(float(sub["bias"].mean()), 2),
        }

    out = {
        "model": "lgbm",
        "note": "按测试年分组，指标在还原后的 kg/亩 尺度计算；bias = 预测−实测均值（正=系统性偏高）",
        "train_years": [int(y) for y in sorted(df.loc[train_idx, "year"].unique())],
        "test_years": [int(y) for y in sorted(te["year"].unique())],
        "overall": block(te),
        "by_year": {str(int(y)): block(sub) for y, sub in te.groupby("year")},
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "metrics_lgbm_by_year.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
