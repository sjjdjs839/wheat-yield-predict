# -*- coding: utf-8 -*-
"""对比 RF / LR / XGBoost / LightGBM"""
import yaml, sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from lightgbm import LGBMRegressor
try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

PROJECT_ROOT = Path(__file__).resolve().parents[2] / "wheat-ml"
sys.path.insert(0, str(PROJECT_ROOT))
with open(PROJECT_ROOT / "config" / "config.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

FEATS = cfg["feature_cols"]
feat = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "feature_table.csv")
y = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "county_yield.csv", encoding="utf-8")
y.columns = [c.strip().lower() for c in y.columns]

from src.data.build_features import match_county
canon = set(feat["county"].unique())
y["county"] = y["county"].map(lambda x: match_county(x, canon))
y = y.dropna(subset=["county", "year", "wheat_yield"])
df = feat.merge(y[["county", "year", "wheat_yield"]], on=["county", "year"], how="left")
df = df.dropna(subset=["wheat_yield"]).copy().reset_index(drop=True)

X = df[FEATS].fillna(df[FEATS].median(numeric_only=True))

test_years = {2022, 2023, 2024}
mask = df["year"].isin(test_years).values
train_idx, test_idx = ~mask, mask

# 只用训练集算县均值（不泄露）
train_df = df[train_idx]
county_mean = train_df.groupby("county")["wheat_yield"].mean()
global_mean = train_df["wheat_yield"].mean()
df["county_mean"] = df["county"].map(county_mean).fillna(global_mean)
y_norm = (df["wheat_yield"] / df["county_mean"]).values

X_tr, X_te = X[train_idx], X[test_idx]
y_tr, y_te = y_norm[train_idx], y_norm[test_idx]
m_te = df.loc[test_idx, "county_mean"].values
y_te_true = y_te * m_te

models = {
    "线性回归": LinearRegression(),
    "随机森林": RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
    "LightGBM": LGBMRegressor(n_estimators=300, max_depth=6, learning_rate=0.1, random_state=42, verbose=-1),
}
if XGBRegressor is not None:
    models["XGBoost"] = XGBRegressor(n_estimators=300, max_depth=6,
                                     learning_rate=0.1, random_state=42)
else:
    print("未安装 xgboost，跳过 XGBoost 对比")

print(f"{'模型':<10} | {'R²':>8} | {'RMSE(kg/亩)':>12} | {'MAE(kg/亩)':>10}")
print("-" * 55)

for name, model in models.items():
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    pred_raw = pred * m_te
    r2 = r2_score(y_te_true, pred_raw)
    rmse = np.sqrt(mean_squared_error(y_te_true, pred_raw))
    mae = mean_absolute_error(y_te_true, pred_raw)
    print(f"{name:<10} | {r2:>8.4f} | {rmse:>12.1f} | {mae:>10.1f}")
