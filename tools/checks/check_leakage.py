# -*- coding: utf-8 -*-
"""
修复数据泄露：county_mean只能用训练集年份计算，不能用测试集
"""
import yaml, sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

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
train_idx = ~mask
test_idx = mask

# ===== 错误做法（泄露）：用全量数据算county_mean =====
county_mean_leak = df.groupby("county")["wheat_yield"].mean()
df["mean_leak"] = df["county"].map(county_mean_leak)
y_norm_leak = (df["wheat_yield"] / df["mean_leak"]).values

# ===== 正确做法：只用训练集年份算county_mean =====
train_df = df[train_idx]
county_mean_correct = train_df.groupby("county")["wheat_yield"].mean()
df["mean_correct"] = df["county"].map(county_mean_correct)
# 有些县可能只在测试集出现，填全局均值
global_mean = train_df["wheat_yield"].mean()
df["mean_correct"] = df["mean_correct"].fillna(global_mean)
y_norm_correct = (df["wheat_yield"] / df["mean_correct"]).values

print("=" * 60)
print("对比：泄露版 vs 正确版（归一化预测，还原到原始尺度）")
print("=" * 60)

for name, model in [
    ("随机森林", RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)),
    ("线性回归", LinearRegression()),
]:
    X_tr, X_te = X[train_idx], X[test_idx]

    # 泄露版
    model.fit(X_tr, y_norm_leak[train_idx])
    pred_leak = model.predict(X_te)
    y_te_true = df.loc[test_idx, "wheat_yield"].values
    pred_leak_raw = pred_leak * df.loc[test_idx, "mean_leak"].values
    r2_leak = r2_score(y_te_true, pred_leak_raw)

    # 正确版
    model.fit(X_tr, y_norm_correct[train_idx])
    pred_correct = model.predict(X_te)
    pred_correct_raw = pred_correct * df.loc[test_idx, "mean_correct"].values
    r2_correct = r2_score(y_te_true, pred_correct_raw)
    rmse_correct = np.sqrt(mean_squared_error(y_te_true, pred_correct_raw))
    mae_correct = mean_absolute_error(y_te_true, pred_correct_raw)

    print(f"\n{name}:")
    print(f"  泄露版（用全量数据算均值）: R²={r2_leak:.4f}  ← 之前的0.92就是这么来的")
    print(f"  正确版（只用训练集算均值）:   R²={r2_correct:.4f}, RMSE={rmse_correct:.1f} kg/亩, MAE={mae_correct:.1f}")
