# -*- coding: utf-8 -*-
"""
验证模型真实性：
1. 用原始单产（kg/亩）直接预测，不做归一化
2. 时间序列滚动交叉验证
"""
import yaml
import sys
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
PROCESSED = PROJECT_ROOT / "data" / "processed"

# 读数据
feat = pd.read_csv(PROCESSED / "feature_table.csv")
y = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "county_yield.csv", encoding="utf-8")
y.columns = [c.strip().lower() for c in y.columns]

# 对齐
from src.data.build_features import match_county
canon = set(feat["county"].unique())
y["county"] = y["county"].map(lambda x: match_county(x, canon))
y = y.dropna(subset=["county", "year", "wheat_yield"])
df = feat.merge(y[["county", "year", "wheat_yield"]], on=["county", "year"], how="left")
df = df.dropna(subset=["wheat_yield"]).copy().reset_index(drop=True)

X = df[FEATS].fillna(df[FEATS].median(numeric_only=True))
y_raw = df["wheat_yield"].values  # 原始单产 kg/亩

# 归一化版本（队友原来的方式）
county_mean = df.groupby("county")["wheat_yield"].mean()
df["county_mean"] = df["county"].map(county_mean)
y_norm = (df["wheat_yield"] / df["county_mean"]).values

print("=" * 60)
print("数据概况")
print("=" * 60)
print(f"样本数: {len(df)}")
print(f"区县数: {df['county'].nunique()}")
print(f"年份范围: {df['year'].min()}—{df['year'].max()}")
print(f"原始单产范围: {y_raw.min():.0f}—{y_raw.max():.0f} kg/亩, 均值={y_raw.mean():.0f}")
print()

# ========== 1. 原始单产 vs 归一化单产 ==========
print("=" * 60)
print("【实验1】原始单产 vs 归一化单产（按年份划分: 2010-2021训练, 2022-2024测试）")
print("=" * 60)

test_years = {2022, 2023, 2024}
mask = df["year"].isin(test_years).values
train_idx = ~mask
test_idx = mask

X_tr, X_te = X[train_idx], X[test_idx]
y_raw_tr, y_raw_te = y_raw[train_idx], y_raw[test_idx]
y_norm_tr, y_norm_te = y_norm[train_idx], y_norm[test_idx]

for name, model in [
    ("随机森林", RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)),
    ("线性回归", LinearRegression()),
]:
    # 原始单产
    model.fit(X_tr, y_raw_tr)
    pred = model.predict(X_te)
    r2_raw = r2_score(y_raw_te, pred)
    rmse_raw = np.sqrt(mean_squared_error(y_raw_te, pred))
    mae_raw = mean_absolute_error(y_raw_te, pred)

    # 归一化单产
    model.fit(X_tr, y_norm_tr)
    pred_norm = model.predict(X_te)
    r2_norm = r2_score(y_norm_te, pred_norm)
    # 还原到原始尺度
    means_te = df.loc[test_idx, "county_mean"].values
    pred_raw_from_norm = pred_norm * means_te
    y_raw_from_norm = y_norm_te * means_te
    r2_norm_raw = r2_score(y_raw_from_norm, pred_raw_from_norm)
    rmse_norm_raw = np.sqrt(mean_squared_error(y_raw_from_norm, pred_raw_from_norm))
    mae_norm_raw = mean_absolute_error(y_raw_from_norm, pred_raw_from_norm)

    print(f"\n{name}:")
    print(f"  直接预测原始单产: R²={r2_raw:.4f}, RMSE={rmse_raw:.1f} kg/亩, MAE={mae_raw:.1f}")
    print(f"  归一化后预测(归一化尺度): R²={r2_norm:.4f}")
    print(f"  归一化后预测(还原到原始尺度): R²={r2_norm_raw:.4f}, RMSE={rmse_norm_raw:.1f} kg/亩, MAE={mae_norm_raw:.1f}")

# ========== 2. 时间序列滚动交叉验证 ==========
print("\n" + "=" * 60)
print("【实验2】时间序列滚动交叉验证（用原始单产）")
print("=" * 60)

years = sorted(df["year"].unique())
print(f"所有年份: {years}")
print()

# 滚动：训练集逐年增加，测试集=下一年
print(f"{'测试年':>6} | {'训练年数':>6} | {'训练样本':>6} | {'测试样本':>6} | {'RF_R²':>8} | {'LR_R²':>8} | {'RF_RMSE':>8}")
print("-" * 75)

rf_r2s, lr_r2s = [], []
rf_rmses, lr_rmses = [], []

for i in range(3, len(years)):  # 至少用前3年训练
    test_year = years[i]
    train_years_set = set(years[:i])
    tr_mask = df["year"].isin(train_years_set).values
    te_mask = df["year"] == test_year

    X_tr_cv, X_te_cv = X[tr_mask], X[te_mask]
    y_tr_cv, y_te_cv = y_raw[tr_mask], y_raw[te_mask]

    if len(y_te_cv) < 10:
        continue

    # RF
    rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    rf.fit(X_tr_cv, y_tr_cv)
    pred_rf = rf.predict(X_te_cv)
    r2_rf = r2_score(y_te_cv, pred_rf)
    rmse_rf = np.sqrt(mean_squared_error(y_te_cv, pred_rf))

    # LR
    lr = LinearRegression()
    lr.fit(X_tr_cv, y_tr_cv)
    pred_lr = lr.predict(X_te_cv)
    r2_lr = r2_score(y_te_cv, pred_lr)
    rmse_lr = np.sqrt(mean_squared_error(y_te_cv, pred_lr))

    rf_r2s.append(r2_rf)
    lr_r2s.append(r2_lr)
    rf_rmses.append(rmse_rf)
    lr_rmses.append(rmse_lr)

    print(f"{test_year:>6} | {i:>6} | {tr_mask.sum():>6} | {te_mask.sum():>6} | {r2_rf:>8.4f} | {r2_lr:>8.4f} | {rmse_rf:>8.1f}")

print("-" * 75)
print(f"{'平均':>6} | {'':>6} | {'':>6} | {'':>6} | {np.mean(rf_r2s):>8.4f} | {np.mean(lr_r2s):>8.4f} | {np.mean(rf_rmses):>8.1f}")
print()
print(f"RF R²范围: {min(rf_r2s):.4f} ~ {max(rf_r2s):.4f}")
print(f"LR R²范围: {min(lr_r2s):.4f} ~ {max(lr_r2s):.4f}")
print()
print("结论: 如果各年R²稳定在0.7-0.9之间，说明模型不是过拟合，是真的能预测产量。")
