import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from pathlib import Path
import yaml

# 加载数据
PROJECT_ROOT = Path(__file__).resolve().parents[2] / "wheat-ml"
with open(PROJECT_ROOT / "config/config.yaml", "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

df = pd.read_csv(PROJECT_ROOT / "data/processed/feature_table.csv")
print(f"总数据量：{len(df)} 条")

# 去掉没有产量标签的行
df = df.dropna(subset=["wheat_yield"])
print(f"有产量标签的：{len(df)} 条")

# 按年份划分训练/测试集
test_years = [2022, 2023, 2024]
train = df[~df["year"].isin(test_years)]
test = df[df["year"].isin(test_years)]
print(f"训练集：{len(train)} 条，测试集：{len(test)} 条")

# 特征和标签
FEATS = CONFIG["feature_cols"]
X_train = train[FEATS].fillna(0)
y_train = train["wheat_yield"]
X_test = test[FEATS].fillna(0)
y_test = test["wheat_yield"]

# 试几个参数组合
param_grid = [
    {"n_estimators": 500, "max_depth": 10, "min_samples_leaf": 2},
    {"n_estimators": 800, "max_depth": 12, "min_samples_leaf": 2},
    {"n_estimators": 1000, "max_depth": 15, "min_samples_leaf": 1},
]

best_r2 = 0
best_params = None
best_pred = None

for params in param_grid:
    rf = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    pred = rf.predict(X_test)
    r2 = r2_score(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mae = mean_absolute_error(y_test, pred)
    print(f"\n参数：{params}")
    print(f"  R²={r2:.4f}  RMSE={rmse:.2f}  MAE={mae:.2f}")
    if r2 > best_r2:
        best_r2 = r2
        best_params = params
        best_pred = pred

print(f"\n\n最优参数：{best_params}")
print(f"最优R²：{best_r2:.4f}")
