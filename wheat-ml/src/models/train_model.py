"""
模型训练模块（对齐项目方案 第六章）
====================================
目标变量: yield_norm = wheat_yield ÷ 该县多年均值（学"相对丰欠"）
预测后:   × county_mean_yield 还原真实单产，再算 R²/RMSE/MAE
模型:     RandomForestRegressor + LinearRegression
划分:     按年份划分训练/测试集（config.model.split.test_years），禁止随机拆分

产量标签来源：
  1. data/raw/county_yield.csv          —— 真实产量（metrics 标记 reliable=true）
  2. data/demo/county_yield_demo.csv    —— 演示标签（仅用于跑通链路，reliable=false）

产出:
  output/models/model_rf.joblib, model_lr.joblib, model_meta.json
  output/models/county_stats.json
  output/reports/metrics_{rf,lr}.json, importance_rf.csv, predictions_rf.csv
"""

import json
import yaml
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from lightgbm import LGBMRegressor

from src.data.build_features import match_county

PROJECT_ROOT = Path(__file__).resolve().parents[2]
with open(PROJECT_ROOT / "config" / "config.yaml", "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

FEATS = CONFIG["feature_cols"]
PROCESSED = PROJECT_ROOT / CONFIG["paths"]["processed"]
MODELS = PROJECT_ROOT / CONFIG["paths"]["models"]
REPORTS = PROJECT_ROOT / CONFIG["paths"]["reports"]


def resolve_yield_source() -> tuple[Path, bool]:
    """优先真实产量标签；缺失时按配置回退到演示标签"""
    real = PROJECT_ROOT / CONFIG["paths"]["raw_yield"]
    demo = PROJECT_ROOT / CONFIG["paths"]["raw_yield_demo"]
    if real.exists():
        return real, True
    if CONFIG["model"].get("allow_demo_labels") and demo.exists():
        return demo, False
    raise FileNotFoundError(
        f"缺少产量标签：请将真实产量放到 {real}"
        f"（字段：{', '.join(['county, year, wheat_yield']) }）"
    )


def load_matrix() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, dict]:
    """
    读特征表 + 产量标签 -> (df, X, y_norm, county_means, meta)
    """
    feat = pd.read_csv(PROCESSED / "feature_table.csv")
    # 去掉已有的wheat_yield列，重新merge
    if "wheat_yield" in feat.columns:
        feat = feat.drop(columns=["wheat_yield"])
    ypath, is_real = resolve_yield_source()

    y = pd.read_csv(ypath, encoding="utf-8")
    y.columns = [c.strip().lower() for c in y.columns]
    if "wheat_yield" not in y.columns:
        alt = [c for c in y.columns if "yield" in c or "产量" in c]
        if not alt:
            raise ValueError(f"产量文件缺少产量列: {ypath}")
        y = y.rename(columns={alt[0]: "wheat_yield"})
    if "region" in y.columns and "county" not in y.columns:
        y = y.rename(columns={"region": "county"})

    canon = set(feat["county"].unique())
    y["county"] = y["county"].map(lambda x: match_county(x, canon))
    y = y.dropna(subset=["county", "year", "wheat_yield"])

    df = feat.merge(y[["county", "year", "wheat_yield"]], on=["county", "year"], how="left")
    df = df.dropna(subset=["wheat_yield"]).copy().reset_index(drop=True)

    X = df[FEATS].fillna(df[FEATS].median(numeric_only=True))

    meta = {
        "yield_file": str(ypath.relative_to(PROJECT_ROOT)),
        "data_source": "real" if is_real else "demo",
        "reliable": bool(is_real),
        "n_samples": int(len(df)),
        "n_counties": int(df["county"].nunique()),
        "year_range": [int(df["year"].min()), int(df["year"].max())],
    }
    return df, X, meta


def split_by_year(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """按年份划分：测试年份取自 config，其余年份为训练集"""
    test_years = set(int(y) for y in CONFIG["model"]["split"]["test_years"])
    mask = df["year"].isin(test_years).values
    if mask.sum() == 0:
        raise ValueError(f"测试年份 {sorted(test_years)} 在标签中无样本，请检查配置")
    if (~mask).sum() == 0:
        raise ValueError("训练集为空，请减少测试年份")
    return df[mask].index.values, df[~mask].index.values


def evaluate(model, X_test, y_test_true, means_test) -> tuple[dict, np.ndarray]:
    """在还原后的真实尺度上算指标"""
    y_pred = model.predict(X_test) * means_test
    return {
        "r2": round(float(r2_score(y_test_true, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test_true, y_pred))), 4),
        "mae": round(float(mean_absolute_error(y_test_true, y_pred)), 4),
    }, y_pred


def run_train() -> dict:
    df, X, meta = load_matrix()
    rs = CONFIG["model"]["random_state"]

    test_idx, train_idx = split_by_year(df)

    # 只用训练集年份计算县域平均单产（避免数据泄露）
    train_df = df.loc[train_idx]
    county_mean = train_df.groupby("county")["wheat_yield"].mean()
    global_mean = train_df["wheat_yield"].mean()
    df["county_mean_yield"] = df["county"].map(county_mean).fillna(global_mean)
    y_norm = (df["wheat_yield"] / df["county_mean_yield"]).values
    means = df["county_mean_yield"].values

    X_tr, X_te = X.loc[train_idx], X.loc[test_idx]
    y_tr, y_te = y_norm[train_idx], y_norm[test_idx]
    m_te = means[test_idx]
    y_te_true = y_te * m_te

    test_years = sorted(int(y) for y in df.loc[test_idx, "year"].unique())
    train_years = sorted(int(y) for y in df.loc[train_idx, "year"].unique())

    MODELS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    out = {"meta": meta, "test_years": test_years, "train_years": train_years}
    models_to_run = {
        "lgbm": LGBMRegressor(n_estimators=500, max_depth=8, learning_rate=0.05,
                              random_state=rs, verbose=-1),
        "rf": RandomForestRegressor(
            n_estimators=500, max_depth=12, min_samples_split=5,
            random_state=rs, n_jobs=-1),
        "lr": LinearRegression(),
    }
    for name, model in models_to_run.items():
        model.fit(X_tr, y_tr)
        met, y_pred = evaluate(model, X_te, y_te_true, m_te)
        met.update({
            "model": name,
            **meta,
            "n_train": int(len(X_tr)),
            "n_test": int(len(X_te)),
            "split": "按年份划分",
            "train_years": train_years,
            "test_years": test_years,
            "target": "yield_norm -> 还原真实单产 (kg/亩)",
        })
        joblib.dump(model, MODELS / f"model_{name}.joblib")
        (REPORTS / f"metrics_{name}.json").write_text(
            json.dumps(met, ensure_ascii=False, indent=2), encoding="utf-8")

        if name == "lgbm":
            pd.DataFrame({"feature": FEATS, "importance": model.feature_importances_}) \
                .sort_values("importance", ascending=False) \
                .to_csv(REPORTS / "importance_lgbm.csv", index=False, encoding="utf-8")
            pd.DataFrame({
                "county": df.loc[test_idx, "county"].values,
                "year": df.loc[test_idx, "year"].values,
                "y_true": y_te_true, "y_pred": y_pred, "error": y_te_true - y_pred,
            }).to_csv(REPORTS / "predictions_lgbm.csv", index=False, encoding="utf-8")

        out[name] = met
        tag = "" if meta["reliable"] else "  [演示标签，不可作为准确率结论]"
        print(f"[{name.upper()}] R²={met['r2']}  RMSE={met['rmse']}  MAE={met['mae']}"
              f"  (训练 {len(X_tr)} / 测试 {len(X_te)} 条){tag}")

    # 供 Web / 大屏做情景预测：县均值单产 + 县内各特征多年基准
    stats = {}
    for county, d in df.groupby("county"):
        stats[county] = {
            "mean_yield": round(float(d["wheat_yield"].mean()), 2),
            "features": {f: float(d[f].mean()) for f in FEATS},
        }
    (MODELS / "county_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False), encoding="utf-8")

    (MODELS / "model_meta.json").write_text(json.dumps({
        "features": FEATS,
        "data_source": meta["data_source"],
        "reliable": meta["reliable"],
        "yield_file": meta["yield_file"],
        "train_years": train_years,
        "test_years": test_years,
        "metrics_rf": out["rf"],  # 随机森林（精度最高）
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n模型与县基准已保存 -> {MODELS}")
    return out


if __name__ == "__main__":
    run_train()
