"""
生成演示用产量标签（仅在真实产量缺失时使用）
==============================================
用途：在 data/raw/county_yield.csv 到位前，用可复现的合成标签跑通
「特征 → 训练 → 预测 → 预警 → 农事建议」全链路。

注意：合成标签不代表任何真实产量，界面与指标必须标注「演示数据」，
      R² / MAE / RMSE 不构成准确率结论。真实产量到位后请直接覆盖
      data/raw/county_yield.csv 并重新训练。

用法：python -m scripts.make_demo_labels
"""

import yaml
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
with open(PROJECT_ROOT / "config" / "config.yaml", "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

FEAT = PROJECT_ROOT / CONFIG["paths"]["processed"] / "feature_table.csv"
OUT = PROJECT_ROOT / CONFIG["paths"]["raw_yield_demo"]

rng = np.random.default_rng(CONFIG["model"]["random_state"])


def build() -> pd.DataFrame:
    df = pd.read_csv(FEAT)

    county_mean = df.groupby("county")[["season_precip", "season_tmax_56", "hot_dry_days"]].mean()

    # 县基准单产：全省基准 + 县间差异（固定随机效应）
    base = 420 + rng.normal(0, 18, len(county_mean))
    base = pd.Series(base, index=county_mean.index)

    out = []
    for _, r in df.iterrows():
        c = r["county"]
        cm = county_mean.loc[c]
        y = (base[c]
             + 0.5 * (r["year"] - 2010)                       # 年际技术提升趋势
             + 0.25 * (r["season_precip"] - cm["season_precip"])
             - 2.5 * (r["season_tmax_56"] - cm["season_tmax_56"])
             - 1.5 * (r["hot_dry_days"] - cm["hot_dry_days"])
             + rng.normal(0, 20))                              # 年际随机波动
        out.append({"county": c, "year": int(r["year"]),
                    "wheat_yield": round(float(np.clip(y, 300, 520)), 1)})

    res = pd.DataFrame(out).sort_values(["county", "year"]).reset_index(drop=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT, index=False, encoding="utf-8")
    print(f"演示标签: {len(res)} 行, {res['county'].nunique()} 个地区, "
          f"单产范围 {res['wheat_yield'].min()}—{res['wheat_yield'].max()} kg/亩 -> {OUT}")
    return res


if __name__ == "__main__":
    build()
