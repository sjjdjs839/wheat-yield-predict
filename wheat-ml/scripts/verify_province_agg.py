# -*- coding: utf-8 -*-
"""省级汇总回测。县级预测按播种面积加权加总，再和年鉴实测对一对。

    python -m scripts.verify_province_agg
结果在 output/reports/province_agg.json
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
LAB = ROOT / "data/raw/source_yield/河南省小麦产量标签_总表.csv"
STATS = ROOT / "output/models/county_stats.json"
MODEL = ROOT / "output/models/model_lgbm.joblib"
OUT = ROOT / "output/reports/province_agg.json"


def main() -> None:
    stats = json.load(io.open(STATS, encoding="utf-8"))
    prior = {k: v["features"]["county_prior"] for k, v in stats.items()
             if "county_prior" in v["features"]}
    global_mean = float(np.mean(list(prior.values())))

    ft = pd.read_csv(FT, encoding="utf-8-sig")
    model = joblib.load(MODEL)
    feats = list(model.feature_names_in_)

    # 模型输出为“相对县均值的比值”，须乘训练期县均值还原绝对单产
    ft["county_prior"] = ft["county"].map(prior).fillna(global_mean)
    ft["pred"] = model.predict(
        ft[feats].fillna(ft[feats].median(numeric_only=True))) * ft["county_prior"]

    lab = pd.read_csv(LAB, encoding="utf-8-sig").rename(columns={
        "年份": "year", "区县": "county", "小麦播种面积_千公顷": "area_kha",
        "小麦单产_kg每公顷": "obs_kgha"})
    lab["obs_kgmu"] = lab["obs_kgha"] / 15.0
    lab = lab.drop_duplicates(subset=["year", "county"])

    t = ft[ft["year"].between(2022, 2024)]
    m = t.merge(lab[["year", "county", "area_kha", "obs_kgmu"]],
                on=["year", "county"], how="inner").dropna(subset=["area_kha", "wheat_yield"])

    rows = []
    print("覆盖 %d 县 / %d 条 / 面积 %.0f 千公顷"
          % (m["county"].nunique(), len(m), m["area_kha"].sum()))
    print("%-6s %-6s %-10s %-10s %s" % ("年份", "县数", "加权预测", "加权实测", "偏差"))
    for y, g in m.groupby("year"):
        p = (g["pred"] * g["area_kha"]).sum() / g["area_kha"].sum()
        o = (g["wheat_yield"] * g["area_kha"]).sum() / g["area_kha"].sum()
        dev = (p - o) / o * 100
        rows.append({"year": int(y), "counties": int(len(g)),
                     "pred_kgmu": round(float(p), 1), "obs_kgmu": round(float(o), 1),
                     "dev_pct": round(float(dev), 2)})
        print("%-6d %-6d %-10.1f %-10.1f %+.2f%%" % (y, len(g), p, o, dev))

    p3 = (m["pred"] * m["area_kha"]).sum() / m["area_kha"].sum()
    o3 = (m["wheat_yield"] * m["area_kha"]).sum() / m["area_kha"].sum()
    total = {"n": int(len(m), ), "pred_kgmu": round(float(p3), 1),
             "obs_kgmu": round(float(o3), 1),
             "dev_pct": round(float((p3 - o3) / o3 * 100), 2)}
    print("三年合计: 预测 %.1f | 实测 %.1f | 偏差 %+.2f%%"
          % (p3, o3, (p3 - o3) / o3 * 100))

    err = (m["pred"] - m["wheat_yield"]).abs()
    ape = (err / m["wheat_yield"] * 100)
    print("县级: MAE %.2f | MAPE 中位 %.2f%%" % (err.mean(), ape.median()))

    payload = {"by_year": rows, "total": total,
               "county_mae": round(float(err.mean()), 2),
               "county_mape_median": round(float(ape.median()), 2)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    io.open(OUT, "w", encoding="utf-8").write(json.dumps(payload, ensure_ascii=False, indent=2))
    print("已写 %s" % OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
