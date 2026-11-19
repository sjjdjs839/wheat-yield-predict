# -*- coding: utf-8 -*-
"""播种期、冬前、越冬期的气象到底影不影响产量。

这个脚本是被问出来的。指导老师看方案时问了句「为什么不考虑播种期、越冬期
这些数据」，于是把口径完全对齐 panel_impact——同样双向固定效应、同样按县
聚类标准误——回头把这四段窗口查了一遍：

  sow     播种期 10 月          降水、最低温、均温
  prewin  冬前 10—11 月         0℃/5℃ 积温、降水
  over    越冬期 12—2 月        极端最低温、≤-8℃/≤-10℃/≤0℃ 日数、降水
  feb     返青前 2 月           均温、最低温、降水

查完的结论分三种情况。

越冬冻害不成立：它被 cold_days 完全吸收，控制之后 t 从 -4.5 掉到 -0.4。
原因也简单——全年低温日里有 84% 就落在 12—2 月，本来就是同一批天。

越冬期降水、冬前 5℃ 积温是真信号，控制住全部变量后仍然显著，是现有
19 维特征没覆盖到的。

但把这两个加进预测模型，测试年 R² 反而降了（年际偏移外推不出去）。所以
只进「影响分析」，不进 19 维预测特征。
"""
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from panel_impact import demean_two_way, cluster_ols, fmt_model  # 复用同一套口径

BASE = Path(__file__).resolve().parents[1]
PROC = BASE / "data" / "processed"
FEAT = PROC / "feature_table.csv"
WEATHER = PROC / "weather_clean.csv"
STATS = BASE / "output" / "models" / "county_stats.json"
OUTDIR = BASE / "output" / "reports"
OUTDIR.mkdir(parents=True, exist_ok=True)

W_SPECS = [
    ("sow", dict(month=[10]), {
        "precip": ("precip", "sum"),
        "tmin_min": ("t2m_min", "min"),
        "tavg": ("t2m", "mean"),
        "dry_days": ("precip", lambda s: int((s < 0.1).sum())),
    }),
    ("prewin", dict(month=[10, 11]), {
        "gdd0": ("t2m", lambda s: float(np.maximum(s, 0).sum())),
        "gdd5": ("t2m", lambda s: float(np.maximum(s - 5, 0).sum())),
        "precip": ("precip", "sum"),
    }),
    ("over", dict(month=[12, 1, 2]), {
        "tmin_min": ("t2m_min", "min"),
        "tmin_mean": ("t2m_min", "mean"),
        "frost8": ("t2m_min", lambda s: int((s <= -8).sum())),
        "frost10": ("t2m_min", lambda s: int((s <= -10).sum())),
        "freeze": ("t2m_min", lambda s: int((s <= 0).sum())),
        "precip": ("precip", "sum"),
        "tavg": ("t2m", "mean"),
    }),
    ("feb", dict(month=[2]), {
        "tavg": ("t2m", "mean"),
        "tmin_min": ("t2m_min", "min"),
        "precip": ("precip", "sum"),
    }),
]


def build_windows() -> pd.DataFrame:
    w = pd.read_csv(WEATHER, usecols=["county", "year", "month", "t2m_max", "t2m_min", "t2m", "precip"])
    parts = []
    for prefix, cond, spec in W_SPECS:
        mask = np.zeros(len(w), dtype=bool)
        for m in cond["month"]:
            mask |= (w["month"] == m).to_numpy()
        g = w[mask].groupby(["county", "year"])
        named = {prefix + "_" + k: pd.NamedAgg(column=v[0], aggfunc=v[1]) for k, v in spec.items()}
        parts.append(g.agg(**named).reset_index())
    out = parts[0]
    for p in parts[1:]:
        out = out.merge(p, on=["county", "year"], how="outer")
    return out


def load_panel() -> pd.DataFrame:
    win = build_windows()
    df = pd.read_csv(FEAT).merge(win, on=["county", "year"], how="left")
    raw = json.load(io.open(STATS, encoding="utf-8"))
    df["base"] = df["county"].map(lambda c: raw[c]["mean_yield"] if c in raw else np.nan)
    df = df[df["wheat_yield"].notna() & df["base"].notna()].copy()
    df["y_norm"] = df["wheat_yield"] / df["base"]
    df["y_raw"] = df["wheat_yield"].astype(float)
    df["county"] = df["county"].astype(str)
    df["year"] = df["year"].astype(int)
    return df


def run(df, dep, xs, tag):
    ent = df["county"].reset_index(drop=True)
    tim = df["year"].reset_index(drop=True)
    dm = demean_two_way(df[[dep] + xs].reset_index(drop=True), ent, tim)
    r = cluster_ols(dm[dep].values, dm[xs].values, ent, xs)
    r["_tag"] = tag
    return r


def main():
    df = load_panel()
    print("样本：%d 县 × %d 年年 = %d 条（%d—%d）\n"
          % (df["county"].nunique(), df["year"].nunique(), len(df), df["year"].min(), df["year"].max()))

    new_cols = [c for c in df.columns if c.split("_")[0] in ("sow", "prewin", "over", "feb") and c.count("_") <= 2]
    new_cols = [c for c in new_cols if c not in ("sow_", "over_")]

    report = {"sample": {"n": int(len(df)), "counties": int(df["county"].nunique()),
                         "years": [int(df["year"].min()), int(df["year"].max())]},
              "note": "与 panel_impact.py 同一口径：双向固定效应 + 按县聚类稳健标准误",
              "single": {}, "joint": {}, "overlap": {}, "dose": {}}

    print("=" * 74)
    print("① 单变量：每个前期特征 ⊥ 县固定 + 年固定")
    print("=" * 74)
    print("%-18s %9s %8s %8s %10s" % ("变量", "β(%)", "t", "p", "withinR2"))
    for c in new_cols:
        sub = df.dropna(subset=[c])
        r = run(sub, "y_norm", [c], "single|" + c)
        b, se, p = float(r["beta"][0]), float(r["se"][0]), float(r["p"][0])
        report["single"][c] = {"beta": round(b, 4), "t": round(b / se, 2), "p": round(p, 4),
                               "within_r2": round(float(r["within_r2"]), 4), "n": r["n"]}
        stars = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.1 else ""))
        print("%-18s %+9.4f %8.2f %8.4f %10.4f %s" % (c, b, b / se, p, float(r["within_r2"]), stars))

    print("\n" + "=" * 74)
    print("② 关键检验：越冬冻害是不是 cold_days 的重复？")
    print("=" * 74)
    pairs = [
        (["over_frost8"], "仅越冬冻害"),
        (["over_frost8", "cold_days"], "越冬冻害 + 全年低温日"),
    ]
    for xs, label in pairs:
        r = run(df, "y_norm", xs, "overlap")
        print("\n[%s]  withinR2=%.4f  n=%d" % (label, float(r["within_r2"]), r["n"]))
        for row in fmt_model(r):
            print("   %-14s β=%+8.4f  se=%.4f  t=%+7.2f  p=%.4f %s"
                  % (row["变量"], row["系数"], row["标准误"], row["t"], row["p值"], row["显著性"]))
        report["overlap"][label] = {"within_r2": round(float(r["within_r2"]), 4),
                                    "n": r["n"], "coef": fmt_model(r)}

    print("\n" + "=" * 74)
    print("③ 联合模型：新信号 + 现有低温变量一起")
    print("=" * 74)
    joint = ["over_frost8", "cold_days", "season_tmin", "over_precip", "prewin_gdd5"]
    sub = df.dropna(subset=joint)
    r = run(sub, "y_norm", joint, "joint")
    print("withinR2=%.4f  n=%d" % (float(r["within_r2"]), r["n"])
          )
    for row in fmt_model(r):
        print("   %-14s β=%+8.4f  se=%.4f  t=%+7.2f  p=%.4f %s"
              % (row["变量"], row["系数"], row["标准误"], row["t"], row["p值"], row["显著性"]))
    report["joint"]["full"] = {"within_r2": round(float(r["within_r2"]), 4), "n": r["n"],
                               "coef": fmt_model(r)}

    # 无霜冻项的联合模型（看新信号自身是否站得住）
    joint2 = ["over_precip", "prewin_gdd5", "cold_days"]
    r2 = run(df.dropna(subset=joint2), "y_norm", joint2, "joint2")
    print("\n[不含越冬冻害] withinR2=%.4f" % float(r2["within_r2"]))
    for row in fmt_model(r2):
        print("   %-14s β=%+8.4f  t=%+7.2f  p=%.4f %s"
              % (row["变量"], row["系数"], row["t"], row["p值"], row["显著性"]))
    report["joint"]["without_frost"] = {"within_r2": round(float(r2["within_r2"]), 4),
                                        "coef": fmt_model(r2)}

    print("\n" + "=" * 74)
    print("④ 越冬冻害剂量分档（残差空间，控制县与年份）")
    print("=" * 74)
    dm = demean_two_way(df[["y_norm"]], df["county"], df["year"])
    df["y_res"] = dm["y_norm"].values
    bins = [-0.1, 0.5, 3.5, 7.5, 14.5, 1e9]
    labels = ["0 天", "1—3 天", "4—7 天", "8—14 天", "≥15 天"]
    df["band"] = pd.cut(df["over_frost8"], bins=bins, labels=labels)
    g = df.groupby("band", observed=True).agg(n=("y_res", "size"), 偏离=("y_res", "mean"),
                                             sd=("y_res", "std"))
    g["偏离%"] = (g["偏离"] * 100).round(2)
    g["标准误%"] = ((g["sd"] / np.sqrt(g["n"])) * 100).round(2)
    g["相对0天档"] = (g["偏离%"] - g.loc["0 天", "偏离%"]).round(2)
    print(g[["n", "偏离%", "标准误%", "相对0天档"]].to_string())
    report["dose"]["over_frost8"] = json.loads(g.reset_index().to_json(orient="records", force_ascii=False))

    print("\n" + "=" * 74)
    print("⑤ 反证：把前期特征塞进预测模型，测试年 R² 会怎样")
    print("=" * 74)
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import r2_score

    base17 = ["spi_12", "precip_anom", "hot_days", "cold_days", "gdd", "season_tmax", "season_tmin",
              "season_precip", "season_wind", "season_sunshine", "season_tmax_34", "season_precip_34",
              "gdd_34", "season_tmax_56", "season_precip_56", "gdd_56", "hot_dry_days"]
    tr, te = df.year <= 2021, df.year >= 2022

    def eval_feats(feats, label):
        X = df[feats].astype(float)
        X = X.fillna(X.median())
        pipe = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 30)))
        pipe.fit(X[tr], df.loc[tr, "y_norm"])
        p = pipe.predict(X[te])
        r2v = r2_score(df.loc[te, "y_norm"], p)
        print("   %-26s 测试年偏离 R2=%.4f  (%d 特征)" % (label, r2v, len(feats)))
        return round(float(r2v), 4)

    report["predictive_test"] = {
        "base17": eval_feats(base17, "现有 17 气象特征"),
        "plus_all": eval_feats(base17 + ["sow_precip", "sow_tavg", "prewin_gdd5", "prewin_precip",
                                         "over_tmin_mean", "over_frost8", "over_precip", "feb_tavg"], "加 8 项前期特征"),
        "plus_2": eval_feats(base17 + ["over_precip", "prewin_gdd5"], "只加 2 项独立信号"),
    }

    out = OUTDIR / "panel_overwinter.json"
    io.open(out, "w", encoding="utf-8").write(json.dumps(report, ensure_ascii=False, indent=2))
    print("\n已写出：", out)


if __name__ == "__main__":
    main()
