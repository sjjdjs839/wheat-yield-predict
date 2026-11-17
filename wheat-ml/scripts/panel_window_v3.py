# -*- coding: utf-8 -*-
"""
panel_window_v3.py —— 关键窗口定位：低温/晚霜冻到底哪个月起决定作用

候选指标（均在逐日数据上重建）：
  fz3_0 / fz4_0 / fz5_0   3 / 4 / 5 月 最低温 ≤0℃ 天数
  fz3_2 / fz4_2 / fz5_2   3 / 4 / 5 月 最低温 ≤2℃ 天数（宽口径）
  fz4_streak              4 月最长连续 ≤0℃ 天数（一次寒潮的持续长度）
  hd5_streak              5 月最长连续干热风天数
联合模型：4 月低温 + 5 月干热风，检验两者是否各自独立起作用。
分区异质性：四区分别估 4 月低温弹性。
"""
import io, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
from panel_impact import load_panel, demean_two_way, cluster_ols, fmt_model, OUTDIR

BASE = SCRIPTS.parent
WEATHER = BASE / "data" / "processed" / "weather_clean.csv"


def _max_streak(b: np.ndarray) -> int:
    best = cur = 0
    for v in b:
        cur = cur + 1 if v else 0
        if cur > best:
            best = cur
    return best


def build() -> pd.DataFrame:
    w = pd.read_csv(WEATHER, usecols=["county", "year", "month", "month_day",
                                      "t2m_max", "t2m_min", "rh"])
    w = w.sort_values(["county", "year", "month", "month_day"])

    rows = []
    for (c, y), g in w.groupby(["county", "year"], sort=False):
        rec = {"county": c, "year": y}
        for m in (3, 4, 5):
            gm = g[g["month"] == m]
            rec["fz%d_0" % m] = int((gm["t2m_min"] <= 0).sum())
            rec["fz%d_2" % m] = int((gm["t2m_min"] <= 2).sum())
        apr = g[g["month"] == 4]
        rec["fz4_streak"] = _max_streak((apr["t2m_min"] <= 0).to_numpy())
        may = g[g["month"] == 5]
        rec["hd5_streak"] = _max_streak(
            ((may["t2m_max"] >= 35) & (may["rh"] <= 40)).to_numpy())
        rows.append(rec)
    return pd.DataFrame(rows)


def reg(df, x, dep="y_norm"):
    ent = df["county"].reset_index(drop=True)
    tim = df["year"].reset_index(drop=True)
    dm = demean_two_way(df[[dep, x]].reset_index(drop=True), ent, tim)
    return cluster_ols(dm[dep].values, dm[[x]].values, ent, [x])


def main() -> None:
    dis = build()
    df = load_panel().merge(dis, on=["county", "year"], how="left")
    print("灾害表 %d 行 / %d 县，匹配后 %d 条观测\n" % (len(dis), dis["county"].nunique(), len(df)))

    cand = ["fz3_0", "fz4_0", "fz5_0", "fz3_2", "fz4_2", "fz5_2", "fz4_streak", "hd5_streak"]
    print("== 单指标（y_norm，双向FE，按县聚类）==")
    rep = {"single": {}}
    for v in cand:
        sub = df[df[v].notna()]
        if float(sub[v].std()) == 0.0:
            print("  %-10s 全省该月无此事件（方差为 0），跳过" % v)
            continue
        r = reg(sub, v)
        b, p, ci = r["beta"][0], r["p"][0], r["ci"][0]
        star = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.1 else ""))
        print("  %-10s 均值%5.1f  β=%+.4f [%+.4f,%+.4f] p=%.4f %-3s withinR²=%.4f"
              % (v, sub[v].mean(), b, b - ci, b + ci, p, star, r["within_r2"]))
        rep["single"][v] = {"mean": round(float(sub[v].mean()), 2),
                            "beta": round(float(b), 4), "p": round(float(p), 4),
                            "ci": [round(float(b - ci), 4), round(float(b + ci), 4)],
                            "within_r2": round(r["within_r2"], 4)}

    # 联合：4月低温 + 5月干热风
    joint = ["fz4_0", "hd5_streak"]
    sub = df.dropna(subset=joint)
    ent = sub["county"].reset_index(drop=True)
    tim = sub["year"].reset_index(drop=True)
    dm = demean_two_way(sub[["y_norm"] + joint].reset_index(drop=True), ent, tim)
    r = cluster_ols(dm["y_norm"].values, dm[joint].values, ent, joint)
    print("\n== 联合模型（4月≤0℃ + 5月连续干热风）within R²=%.4f ==" % r["within_r2"])
    for row in fmt_model(r):
        print("  %-12s β=%+.4f [%+.4f,%+.4f] p=%.4f %s"
              % (row["变量"], row["系数"], row["95%CI下限"], row["95%CI上限"],
                 row["p值"], row["显著性"]))
    rep["joint"] = {"coef": fmt_model(r), "within_r2": round(r["within_r2"], 4), "n": r["n"]}

    # 边际效应换算：以 4 月低温 0→3 天为例
    b4 = rep["single"]["fz4_0"]["beta"]
    print("\n边际换算（4 月 ≤0℃）：0→2 天 %+.1f%%，0→5 天 %+.1f%%，0→8 天 %+.1f%%"
          % (b4 * 2 * 100, b4 * 5 * 100, b4 * 8 * 100))

    # 分区异质性
    cz = df.groupby("county").agg(base=("base", "first"),
                                  hd=("hd5_streak", "mean")).reset_index()
    ymed, hmed = cz["base"].median(), cz["hd"].median()

    def zone(r0):
        hi_y, hi_h = r0["base"] >= ymed, r0["hd"] >= hmed
        if hi_y and not hi_h: return "A 稳产核心区"
        if hi_y and hi_h: return "B 高产风险区"
        if (not hi_y) and (not hi_h): return "C 潜力提升区"
        return "D 脆弱敏感区"

    cz["zone"] = cz.apply(zone, axis=1)
    df = df.merge(cz[["county", "zone"]], on="county", how="left")
    print("\n== 分区：4月≤0℃ 弹性 ==")
    rep["zone"] = {}
    for z in sorted(df["zone"].dropna().unique()):
        sub = df[df["zone"] == z]
        if len(sub) < 40:
            continue
        r = reg(sub, "fz4_0")
        print("  %-10s n=%4d β=%+.4f [%+.4f,%+.4f] p=%.3f"
              % (z, r["n"], r["beta"][0], r["beta"][0] - r["ci"][0],
                 r["beta"][0] + r["ci"][0], r["p"][0]))
        rep["zone"][z] = {"n": r["n"], "beta": round(float(r["beta"][0]), 4),
                          "p": round(float(r["p"][0]), 4)}

    out = OUTDIR / "panel_window_v3.json"
    io.open(out, "w", encoding="utf-8").write(json.dumps(rep, ensure_ascii=False, indent=2))
    print("\n已写出：", out)


if __name__ == "__main__":
    main()
