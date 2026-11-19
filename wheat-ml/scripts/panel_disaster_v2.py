# -*- coding: utf-8 -*-
"""灾害指标的口径验证。不写主特征表，先验一遍，够格才考虑并进去。

现在只用 5 月总天数一个口径。这里从逐日原始数据重建一批候选：

  hd_may       5 月干热风天数（现行口径，拿来做对照）
  hd_win       5/1—6/10 干热风天数（灌浆 + 成熟前期，窗口放宽）
  hd_streak    W2 内最长连续干热风天数（看事件持续多长）
  hd_event3    W2 内"连续 ≥3 天"的事件次数
  hd_excess    W2 内 Σ max(tmax-35, 0)，只对干热风日累加（热强度）
  hdw_win      W2 内高温+低湿+大风(≥3m/s) 的复合干热风天数（对标 HDW 文献）
  frost_apr    4 月最低温 ≤0℃ 天数（更贴晚霜冻）
  frost_32     3—4 月最低温 ≤2℃ 天数（宽口径晚霜）

每套指标单独进一次双向固定效应，看系数方向和显著性。
"""
import io, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
from panel_impact import (  # 复用面板工具
    load_panel, demean_two_way, cluster_ols, fmt_model, OUTDIR,
)

BASE = SCRIPTS.parent
WEATHER = BASE / "data" / "processed" / "weather_clean.csv"

W2_MONTHS = [5, 6]
W2_DAY_CUTOFF = 10  # 6 月只取到 10 日


def _max_streak(b: np.ndarray) -> int:
    best = cur = 0
    for v in b:
        cur = cur + 1 if v else 0
        if cur > best:
            best = cur
    return best


def _n_event_ge3(b: np.ndarray) -> int:
    """连续 ≥3 天的 True 段计数"""
    cnt = run = 0
    for v in b:
        if v:
            run += 1
        else:
            if run >= 3:
                cnt += 1
            run = 0
    if run >= 3:
        cnt += 1
    return cnt


def build_disaster_table() -> pd.DataFrame:
    w = pd.read_csv(
        WEATHER,
        usecols=["county", "year", "month", "month_day", "t2m_max", "t2m_min", "rh", "wind"],
    )

    hd_flag = (w["t2m_max"] >= 35) & (w["rh"] <= 40)
    hdw_flag = hd_flag & (w["wind"] >= 3)

    # 现行口径：5 月
    may = w[w["month"] == 5].copy()
    may["hd"] = hd_flag.loc[may.index]
    t_may = may.groupby(["county", "year"]).agg(hd_may=("hd", "sum")).reset_index()

    # 扩展窗口：5 月 + 6 月前 10 天
    w2 = w[(w["month"] == 5) | ((w["month"] == 6) & (w["month_day"] <= "06-%02d" % W2_DAY_CUTOFF))].copy()
    w2["hd"] = hd_flag.loc[w2.index]
    w2["hdw"] = hdw_flag.loc[w2.index]
    w2["excess"] = np.where(w2["hd"], np.clip(w2["t2m_max"] - 35, 0, None), 0.0)
    w2 = w2.sort_values(["county", "year", "month", "month_day"])

    rows = []
    for (c, y), g in w2.groupby(["county", "year"], sort=False):
        f = g["hd"].to_numpy()
        rows.append({
            "county": c, "year": y,
            "hd_win": int(f.sum()),
            "hd_streak": _max_streak(f),
            "hd_event3": _n_event_ge3(f),
            "hd_excess": float(g["excess"].sum()),
            "hdw_win": int(g["hdw"].sum()),
        })
    t_w2 = pd.DataFrame(rows)

    # 晚霜口径
    apr = w[(w["month"] == 4)].copy()
    apr["fr"] = (apr["t2m_min"] <= 0).astype(int)
    t_apr = apr.groupby(["county", "year"]).agg(frost_apr=("fr", "sum")).reset_index()

    mar_apr = w[(w["month"].isin([3, 4]))].copy()
    mar_apr["fr"] = (mar_apr["t2m_min"] <= 2).astype(int)
    t_34 = mar_apr.groupby(["county", "year"]).agg(frost_32=("fr", "sum")).reset_index()

    out = t_may.merge(t_w2, on=["county", "year"], how="outer")
    out = out.merge(t_apr, on=["county", "year"], how="outer")
    out = out.merge(t_34, on=["county", "year"], how="outer")
    return out


def main() -> None:
    print("构建灾害指标（从逐日数据）…")
    dis = build_disaster_table()
    print("灾害表：%d 行，%d 县" % (len(dis), dis["county"].nunique()))

    df = load_panel()
    df = df.merge(dis, on=["county", "year"], how="left")

    cand = ["hd_may", "hd_win", "hd_streak", "hd_event3", "hd_excess", "hdw_win",
            "frost_apr", "frost_32"]

    print("\n描述统计（灾害指标）：")
    print(df[cand].describe().loc[["mean", "std", "min", "max"]].round(3).to_string())

    report = {"metrics": {}}
    print("\n== 单指标面板回归（y_norm，县+年双向固定效应，按县聚类）==")
    for v in cand:
        sub = df[df[v].notna()]
        ent = sub["county"].reset_index(drop=True)
        tim = sub["year"].reset_index(drop=True)
        dm = demean_two_way(sub[["y_norm", v]].reset_index(drop=True), ent, tim)
        r = cluster_ols(dm["y_norm"].values, dm[[v]].values, ent, [v])
        b, se, p, ci = r["beta"][0], r["se"][0], r["p"][0], r["ci"][0]
        row = {"beta": round(float(b), 4), "se": round(float(se), 4),
               "p": round(float(p), 4),
               "ci": [round(float(b - ci), 4), round(float(b + ci), 4)],
               "within_r2": round(r["within_r2"], 4), "n": r["n"]}
        report["metrics"][v] = row
        star = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.1 else ""))
        print("  %-11s β=%+.4f [%+.4f,%+.4f]  p=%.4f %-3s withinR²=%.4f"
              % (v, b, b - ci, b + ci, p, star, r["within_r2"]))

    # 联合模型：把显著项放在一起
    joint = ["hd_streak", "hd_excess", "frost_apr"]
    sub = df.dropna(subset=joint + ["y_norm"])
    ent = sub["county"].reset_index(drop=True)
    tim = sub["year"].reset_index(drop=True)
    dm = demean_two_way(sub[["y_norm"] + joint].reset_index(drop=True), ent, tim)
    r = cluster_ols(dm["y_norm"].values, dm[joint].values, ent, joint)
    print("\n== 联合模型（hd_streak + hd_excess + frost_apr）within R²=%.4f ==" % r["within_r2"])
    for row in fmt_model(r):
        print("  %-11s β=%+.4f [%+.4f,%+.4f] p=%.4f %s"
              % (row["变量"], row["系数"], row["95%CI下限"], row["95%CI上限"],
                 row["p值"], row["显著性"]))
    report["joint"] = {"coef": fmt_model(r), "within_r2": round(r["within_r2"], 4), "n": r["n"]}

    out = OUTDIR / "panel_disaster_v2.json"
    io.open(out, "w", encoding="utf-8").write(json.dumps(report, ensure_ascii=False, indent=2))
    print("\n已写出：", out)


if __name__ == "__main__":
    main()
