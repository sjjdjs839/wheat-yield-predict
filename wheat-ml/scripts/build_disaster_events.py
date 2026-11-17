# -*- coding: utf-8 -*-
"""
build_disaster_events.py —— 生成县级灾害事件表（预警的数据底座）

口径约定（重要）
1. 3 月低温、4 月低温都是「相对该县常年」的距平，不是绝对天数。
   原因：3 月 ≤0℃ 平均每年 6 天，属于常态，若用绝对值折算，每个县每年都会被
   算出一笔"减产"，那是把常态当灾害。距平才符合"比常年多冷几天"的预警语义。
2. 4 月 ≤0℃ 属罕见事件（74% 的县年为零），距平≈绝对值，直接可用。
3. 预估影响只用显著因子（3 月、4 月低温）；5 月干热风 p=0.21，不计入减产，
   仅作「暴露强度」提示。
"""
import io, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
from panel_impact import load_panel, demean_two_way, cluster_ols, OUTDIR, BASE

WEATHER = BASE / "data" / "processed" / "weather_clean.csv"
OUT_CSV = BASE / "data" / "processed" / "disaster_events.csv"
COEF_JSON = OUTDIR / "disaster_coefficients.json"


def _max_streak(b: np.ndarray) -> int:
    best = cur = 0
    for v in b:
        cur = cur + 1 if v else 0
        if cur > best:
            best = cur
    return best


def build_raw() -> pd.DataFrame:
    w = pd.read_csv(
        WEATHER,
        usecols=["county", "city", "year", "month", "month_day", "t2m_max", "t2m_min", "rh"],
    ).sort_values(["county", "year", "month", "month_day"])

    rows = []
    for (c, y), g in w.groupby(["county", "year"], sort=False):
        mar = g[g["month"] == 3]
        apr = g[g["month"] == 4]
        may = g[g["month"] == 5]
        jun = g[(g["month"] == 6) & (g["month_day"] <= "06-10")]

        hd_may = ((may["t2m_max"] >= 35) & (may["rh"] <= 40))
        hd_jun = ((jun["t2m_max"] >= 35) & (jun["rh"] <= 40))

        rows.append({
            "county": c, "city": g["city"].iloc[0], "year": y,
            "fz3_0": int((mar["t2m_min"] <= 0).sum()),
            "fz4_0": int((apr["t2m_min"] <= 0).sum()),
            "fz4_streak": _max_streak((apr["t2m_min"] <= 0).to_numpy()),
            "tmin4": round(float(apr["t2m_min"].min()) if len(apr) else np.nan, 1),
            "hd_may": int(hd_may.sum()),
            "hd_streak": _max_streak(hd_may.to_numpy()),
            "hd_jun": int(hd_jun.sum()),
            "tmax5": round(float(may["t2m_max"].max()) if len(may) else np.nan, 1),
        })
    return pd.DataFrame(rows)


def fit_joint(df: pd.DataFrame, xs: list[str]) -> dict:
    sub = df.dropna(subset=xs + ["y_norm"]).copy()
    ent = sub["county"].reset_index(drop=True)
    tim = sub["year"].reset_index(drop=True)
    dm = demean_two_way(sub[["y_norm"] + xs].reset_index(drop=True), ent, tim)
    r = cluster_ols(dm["y_norm"].values, dm[xs].values, ent, xs)
    out = {"within_r2": round(r["within_r2"], 4), "n": r["n"]}
    for i, v in enumerate(xs):
        b, p, ci = float(r["beta"][i]), float(r["p"][i]), float(r["ci"][i])
        out[v] = {"beta": round(b, 5), "p": round(p, 5),
                  "ci_low": round(b - ci, 5), "ci_high": round(b + ci, 5)}
    return out


def main() -> None:
    print("重建灾害事件表…")
    ev = build_raw()
    print("  行数 %d，县 %d，年 %d—%d" % (len(ev), ev["county"].nunique(),
                                        ev["year"].min(), ev["year"].max()))

    df = load_panel().merge(ev, on=["county", "year"], how="left")
    xs = ["fz3_0", "fz4_0", "hd_streak"]
    coef = fit_joint(df, xs)
    print("\n联合模型（3月低温 + 4月低温 + 5月连续干热风）within R²=%.4f  n=%d"
          % (coef["within_r2"], coef["n"]))
    for v in xs:
        c = coef[v]
        print("  %-10s β=%+.5f [%+.5f,%+.5f] p=%.4f" % (v, c["beta"], c["ci_low"], c["ci_high"], c["p"]))

    # 距平：相对该县常年
    ev["base3"] = ev.groupby("county")["fz3_0"].transform("mean").round(1)
    ev["base4"] = ev.groupby("county")["fz4_0"].transform("mean").round(2)
    ev["fz3_anom"] = (ev["fz3_0"] - ev["base3"]).round(1)
    ev["fz4_anom"] = (ev["fz4_0"] - ev["base4"]).round(2)

    b3, b4 = coef["fz3_0"]["beta"], coef["fz4_0"]["beta"]
    lo3, hi3 = coef["fz3_0"]["ci_low"], coef["fz3_0"]["ci_high"]
    lo4, hi4 = coef["fz4_0"]["ci_low"], coef["fz4_0"]["ci_high"]

    ev["loss_pct"] = (b3 * ev["fz3_anom"] + b4 * ev["fz4_anom"]) * 100
    ev["loss_low"] = (min(lo3, hi3) * ev["fz3_anom"] + min(lo4, hi4) * ev["fz4_anom"]) * 100
    ev["loss_high"] = (max(lo3, hi3) * ev["fz3_anom"] + max(lo4, hi4) * ev["fz4_anom"]) * 100

    # 拆开供页面分别展示：晚霜冻按绝对天数（以"4 月无 ≤0℃"为基准），返青期低温按距平
    ev["frost_loss"] = (b4 * ev["fz4_0"]) * 100
    ev["frost_loss_low"] = (min(lo4, hi4) * ev["fz4_0"]) * 100
    ev["frost_loss_high"] = (max(lo4, hi4) * ev["fz4_0"]) * 100
    ev["cold3_loss"] = (b3 * ev["fz3_anom"]) * 100

    for c in ["loss_pct", "loss_low", "loss_high",
              "frost_loss", "frost_loss_low", "frost_loss_high", "cold3_loss"]:
        ev[c] = ev[c].round(2)

    print("\n预估影响分布（%，距平口径，均值应≈0）:")
    print(ev["loss_pct"].describe(percentiles=[.05, .5, .95, .99]).round(2).to_string())

    # 分级：4 月用绝对天数（0 为常态）；3 月用距平
    ev["frost_level"] = ev["fz4_0"].map(
        lambda v: 0 if v == 0 else (1 if v == 1 else (2 if v <= 3 else 3))).astype(int)
    ev["cold3_level"] = ev["fz3_anom"].map(
        lambda v: 0 if v <= 1 else (1 if v <= 3 else (2 if v <= 6 else 3))).astype(int)
    ev["hd_level"] = ev["hd_may"].map(
        lambda v: 0 if v == 0 else (1 if v <= 2 else (2 if v <= 6 else 3))).astype(int)

    print("\n4 月低温（晚霜冻）县年分布：", ev["fz4_0"].value_counts().sort_index().to_dict())
    print("3 月低温距平分布（分位）：%s" % ev["fz3_anom"].quantile([.5, .75, .95]).round(1).to_dict())

    ev.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print("\n已写出：", OUT_CSV)

    payload = {"joint": coef,
               "cuts": {"fz4_0": [0, 1, 3], "fz3_anom": [1, 3, 6], "hd_may": [0, 2, 6]},
               "note": "loss_pct 为相对该县常年的距平折算；干热风 p=%.3f 不计入减产，仅作暴露提示"
                       % coef["hd_streak"]["p"]}
    io.open(COEF_JSON, "w", encoding="utf-8").write(json.dumps(payload, ensure_ascii=False, indent=2))
    print("已写出：", COEF_JSON)

    print("\n== 抽样年份：4 月低温 ≥1 天的县（晚霜冻实况）==")
    for y in sorted(ev["year"].unique())[-9:]:
        s = ev[(ev["year"] == y) & (ev["fz4_0"] > 0)].sort_values("fz4_0", ascending=False)
        if len(s) == 0:
            print("  %d 年：无县出现 4 月 ≤0℃" % y)
            continue
        top = s.head(5)
        print("  %d 年：%d 县 | %s" % (
            y, len(s),
            "、".join("%s %d天 霜冻%.1f%% 最低%.1f℃" % (r.county, r.fz4_0, r.frost_loss, r.tmin4)
                     for r in top.itertuples())))

    # 2018 大范围倒春寒的口径核查
    s18 = ev[(ev["year"] == 2018) & (ev["fz4_0"] > 0)]
    print("\n2018 年核查：%d 县有 4 月 ≤0℃，天数分布 %s，最低温中位 %.1f℃"
          % (len(s18), s18["fz4_0"].value_counts().sort_index().to_dict(),
             s18["tmin4"].median()))


if __name__ == "__main__":
    main()
