# -*- coding: utf-8 -*-
"""生成县级灾害事件表，预警页的数据底座。

口径上有个坑必须说清楚：3 月和 4 月的低温都按「相对该县常年」的距平算，
不是绝对天数。3 月 ≤0℃ 平均每年就有 6 天，是常态；拿绝对值折算的话，
每个县每年都会背一笔"减产"，等于把常态当灾害。

4 月不同，≤0℃ 是罕见事件（74% 的县年为零），距平≈绝对值，直接能用。

预估影响只算显著的那两个（3 月、4 月低温）。5 月干热风 p=0.21，不进减产，
只在页面上提示这一年干热风有多凶。
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

    # 年效应：该年全省单产相对常年的平均偏离（双向固定效应中被年份项吸收的部分）
    # β 只捕捉"同年内县与县的相对差异"；2018 年这种全省普遍偏冷的年份，
    # 共同影响落在年效应上，只报县距平会漏掉大范围年景的作用。
    ymean_by_year = df.groupby("year")["y_norm"].mean()
    grand_mean = df["y_norm"].mean()
    year_effect = ((ymean_by_year - grand_mean) * 100).round(2)
    ev["year_effect"] = ev["year"].map(year_effect)

    b3, b4 = coef["fz3_0"]["beta"], coef["fz4_0"]["beta"]
    lo3, hi3 = coef["fz3_0"]["ci_low"], coef["fz3_0"]["ci_high"]
    lo4, hi4 = coef["fz4_0"]["ci_low"], coef["fz4_0"]["ci_high"]

    ev["loss_pct"] = (b3 * ev["fz3_anom"] + b4 * ev["fz4_anom"]) * 100
    ev["loss_low"] = (min(lo3, hi3) * ev["fz3_anom"] + min(lo4, hi4) * ev["fz4_anom"]) * 100
    ev["loss_high"] = (max(lo3, hi3) * ev["fz3_anom"] + max(lo4, hi4) * ev["fz4_anom"]) * 100

    # 权威口径：距平折算。within 估计的系数度量的是"相对该县常年"的偏离，
    # 因此必须用距平天数，不能用绝对天数——否则会把该县常年本就有的低温日算成灾害
    # （如林州常年 4 月就有 2.67 天 ≤0℃，2018 年 4 天只比常年多 1.33 天）。
    ev["frost_anom_loss"] = (b4 * ev["fz4_anom"]) * 100
    ev["frost_anom_low"] = (min(lo4, hi4) * ev["fz4_anom"]) * 100
    ev["frost_anom_high"] = (max(lo4, hi4) * ev["fz4_anom"]) * 100

    # 反事实口径：若 4 月低温全部消失的潜在增益（相对"无霜冻情景"），
    # 仅用于"如果这场霜冻没发生"的推演，不能当作灾害损失展示。
    ev["frost_loss"] = (b4 * ev["fz4_0"]) * 100
    ev["frost_loss_low"] = (min(lo4, hi4) * ev["fz4_0"]) * 100
    ev["frost_loss_high"] = (max(lo4, hi4) * ev["fz4_0"]) * 100
    ev["cold3_loss"] = (b3 * ev["fz3_anom"]) * 100

    # 总影响 = 年效应（全省共同年景）+ 县距平（本县相对全省的差异）
    ev["total_loss"] = ev["year_effect"] + ev["frost_anom_loss"]
    ev["total_low"] = ev["year_effect"] + ev["frost_anom_low"]
    ev["total_high"] = ev["year_effect"] + ev["frost_anom_high"]

    for c in ["loss_pct", "loss_low", "loss_high",
              "frost_loss", "frost_loss_low", "frost_loss_high", "cold3_loss",
              "frost_anom_loss", "frost_anom_low", "frost_anom_high",
              "total_loss", "total_low", "total_high"]:
        ev[c] = ev[c].round(2)

    print("\n预估影响分布（%，距平口径，均值应≈0）:")
    print(ev["loss_pct"].describe(percentiles=[.05, .5, .95, .99]).round(2).to_string())
    print("\n晚霜冻距平折算（%，均值应≈0）:")
    print(ev["frost_anom_loss"].describe(percentiles=[.05, .5, .95, .99]).round(2).to_string())
    print("对照：绝对天数口径 frost_loss 均值=%.2f%%（偏高，不可作灾害损失展示）"
          % ev["frost_loss"].mean())

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
               "year_effect": {int(k): float(v) for k, v in year_effect.items()},
               "cuts": {"fz4_0": [0, 1, 3], "fz3_anom": [1, 3, 6], "hd_may": [0, 2, 6]},
               "note": ("loss_pct / frost_anom_loss 均为相对该县常年的距平折算，"
                        "与 within 估计口径一致；frost_loss 为反事实口径（4 月低温归零的潜在增益），"
                        "不是灾害损失；total_loss = 年效应（全省共同年景）+ 县距平，"
                        "两级分解不可混用；干热风 p=%.3f 不计入减产，仅作暴露提示"
                        % coef["hd_streak"]["p"])}
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
