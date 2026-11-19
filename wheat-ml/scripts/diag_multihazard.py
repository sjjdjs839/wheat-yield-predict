"""多灾种一起进面板：干热风、春旱、花期连阴雨、晚霜冻，同一套双向固定效应。

主要想回答答辩会被问到的那句"凭什么只给低温折算减产"。用 p 值说话，
不靠印象。
"""
import sys, json
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from panel_impact import load_panel, run, fmt_model, OUTDIR, BASE

W = BASE / "data" / "processed" / "weather_clean.csv"
EV = BASE / "data" / "processed" / "disaster_events.csv"


def flower_rain() -> pd.DataFrame:
    """花期（4/11—5/5）连阴雨指标：雨日数 + 最长连续雨日。"""
    df = pd.read_csv(W, usecols=["county", "year", "month", "month_day", "precip"])
    m = ((df["month"] == 4) & (df["month_day"] >= "04-11")) | ((df["month"] == 5) & (df["month_day"] <= "05-05"))
    d = df[m].copy()
    d["is_rain"] = (d["precip"].fillna(0) >= 0.1).astype(int)

    rows = []
    for (c, y), g in d.sort_values(["county", "year", "month", "month_day"]).groupby(["county", "year"]):
        rain = g["is_rain"].values
        best = cur = 0
        for v in rain:
            cur = cur + 1 if v else 0
            best = max(best, cur)
        rows.append({"county": c, "year": y,
                     "flower_rain_days": int(rain.sum()),
                     "flower_rain_streak": int(best)})
    return pd.DataFrame(rows)


def main():
    df = load_panel()
    ev = pd.read_csv(EV)
    fr = flower_rain()

    df = df.merge(ev[["county", "year", "fz3_0", "fz4_0", "hd_may", "hd_streak", "hd_jun"]],
                  on=["county", "year"], how="left")
    df = df.merge(fr, on=["county", "year"], how="left")

    print("样本 %d 条；连阴雨指标覆盖 %d 条" % (len(df), df["flower_rain_streak"].notna().sum()))

    # 分布：有没有足够变异
    for c in ["fz4_0", "hd_may", "hd_streak", "spi_12", "precip_anom",
              "flower_rain_days", "flower_rain_streak"]:
        if c in df:
            v = df[c].dropna()
            print("  %-18s 均值%7.2f  标准差%7.2f  非零占比%5.1f%%"
                  % (c, v.mean(), v.std(), (v > 0).mean() * 100))

    out = {"models": {}}

    def show(tag, xs):
        r = run(df, "y_norm", xs, tag)
        rows = fmt_model(r)
        out["models"][tag] = {"n": r["n"], "within_r2": round(r["within_r2"], 4), "coef": rows}
        print("\n[%s]  within R²=%.4f  n=%d" % (tag, r["within_r2"], r["n"]))
        for row in rows:
            print("   %-20s β=%+.4f  [%+.4f, %+.4f]  p=%.4f %s"
                  % (row["变量"], row["系数"], row["95%CI下限"], row["95%CI上限"], row["p值"], row["显著性"]))

    print("\n== 单灾种（各自独立进模型）==")
    show("alone|晚霜冻4月", ["fz4_0"])
    show("alone|干热风天数5月", ["hd_may"])
    show("alone|干热风连片", ["hd_streak"])
    show("alone|春旱(降水距平)", ["precip_anom"])
    show("alone|春旱(SPI12)", ["spi_12"])
    show("alone|花期雨日", ["flower_rain_days"])
    show("alone|花期连阴雨最长", ["flower_rain_streak"])

    print("\n== 联合（控制晚霜冻后，其它灾种还剩多少）==")
    show("joint|霜冻+干热风+春旱", ["fz4_0", "hd_streak", "precip_anom"])
    show("joint|霜冻+干热风+连阴雨", ["fz4_0", "hd_streak", "flower_rain_streak"])
    show("joint|四灾种全进", ["fz4_0", "hd_streak", "precip_anom", "flower_rain_streak"])

    json.dump(out, open(Path(OUTDIR) / "multihazard.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("\n写出", Path(OUTDIR) / "multihazard.json")


if __name__ == "__main__":
    main()
