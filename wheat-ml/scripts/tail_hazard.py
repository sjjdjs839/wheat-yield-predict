"""尾部效应诊断：灾害是否主要在"制造歉年"，而不是拉低平均单产。

如果年产量对灾害不敏感，是因为灾种作用在分布尾部（少数年份被打穿），
那么把因变量从"平均偏离"换成"歉年概率"，信号应当重新出现。
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from panel_impact import load_panel, run, fmt_model, BASE, OUTDIR

ev = pd.read_csv(BASE / "data" / "processed" / "disaster_events.csv")

df = load_panel().merge(
    ev[["county", "year", "fz4_0", "hd_may", "hd_streak"]], on=["county", "year"], how="left")

# 连阴雨指标复用已有脚本
sys.path.insert(0, str(Path(__file__).resolve().parent))
from diag_multihazard import flower_rain
df = df.merge(flower_rain(), on=["county", "year"], how="left")

print("样本 %d" % len(df))

# 歉年定义：本县单产低于常年 3% 以上
df["bad"] = (df["y_norm"] < 0.97).astype(int)
print("歉年占比 %.1f%%（%d 条）" % (df["bad"].mean() * 100, df["bad"].sum()))

# 描述：歉年 vs 非歉年的灾种强度
print("\n== 歉年 / 非歉年 的灾种强度对比 ==")
print("%-20s %10s %10s %8s" % ("变量", "歉年均值", "非歉年均值", "差值"))
for c in ["fz4_0", "hd_may", "hd_streak", "flower_rain_streak"]:
    a = df.loc[df.bad == 1, c].mean()
    b = df.loc[df.bad == 0, c].mean()
    print("%-20s %10.2f %10.2f %8.2f" % (c, a, b, a - b))

# 极歉年（最低 10%）
q10 = df["y_norm"].quantile(0.10)
print("\n== 极歉年（单产最低一成，阈值 %.3f）==" % q10)
for c in ["fz4_0", "hd_may", "hd_streak", "flower_rain_streak"]:
    a = df.loc[df.y_norm <= q10, c].mean()
    b = df.loc[df.y_norm > q10, c].mean()
    print("%-20s 极歉年%6.2f  其余%6.2f  差%+6.2f" % (c, a, b, a - b))

print("\n== 双向固定效应：灾种 -> 歉年概率（线性概率模型）==")
out = {"bad_share": float(df["bad"].mean()), "models": {}}
for xs, tag in [(["fz4_0"], "霜冻"), (["hd_may"], "干热风天数"),
                (["hd_streak"], "干热风连片"), (["flower_rain_streak"], "花期连阴雨"),
                (["fz4_0", "hd_streak", "flower_rain_streak"], "三灾种联合")]:
    r = run(df, "bad", xs, tag)
    rows = fmt_model(r)
    out["models"][tag] = {"n": r["n"], "coef": rows}
    print("\n[%s] n=%d" % (tag, r["n"]))
    for row in rows:
        print("   %-20s β=%+.5f [%+.5f, %+.5f] p=%.4f %s"
              % (row["变量"], row["系数"], row["95%CI下限"], row["95%CI上限"],
                 row["p值"], row["显著性"]))

json.dump(out, open(Path(OUTDIR) / "tail_hazard.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
print("\n写出", Path(OUTDIR) / "tail_hazard.json")
