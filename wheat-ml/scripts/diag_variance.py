"""方差分解：区分"年际共同波动（被年份固定效应吸收）"与"县际差异（可被识别）"。

用来解释为什么晚霜冻能估出减产弹性，而干热风/春旱/连阴雨估不出来。
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from panel_impact import load_panel, BASE

ev = pd.read_csv(BASE / "data" / "processed" / "disaster_events.csv")
df = load_panel().merge(
    ev[["county", "year", "fz4_0", "hd_may", "hd_streak"]], on=["county", "year"], how="left")

print("因素方差分解（n=%d 县年）" % len(df))
print("%-16s %10s %10s %10s %12s" % ("变量", "年际均值std", "同年县间std", "县均值std", "年际占比"))
for c in ["fz4_0", "hd_may", "hd_streak"]:
    v = df[c].astype(float)
    year_mean = df.groupby("year")[c].mean()
    within_year = df.groupby("year")[c].std().mean()
    county_mean = df.groupby("county")[c].mean()
    tot = v.std()
    print("%-16s %10.3f %10.3f %10.3f %11.1f%%"
          % (c, year_mean.std(), within_year, county_mean.std(), year_mean.std() / tot * 100))

print("\n按年均值与年份效应：")
g = df.groupby("year").agg(fz4=("fz4_0", "mean"), hd=("hd_may", "mean"),
                           hd_streak=("hd_streak", "mean"))
print(g.round(2).to_string())
print("\n峰值年 vs 谷年比值：")
for c in ["fz4_0", "hd_may"]:
    ym = df.groupby("year")[c].mean()
    print("  %-10s 最低%.2f(%d) 最高%.2f(%d) 极差%.2f"
          % (c, ym.min(), ym.idxmin(), ym.max(), ym.idxmax(), ym.max() - ym.min()))
