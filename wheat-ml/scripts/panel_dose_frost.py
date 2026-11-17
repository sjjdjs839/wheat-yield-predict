"""4 月 / 3 月低温的剂量响应：低温天数分档后，看该县当年单产相对常年的偏离。

方法：与面板一致，先剔除县固定效应与年份固定效应，再在残差空间按低温天数分档，
得到"同样年景下，多一天低温意味着少收多少"。这给出预警线应卡在哪里。
"""
import sys, json
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from panel_impact import load_panel, demean_two_way, OUTDIR, BASE


def banded(df, col, edges, labels):
    b = pd.cut(df[col], bins=edges, labels=labels, right=False)
    g = df.groupby(b, observed=True).agg(
        n=("resid", "size"), dev=("resid", "mean"), sd=("resid", "std"))
    g["dev_pct"] = (g["dev"] * 100).round(2)
    g["se_pct"] = (g["sd"] / np.sqrt(g["n"]) * 100).round(2)
    g = g[g["n"] >= 5].copy()
    zero = g["dev_pct"].iloc[0]
    g["vs0"] = (g["dev_pct"] - zero).round(2)
    g["edge"] = [labels[list(g.index).index(i)] for i in g.index]
    return g


def main():
    df = load_panel()
    ev = pd.read_csv(Path(BASE) / "data" / "processed" / "disaster_events.csv")
    df = df.merge(ev[["county", "year", "fz3_0", "fz4_0", "base3", "base4"]],
                  on=["county", "year"], how="left")

    dm = demean_two_way(df[["y_norm"]], df["county"], df["year"])
    df["resid"] = dm["y_norm"].values

    out = {"note": "已剔除县与年份固定效应；偏离% = 相对该县常年的百分点；vs0 = 相对最低档的增量",
           "n_total": int(len(df))}

    plan = {
        "fz4": ("4 月 ≤0℃ 天数", df["fz4_0"], [-0.1, 0.5, 1.5, 2.5, 4.5, 100],
                ["0 天", "1 天", "2 天", "3—4 天", "≥5 天"]),
        "fz3": ("3 月 ≤0℃ 天数", df["fz3_0"], [-0.1, 2.5, 5.5, 8.5, 100],
                ["0—2 天", "3—5 天", "6—8 天", "≥9 天"]),
    }

    for key, (label, s, edges, labs) in plan.items():
        d = df.copy()
        d[s.name] = s.values
        g = banded(d, s.name, edges, labs)
        out[key] = {"label": label,
                    "bands": json.loads(g.reset_index(drop=True).to_json(orient="records", force_ascii=False))}
        print("\n%s" % label)
        print(g[["edge", "n", "dev_pct", "se_pct", "vs0"]].to_string(index=False))

    json.dump(out, open(Path(OUTDIR) / "dose_frost.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("\n写出", Path(OUTDIR) / "dose_frost.json")


if __name__ == "__main__":
    main()
