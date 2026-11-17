"""晚霜冻弹性的稳健性检验：分段、逐剔除、留一县。

回答"这个模型靠不靠得住"——如果系数在各种切法下都稳定，说明它不是拟合出来的偶然。
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from panel_impact import load_panel, run, fmt_model, BASE, OUTDIR


def main():
    df = load_panel()
    ev = pd.read_csv(BASE / "data" / "processed" / "disaster_events.csv")
    df = df.merge(ev[["county", "year", "fz4_0"]], on=["county", "year"], how="left")

    out = {}

    def one(tag, d):
        r = run(d, "y_norm", ["fz4_0"], tag)
        c = fmt_model(r)[0]
        rec = {"n": r["n"], "beta": c["系数"], "lo": c["95%CI下限"], "hi": c["95%CI上限"],
               "p": c["p值"], "r2": round(r["within_r2"], 4)}
        print("%-22s n=%-5d β=%+.4f [%+.4f, %+.4f] p=%.4f  withinR²=%.4f"
              % (tag, rec["n"], rec["beta"], rec["lo"], rec["hi"], rec["p"], rec["r2"]))
        return rec

    print("== 全样本 ==")
    out["all"] = one("全样本 2010-2024", df)

    print("\n== 分两段（时间外推）==")
    out["seg1"] = one("前段 2010-2016", df[df.year <= 2016])
    out["seg2"] = one("后段 2017-2024", df[df.year >= 2017])

    print("\n== 逐年剔除（leave-one-year-out）==")
    loo = []
    for y in sorted(df.year.unique()):
        rec = one("剔除 %d" % y, df[df.year != y])
        loo.append({"drop_year": int(y), **rec})
    out["loo"] = loo
    betas = [x["beta"] for x in loo]
    print("\nLOO β 范围: %.4f ~ %.4f（全样本 %.4f），极差 %.4f"
          % (min(betas), max(betas), out["all"]["beta"], max(betas) - min(betas)))

    print("\n== 留一县（每次剔除一个县）==")
    cb = []
    for c in sorted(df.county.unique()):
        d = df[df.county != c]
        try:
            r = run(d, "y_norm", ["fz4_0"], c)
            cb.append(fmt_model(r)[0]["系数"])
        except Exception:
            pass
    print("留一县 β 范围: %.4f ~ %.4f（%d 个县依次剔除）" % (min(cb), max(cb), len(cb)))
    out["leave_county"] = {"min": min(cb), "max": max(cb), "n_county": len(cb)}

    json.dump(out, open(Path(OUTDIR) / "robustness_frost.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("\n写出", Path(OUTDIR) / "robustness_frost.json")


if __name__ == "__main__":
    main()
