# -*- coding: utf-8 -*-
"""验证：10 月温度是否独立于 cold_days（用于修正说明书附 5.1 的表述）

复用 scripts.panel_overwinter 的建表与回归口径（双向固定效应 + 县聚类标准误）。
运行：python -m scripts.verify_sowing_temp
"""
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd

from panel_overwinter import load_panel, run
from panel_impact import fmt_model

OUT = ROOT / "output" / "reports"


def main():
    df = load_panel()
    if "y_norm" not in df.columns:
        print("y_norm 缺失，列：", [c for c in df.columns if c.startswith(("sow", "y_"))])
        return

    cases = [
        (["sow_tmin_min"], "10 月最低温极值（单变量）"),
        (["sow_tavg"], "10 月均温（单变量）"),
        (["sow_precip"], "10 月降水（单变量）"),
        (["sow_dry_days"], "10 月干燥日数（单变量）"),
        (["sow_tmin_min", "cold_days"], "10 月最低温极值 + cold_days"),
        (["sow_tavg", "cold_days"], "10 月均温 + cold_days"),
        (["sow_tmin_min", "sow_tavg"], "10 月最低温 + 均温"),
    ]
    results = {}
    for xs, label in cases:
        print("\n=== %s ===" % label)
        r = run(df, "y_norm", xs, label)
        rows = fmt_model(r)
        for row in rows:
            print("  %-14s β=%s  t=%s  p=%s" % (row["变量"], row["系数"], row["t"], row["p值"]))
        results[label] = rows
        print("  within R2 =", r.get("within_r2"), " n =", r.get("n"))

    (OUT / "sowing_temp_check.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n已写出", OUT / "sowing_temp_check.json")


if __name__ == "__main__":
    main()
