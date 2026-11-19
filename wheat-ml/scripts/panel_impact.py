# -*- coding: utf-8 -*-
"""气象灾害对河南小麦产量的影响，双向固定效应面板。

几个设计上的决定：

被解释变量给两套，同时出、互相印证。y_norm 是当年单产除以该县常年单产，
无量纲，系数直接读成百分点；y_raw 是原始单产 kg/亩，系数就是 kg/亩每天。

固定效应挂县和年两个，用交替投影迭代去均值，非平衡面板也能吃。

标准误按县聚类（CR1 小样本校正）。同一县连着好几年，观测本来就不独立，
不聚类 t 值会虚高。

剂量响应在去均值后的残差空间里按灾害强度分档取均值，得到"控制住县和
年份之后"的曲线，比直接画原始散点靠谱。

分区异质性按 A/B/C/D 四区各估一次弹性，看脆弱区是不是真的更敏感。

不用 linearmodels 也不用 statsmodels，numpy + scipy 手撸的，免得为了一个
回归去动已经跑通的环境。
"""
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st

sys.stdout.reconfigure(encoding="utf-8")

BASE = Path(__file__).resolve().parents[1]
FEAT = BASE / "data" / "processed" / "feature_table.csv"
STATS = BASE / "output" / "models" / "county_stats.json"
OUTDIR = BASE / "output" / "reports"
OUTDIR.mkdir(parents=True, exist_ok=True)

# 灾害指标（均为已有字段，无需补数据）
DISASTERS = {
    "hot_dry_days": "灌浆期干热风（天，5月≥35℃&RH≤40%）",
    "hot_days": "全生育期高温（天，3—5月≥35℃）",
    "cold_days": "低温/霜冻（天，3—5月最低温<0℃）",
    "precip_anom": "降水距平（比例，相对县内常年）",
}


# ---------- 数据 ----------
def load_panel() -> pd.DataFrame:
    df = pd.read_csv(FEAT)
    raw = json.load(io.open(STATS, encoding="utf-8"))
    df["base"] = df["county"].map(lambda c: raw[c]["mean_yield"] if c in raw else np.nan)
    df = df[df["wheat_yield"].notna() & df["base"].notna()].copy()
    df["y_norm"] = df["wheat_yield"] / df["base"]
    df["y_raw"] = df["wheat_yield"].astype(float)
    df["county"] = df["county"].astype(str)
    df["year"] = df["year"].astype(int)
    return df


def demean_two_way(mat: pd.DataFrame, ent: pd.Series, tim: pd.Series,
                   iters: int = 60, tol: float = 1e-11) -> pd.DataFrame:
    """双向固定效应去均值（交替投影法），非平衡面板也收敛。"""
    out = mat.astype(float).copy()
    for _ in range(iters):
        prev = out.values.copy()
        out = out - out.groupby(ent).transform("mean")
        out = out - out.groupby(tim).transform("mean")
        if np.max(np.abs(out.values - prev)) < tol:
            break
    return out


def cluster_ols(y: np.ndarray, X: np.ndarray, groups: pd.Series, names: list[str]) -> dict:
    """无截距 OLS + 按实体聚类稳健标准误（CR1）。"""
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    k = X.shape[1]
    meat = np.zeros((k, k))
    grp = groups.reset_index(drop=True).values
    for gval in pd.unique(grp):
        pos = grp == gval
        s = X[pos].T @ resid[pos]
        meat += np.outer(s, s)
    G = groups.nunique()
    N = X.shape[0]
    V = XtX_inv @ meat @ XtX_inv
    V *= (G / (G - 1)) * ((N - 1) / (N - k))
    se = np.sqrt(np.diag(V))
    tval = beta / se
    pval = 2 * st.t.sf(np.abs(tval), df=G - 1)
    tcrit = st.t.ppf(0.975, df=G - 1)
    rss = float(resid @ resid)
    tss = float(y @ y)
    within_r2 = 1 - rss / tss if tss > 0 else float("nan")
    return {
        "beta": beta, "se": se, "p": pval, "ci": tcrit * se,
        "within_r2": within_r2, "n": int(N), "g": int(G), "k": k,
        "names": names,
    }


def fmt_model(res: dict) -> list[dict]:
    rows = []
    for i, nm in enumerate(res["names"]):
        b, se, p, ci = res["beta"][i], res["se"][i], res["p"][i], res["ci"][i]
        stars = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.1 else ""))
        rows.append({
            "变量": nm, "系数": round(float(b), 4), "标准误": round(float(se), 4),
            "t": round(float(b / se), 2), "p值": round(float(p), 4),
            "95%CI下限": round(float(b - ci), 4), "95%CI上限": round(float(b + ci), 4),
            "显著性": stars,
        })
    return rows


def run(df: pd.DataFrame, depend: str, xs: list[str], tag: str) -> dict:
    ent = df["county"].reset_index(drop=True)
    tim = df["year"].reset_index(drop=True)
    dm = demean_two_way(df[[depend] + xs].reset_index(drop=True), ent, tim)
    res = cluster_ols(dm[depend].values, dm[xs].values, ent, xs)
    res["_tag"] = tag
    res["_dep"] = depend
    return res


def main() -> None:
    df = load_panel()
    print("样本：%d 个县 × %d 个年份 = %d 条观测（%d—%d）"
          % (df["county"].nunique(), df["year"].nunique(), len(df),
             df["year"].min(), df["year"].max()))
    print("县内年数：最少 %d / 中位 %d / 最多 %d"
          % (df.groupby("county")["year"].count().min(),
             int(df.groupby("county")["year"].count().median()),
             df.groupby("county")["year"].count().max()))
    print("相对常年单产的县内均值 %.4f（≈1 说明口径正确，固定效应已吸收县际差异）\n"
          % df.groupby("county")["y_norm"].mean().mean())

    report = {"sample": {"n": len(df), "counties": int(df["county"].nunique()),
                         "years": [int(df["year"].min()), int(df["year"].max())]},
              "models": {}}

    # 单灾害模型：逐个识别，避免高温天数与干热风天数互相吸收
    for var, label in DISASTERS.items():
        for dep in ["y_norm", "y_raw"]:
            r = run(df, dep, [var], "%s|%s" % (var, dep))
            key = "%s|%s" % (var, dep)
            report["models"][key] = {"label": label, "dep": dep,
                                     "within_r2": round(r["within_r2"], 4),
                                     "n": r["n"], "g": r["g"],
                                     "coef": fmt_model(r)}
            print("[%s] %s → %s  β=%+.4f  p=%.4f  within R²=%.4f"
                  % (dep, label[:14], var, r["beta"][0], r["p"][0], r["within_r2"]))

    # 联合模型（注意：hot_days 与 hot_dry_days 存在包含关系，仅作稳健性参考）
    joint = ["hot_dry_days", "cold_days", "precip_anom"]
    for dep in ["y_norm", "y_raw"]:
        r = run(df, dep, joint, "joint|%s" % dep)
        key = "joint|%s" % dep
        report["models"][key] = {"label": "联合模型（干热风+低温+降水）", "dep": dep,
                                 "within_r2": round(r["within_r2"], 4),
                                 "n": r["n"], "g": r["g"], "coef": fmt_model(r)}
        print("\n联合模型（%s）within R²=%.4f" % (dep, r["within_r2"]))
        for row in fmt_model(r):
            print("   %-14s β=%+.4f  [%+.4f, %+.4f]  p=%.4f %s"
                  % (row["变量"], row["系数"], row["95%CI下限"], row["95%CI上限"],
                     row["p值"], row["显著性"]))

    # 剂量响应：残差空间分档（控制县与年份之后）
    dm = demean_two_way(df[["y_norm"]], df["county"], df["year"])
    y_res = dm["y_norm"].values
    bins = [-0.1, 0.5, 2.5, 6.5, 9.5, 100]
    labels = ["0 天", "1—2 天", "3—6 天", "7—9 天", "≥10 天"]
    band = pd.cut(df["hot_dry_days"], bins=bins, labels=labels)
    dose = pd.DataFrame({"band": band, "y": y_res, "hd": df["hot_dry_days"].values})
    g = dose.groupby("band", observed=True).agg(
        n=("y", "size"), 偏离=("y", "mean"), 标准差=("y", "std"))
    g["偏离%"] = (g["偏离"] * 100).round(2)
    g["标准误%"] = ((g["标准差"] / np.sqrt(g["n"])) * 100).round(2)
    g["相对0天档"] = (g["偏离%"] - g.loc["0 天", "偏离%"]).round(2)
    dose_csv = OUTDIR / "panel_dose_response.csv"
    g.to_csv(dose_csv, encoding="utf-8-sig")
    print("\n干热风剂量响应（已剔除县与年份效应）：")
    print(g[["n", "偏离%", "标准误%", "相对0天档"]].to_string())
    report["dose_response"] = json.loads(g.reset_index().to_json(orient="records", force_ascii=False))

    # 分区异质性：单产中位数 × 干热风暴露中位数 四象限
    cz = df.groupby("county").agg(base=("base", "first"), hd=("hot_dry_days", "mean")).reset_index()
    ymed, hmed = cz["base"].median(), cz["hd"].median()

    def zone(r):
        hi_y, hi_h = r["base"] >= ymed, r["hd"] >= hmed
        if hi_y and not hi_h: return "A 稳产核心区"
        if hi_y and hi_h: return "B 高产风险区"
        if (not hi_y) and (not hi_h): return "C 潜力提升区"
        return "D 脆弱敏感区"

    cz["zone"] = cz.apply(zone, axis=1)
    df = df.merge(cz[["county", "zone"]], on="county", how="left")
    het = {}
    print("\n分区干热风弹性（y_norm 口径）：")
    for zname in sorted(df["zone"].dropna().unique()):
        sub = df[df["zone"] == zname]
        if len(sub) < 40:
            continue
        r = run(sub, "y_norm", ["hot_dry_days"], "het|%s" % zname)
        het[zname] = {"n": r["n"], "beta": round(float(r["beta"][0]), 4),
                      "p": round(float(r["p"][0]), 4),
                      "ci": [round(float(r["beta"][0] - r["ci"][0]), 4),
                             round(float(r["beta"][0] + r["ci"][0]), 4)],
                      "within_r2": round(r["within_r2"], 4)}
        print("   %-10s n=%4d  β=%+.4f  95%%CI[%+.4f, %+.4f]  p=%.3f"
              % (zname, r["n"], r["beta"][0], r["beta"][0] - r["ci"][0],
                 r["beta"][0] + r["ci"][0], r["p"][0]))
    report["zone_heterogeneity"] = het

    out_json = OUTDIR / "panel_impact.json"
    io.open(out_json, "w", encoding="utf-8").write(
        json.dumps(report, ensure_ascii=False, indent=2))
    print("\n已写出：", out_json)
    print("已写出：", dose_csv)


if __name__ == "__main__":
    main()
