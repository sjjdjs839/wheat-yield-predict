"""
气象数据分析与可视化（对齐项目方案 2.2）
==========================================
七类图表：
  1. 最高温 / 最低温 / 平均温变化趋势
  2. 日降水量变化
  3. 2m 风速变化
  4. 地表短波辐射变化
  5. 露点温度 + 相对湿度变化
  6. 不同年份 / 时间段的月度对比
  7. 气象条件综合评估（年度指标 + 与常年对比）

图表以 PNG(base64) 返回，直接内嵌到 Web 页面，无外网依赖。
"""

import base64
import io
import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.facecolor"] = "white"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
with open(PROJECT_ROOT / "config" / "config.yaml", "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

COLORS = {"tmax": "#e53935", "tmin": "#1e88e5", "tmean": "#43a047",
          "precip": "#29b6f6", "wind": "#8e24aa", "radiation": "#fb8c00",
          "dew": "#00acc1", "rh": "#6d4c41"}


def _fig_to_uri(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()


def _empty(msg: str) -> dict:
    return {"img": "", "title": msg, "desc": "当前数据缺少该字段，无法绘制。"}


def _sub(df: pd.DataFrame, year: int | None, month_from: int | None,
         month_to: int | None) -> pd.DataFrame:
    d = df
    if year:
        d = d[d["year"] == year]
    if month_from or month_to:
        lo = month_from or 1
        hi = month_to or 12
        d = d[d["month"].between(lo, hi)]
    return d


def chart_temperature(df: pd.DataFrame, year=None, month_from=None, month_to=None) -> dict:
    d = _sub(df, year, month_from, month_to).dropna(subset=["date"])
    if d.empty or "t2m_max" not in d.columns:
        return _empty("温度趋势")
    fig, ax = plt.subplots(figsize=(11, 4.2))
    for col, key, label in [("t2m_max", "tmax", "最高温"), ("t2m_min", "tmin", "最低温"),
                            ("t2m", "tmean", "平均温")]:
        if col in d.columns:
            ax.plot(d["date"], d[col], color=COLORS[key], lw=0.7, alpha=0.55, label=label)
            ax.plot(d["date"], d[col].rolling(15, min_periods=1).mean(),
                    color=COLORS[key], lw=1.8)
    span = f"{d['date'].min():%Y-%m-%d} ~ {d['date'].max():%Y-%m-%d}"
    ax.set_title(f"最高温 / 最低温 / 平均温变化趋势（{span}）", fontsize=12)
    ax.set_ylabel("温度 (℃)")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=9, ncol=3)
    return {"img": _fig_to_uri(fig), "title": "温度变化趋势",
            "desc": "细线为逐日值，粗线为 15 日滑动平均，可直观看出冷暖过程与倒春寒。"}


def chart_precip(df: pd.DataFrame, year=None, month_from=None, month_to=None) -> dict:
    d = _sub(df, year, month_from, month_to).dropna(subset=["date"])
    if d.empty or "precip" not in d.columns:
        return _empty("降水变化")
    fig, ax = plt.subplots(figsize=(11, 4.2))
    multi_year = d["year"].nunique() > 1
    if multi_year:
        m = d.groupby(d["date"].dt.to_period("M"))["precip"].sum()
        ax.bar([p.to_timestamp() for p in m.index], m.values, width=20, color=COLORS["precip"])
        ax.set_title("逐月累计降水量", fontsize=12)
        desc = "按年比对月度降水总量，可快速识别干旱月份。"
    else:
        ax.bar(d["date"], d["precip"], width=1, color=COLORS["precip"])
        ax2 = ax.twinx()
        ax2.plot(d["date"], d["precip"].cumsum(), color="#ef6c00", lw=1.6, label="累计降水")
        ax2.set_ylabel("累计降水 (mm)", color="#ef6c00")
        ax.set_title(f"{int(d['year'].iloc[0])} 年日降水量与累计降水", fontsize=12)
        desc = "柱状为日降水，橙线为年内累计降水。"
    ax.set_ylabel("降水量 (mm)")
    ax.grid(alpha=0.25)
    return {"img": _fig_to_uri(fig), "title": "降水变化", "desc": desc}


def chart_wind(df: pd.DataFrame, year=None, month_from=None, month_to=None) -> dict:
    d = _sub(df, year, month_from, month_to).dropna(subset=["date"])
    if d.empty or "wind" not in d.columns:
        return _empty("风速变化")
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.plot(d["date"], d["wind"], color=COLORS["wind"], lw=0.6, alpha=0.45, label="逐日风速")
    ax.plot(d["date"], d["wind"].rolling(30, min_periods=1).mean(),
            color=COLORS["wind"], lw=1.8, label="30 日滑动平均")
    ax.set_ylabel("2m 风速 (m/s)")
    ax.set_title("2m 风速变化", fontsize=12)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    return {"img": _fig_to_uri(fig), "title": "风速变化",
            "desc": "30 日滑动平均反映风速的季节变化，风速偏大时需关注植株倒伏与蒸发。"}


def chart_radiation(df: pd.DataFrame, year=None, month_from=None, month_to=None) -> dict:
    d = _sub(df, year, month_from, month_to).dropna(subset=["date"])
    if d.empty or "radiation" not in d.columns:
        return _empty("辐射变化")
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.plot(d["date"], d["radiation"], color=COLORS["radiation"], lw=0.7, alpha=0.5)
    ax.plot(d["date"], d["radiation"].rolling(30, min_periods=1).mean(),
            color="#e65100", lw=1.8, label="30 日滑动平均")
    ax.set_ylabel("地表短波辐射 (MJ/m²/day)")
    ax.set_title("地表短波辐射变化", fontsize=12)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    return {"img": _fig_to_uri(fig), "title": "辐射变化",
            "desc": "辐射是光合作用的能量来源，灌浆期持续偏低会限制产量形成。"}


def chart_dew_rh(df: pd.DataFrame, year=None, month_from=None, month_to=None) -> dict:
    d = _sub(df, year, month_from, month_to).dropna(subset=["date"])
    if d.empty:
        return _empty("露点与湿度")
    fig, ax = plt.subplots(figsize=(11, 4.2))
    plotted = False
    if "dew" in d.columns:
        ax.plot(d["date"], d["dew"].rolling(7, min_periods=1).mean(),
                color=COLORS["dew"], lw=1.6, label="露点温度")
        ax.set_ylabel("露点温度 (℃)", color=COLORS["dew"])
        plotted = True
    if "rh" in d.columns:
        ax2 = ax.twinx()
        ax2.plot(d["date"], d["rh"].rolling(7, min_periods=1).mean(),
                 color=COLORS["rh"], lw=1.6, label="相对湿度")
        ax2.set_ylabel("相对湿度 (%)", color=COLORS["rh"])
        ax2.set_ylim(0, 100)
        plotted = True
    if not plotted:
        plt.close(fig)
        return _empty("露点与湿度")
    ax.set_title("露点温度与相对湿度变化（7 日滑动平均）", fontsize=12)
    ax.grid(alpha=0.25)
    return {"img": _fig_to_uri(fig), "title": "露点与湿度",
            "desc": "湿度与露点持续偏高时，抽穗扬花期病害风险上升。"}


def chart_month_compare(df: pd.DataFrame, year: int | None) -> dict:
    """不同年份 / 时间段的月度气象条件对比"""
    if "year" not in df.columns or df["year"].nunique() < 2:
        return _empty("年月对比")
    pivot_t = df.pivot_table(index="month", columns="year", values="t2m", aggfunc="mean")
    pivot_p = df.pivot_table(index="month", columns="year", values="precip", aggfunc="sum")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    years = list(pivot_t.columns)
    highlight = year if year in pivot_t.columns else years[-1]
    for y in years:
        alpha = 0.9 if y == highlight else 0.3
        lw = 2.4 if y == highlight else 1.0
        axes[0].plot(pivot_t.index, pivot_t[y], marker="o", ms=3, lw=lw, alpha=alpha,
                     label=str(int(y)))
        axes[1].plot(pivot_p.index, pivot_p[y], marker="o", ms=3, lw=lw, alpha=alpha,
                     label=str(int(y)))
    axes[0].set_title("逐月平均气温", fontsize=11)
    axes[0].set_ylabel("℃")
    axes[1].set_title("逐月累计降水", fontsize=11)
    axes[1].set_ylabel("mm")
    for ax in axes:
        ax.set_xticks(range(1, 13))
        ax.set_xlabel("月份")
        ax.grid(alpha=0.25)
    axes[1].legend(fontsize=8, ncol=2)
    fig.suptitle(f"不同年份月度气象条件对比（突出显示 {int(highlight)} 年）", fontsize=12)
    return {"img": _fig_to_uri(fig), "title": "年月对比",
            "desc": "同月跨年对比，用于判断某一年在关键生育期的冷暖与干湿位置。"}


def annual_summary(df: pd.DataFrame) -> pd.DataFrame:
    """分年气象统计表（综合评估基础）"""
    g = df.groupby("year")
    out = pd.DataFrame({
        "平均气温": g["t2m"].mean().round(2),
        "平均最高温": g["t2m_max"].mean().round(2),
        "平均最低温": g["t2m_min"].mean().round(2),
        "年降水量": g["precip"].sum().round(1),
        "平均风速": g["wind"].mean().round(2) if "wind" in df else np.nan,
        "平均辐射": g["radiation"].mean().round(2) if "radiation" in df else np.nan,
        "高温日数": g["t2m_max"].apply(lambda s: int((s >= 35).sum())),
        "霜冻日数": g["t2m_min"].apply(lambda s: int((s < 0).sum())),
    })
    return out.reset_index()


def chart_annual_overview(df: pd.DataFrame, year: int | None) -> dict:
    """气象条件综合评估：年度指标 + 与多年平均对比"""
    if "year" not in df.columns or df["year"].nunique() < 2:
        return _empty("综合评估")
    s = annual_summary(df)
    years = s["year"]
    highlight = year if year in set(years) else int(years.iloc[-1])

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    items = [("平均气温", "℃", axes[0][0]), ("年降水量", "mm", axes[0][1]),
             ("高温日数", "天", axes[1][0]), ("平均辐射", "MJ/m²/day", axes[1][1])]
    for name, unit, ax in items:
        vals = s[name]
        colors = ["#ef5350" if y == highlight else "#90a4ae" for y in years]
        ax.bar(years.astype(str), vals, color=colors)
        ax.axhline(vals.mean(), color="#37474f", ls="--", lw=1, label="多年平均")
        ax.set_title(f"{name}（{unit}）", fontsize=11)
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle(f"气象条件综合评估（红色为 {highlight} 年）", fontsize=13)
    fig.tight_layout()
    return {"img": _fig_to_uri(fig), "title": "综合评估",
            "desc": "年度指标与多年平均的对比，用于判断该年份整体气候条件。"}


def evaluate_period(df: pd.DataFrame, year: int | None) -> dict:
    """
    文字化综合评估：给定年份 vs 多年平均
    """
    s = annual_summary(df)
    if s.empty:
        return {"year": year, "text": "数据不足以做年度评估。", "rows": []}

    multi = s.drop(columns=["year"]).mean()
    target = year if year and year in set(s["year"]) else int(s["year"].iloc[-1])
    cur = s[s["year"] == target].iloc[0]

    def judge(name, better_high):
        c, m = float(cur[name]), float(multi[name])
        fmt = "{:.0f}" if name.endswith("日数") else "{:.1f}"
        if m == 0 or pd.isna(c) or pd.isna(m):
            return f"{name} {fmt.format(c)}"
        diff = (c - m) / abs(m) * 100
        if abs(diff) < 5:
            word = "接近常年"
        elif diff > 0:
            word = f"高于常年 {diff:.0f}%"
        else:
            word = f"低于常年 {abs(diff):.0f}%"
        return f"{name} {fmt.format(c)}（{word}）"

    text = (f"{target} 年：{judge('平均气温', True)}；{judge('年降水量', True)}；"
            f"{judge('高温日数', False)}；{judge('霜冻日数', False)}。")
    if cur["年降水量"] < multi["年降水量"] * 0.8:
        text += " 全年降水明显偏少，需关注灌溉水源与土壤墒情。"
    if cur["高温日数"] > multi["高温日数"] * 1.2 and cur["高温日数"] >= 5:
        text += " 高温日数偏多，灌浆期逼熟风险偏高。"
    if cur["霜冻日数"] > multi["霜冻日数"] * 1.2 and cur["霜冻日数"] >= 5:
        text += " 霜冻日数偏多，返青拔节期需防低温冷害。"

    return {"year": target, "text": text,
            "rows": s.round(2).to_dict("records")}


def build_all(df: pd.DataFrame, year: int | None = None,
              month_from: int | None = None, month_to: int | None = None) -> dict:
    """一次性生成全部分析结果，供 Web 分析页使用"""
    return {
        "charts": [
            chart_temperature(df, year, month_from, month_to),
            chart_precip(df, year, month_from, month_to),
            chart_wind(df, year, month_from, month_to),
            chart_radiation(df, year, month_from, month_to),
            chart_dew_rh(df, year, month_from, month_to),
            chart_month_compare(df, year),
            chart_annual_overview(df, year),
        ],
        "evaluation": evaluate_period(df, year),
        "years": sorted(int(y) for y in df["year"].dropna().unique()),
    }
