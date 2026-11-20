"""分析页交互图的数据。前端用 Chart.js。

这里只负责把序列整理成能直接 json 化的结构，渲染交给前端。
matplotlib 那套（weather_charts.py）留着，离线大屏和导图片还用得上。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .weather_charts import (FULL_YEAR_MIN_DAYS, _sub, annual_summary,
                              evaluate_period)

COLORS = {
    "tmax": "#A65B3A",
    "tmin": "#5C7E86",
    "tmean": "#1D4A2C",
    "precip": "#5C7E86",
    "cumulative": "#D2A24C",
    "wind": "#5B7F3F",
    "radiation": "#D2A24C",
    "dew": "#5C7E86",
    "rh": "#7C5C3F",
}


def _num(value, digits=1):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return round(float(value), digits)


def _nums(series, digits=1):
    return [_num(v, digits) for v in series]


def _date_labels(dates):
    return [pd.Timestamp(d).strftime("%Y-%m-%d") for d in dates]


def _monthly(df, column, agg="mean", digits=1):
    if column not in df.columns:
        return [], []
    grouped = df.groupby(df["date"].dt.to_period("M"))[column].agg(agg)
    labels = [period.strftime("%Y-%m") for period in grouped.index]
    return labels, _nums(grouped.values, digits)


def _empty(title, desc="当前筛选条件下缺少该字段或数据。"):
    return {"kind": "empty", "title": title, "desc": desc, "payload": {}}


def _line(title, desc, labels, datasets, y_label="", y2_label=None):
    return {
        "kind": "line",
        "title": title,
        "desc": desc,
        "payload": {
            "labels": labels,
            "datasets": datasets,
            "yLabel": y_label,
            "y2Label": y2_label,
        },
    }


def _daily_roll(df, column, label, color, window, axis="y", faint=False):
    if column not in df.columns:
        return None
    return {
        "label": label,
        "data": _nums(df[column].rolling(window, min_periods=1).mean()),
        "color": color,
        "axis": axis,
        "faint": faint,
    }


def temperature_chart(df, year=None, month_from=None, month_to=None):
    d = _sub(df, year, month_from, month_to).dropna(subset=["date"]).sort_values("date")
    if d.empty or "t2m" not in d.columns:
        return _empty("温度变化", "当前数据缺少平均气温字段。")

    multi = d["year"].nunique() > 1
    if multi:
        labels, _ = _monthly(d, "t2m")
        datasets = []
        for column, label, color in [
            ("t2m_max", "最高温", COLORS["tmax"]),
            ("t2m", "平均温", COLORS["tmean"]),
            ("t2m_min", "最低温", COLORS["tmin"]),
        ]:
            _, values = _monthly(d, column)
            if values:
                datasets.append({"label": label, "data": values, "color": color, "axis": "y"})
        desc = "逐月平均；全部年份并列，观察冷暖年际差异。"
    else:
        labels = _date_labels(d["date"])
        datasets = [
            ds for ds in [
                _daily_roll(d, "t2m_max", "最高温 · 15 日均", COLORS["tmax"], 15),
                _daily_roll(d, "t2m", "平均温 · 15 日均", COLORS["tmean"], 15),
                _daily_roll(d, "t2m_min", "最低温 · 15 日均", COLORS["tmin"], 15),
            ] if ds
        ]
        desc = "15 日滑动平均；看冷暖过程与倒春寒。"
    if not datasets:
        return _empty("温度变化")
    return _line("温度变化", desc, labels, datasets, "℃")


def precip_chart(df, year=None, month_from=None, month_to=None):
    d = _sub(df, year, month_from, month_to).dropna(subset=["date"]).sort_values("date")
    if d.empty or "precip" not in d.columns:
        return _empty("降水变化", "当前数据缺少日降水量字段。")

    if d["year"].nunique() > 1:
        labels, bars = _monthly(d, "precip", agg="sum")
        payload = {
            "labels": labels, "bars": bars, "barLabel": "月降水量",
            "cumulative": None, "lineLabel": None, "yLabel": "mm", "multi": True,
        }
        desc = "逐月累计降水；跨年并列，快速识别干旱月份。"
    else:
        labels = _date_labels(d["date"])
        payload = {
            "labels": labels,
            "bars": _nums(d["precip"]),
            "barLabel": "日降水量",
            "cumulative": _nums(d["precip"].cumsum()),
            "lineLabel": "年内累计",
            "yLabel": "mm",
            "multi": False,
        }
        desc = "柱状为日降水，金色线为年内累计。"
    return {"kind": "precip", "title": "降水变化", "desc": desc, "payload": payload}


def wind_chart(df, year=None, month_from=None, month_to=None):
    d = _sub(df, year, month_from, month_to).dropna(subset=["date"]).sort_values("date")
    if d.empty or "wind" not in d.columns:
        return _empty("风速变化", "当前数据缺少风速字段。")

    if d["year"].nunique() > 1:
        labels, values = _monthly(d, "wind")
        datasets = [{"label": "月平均风速", "data": values, "color": COLORS["wind"], "axis": "y"}]
        desc = "逐月平均风速。"
    else:
        labels = _date_labels(d["date"])
        datasets = [
            ds for ds in [
                _daily_roll(d, "wind", "逐日风速", COLORS["wind"], 1, faint=True),
                _daily_roll(d, "wind", "30 日滑动平均", COLORS["wind"], 30),
            ] if ds
        ]
        desc = "浅色为逐日值，深色为 30 日滑动平均。"
    return _line("风速变化", desc, labels, datasets, "m/s")


def radiation_chart(df, year=None, month_from=None, month_to=None):
    d = _sub(df, year, month_from, month_to).dropna(subset=["date"]).sort_values("date")
    if d.empty or "radiation" not in d.columns:
        return _empty("辐射变化", "当前数据缺少地表短波辐射字段。")

    if d["year"].nunique() > 1:
        labels, values = _monthly(d, "radiation")
        datasets = [{"label": "月平均辐射", "data": values, "color": COLORS["radiation"], "axis": "y"}]
        desc = "逐月平均辐射。"
    else:
        labels = _date_labels(d["date"])
        datasets = [
            ds for ds in [
                _daily_roll(d, "radiation", "逐日辐射", COLORS["radiation"], 1, faint=True),
                _daily_roll(d, "radiation", "30 日滑动平均", COLORS["radiation"], 30),
            ] if ds
        ]
        desc = "浅色为逐日值，深色为 30 日滑动平均。"
    return _line("辐射变化", desc, labels, datasets, "MJ/m²/day")


def dew_rh_chart(df, year=None, month_from=None, month_to=None):
    d = _sub(df, year, month_from, month_to).dropna(subset=["date"]).sort_values("date")
    has_dew = "dew" in d.columns and d["dew"].notna().any()
    has_rh = "rh" in d.columns and d["rh"].notna().any()
    if d.empty or not (has_dew or has_rh):
        return _empty("露点与湿度", "当前数据缺少露点温度与相对湿度字段。")

    if d["year"].nunique() > 1:
        labels = None
        datasets = []
        if has_dew:
            labels, values = _monthly(d, "dew")
            datasets.append({"label": "露点温度", "data": values, "color": COLORS["dew"], "axis": "y"})
        if has_rh:
            labels, values = _monthly(d, "rh")
            datasets.append({"label": "相对湿度", "data": values, "color": COLORS["rh"], "axis": "y2"})
        desc = "逐月平均露点与相对湿度。"
    else:
        labels = _date_labels(d["date"])
        datasets = [
            ds for ds in [
                _daily_roll(d, "dew", "露点温度 · 7 日均", COLORS["dew"], 7, axis="y") if has_dew else None,
                _daily_roll(d, "rh", "相对湿度 · 7 日均", COLORS["rh"], 7, axis="y2") if has_rh else None,
            ] if ds
        ]
        desc = "7 日滑动平均；湿度与露点持续偏高时，抽穗扬花期病害风险上升。"
    return _line("露点与湿度", desc, labels, datasets, "℃", "%")


def month_compare_chart(df, year=None):
    if "year" not in df.columns or df["year"].nunique() < 2:
        return _empty("年月对比", "需要至少两个年份的数据才能做跨年对比。")

    years = [int(y) for y in sorted(df["year"].dropna().unique())]
    highlight = int(year) if year in years else years[-1]
    labels = [f"{m}月" for m in range(1, 13)]

    def build(column, agg):
        if column not in df.columns:
            return []
        pivot = df.pivot_table(index="month", columns="year", values=column,
                               aggfunc=agg).reindex(range(1, 13))
        out = []
        for y in years:
            if y not in pivot.columns:
                continue
            out.append({
                "label": str(y),
                "data": _nums(pivot[y].values),
                "highlight": y == highlight,
            })
        return out

    temp = build("t2m", "mean")
    precip = build("precip", "sum")
    if not temp and not precip:
        return _empty("年月对比")
    # 部分年只有几个月的数据，曲线在那里断开；不写清楚会被读成"其余月份为零"
    thin_years = []
    for y in years:
        rows_n = int((df["year"] == y).sum())
        if rows_n < FULL_YEAR_MIN_DAYS:
            thin_years.append((y, rows_n))

    desc = f"同月跨年对比，突出显示 {highlight} 年；其余年份为浅绿。"
    if thin_years:
        detail = "、".join(f"{y} 年（仅 {n} 天）" for y, n in thin_years)
        desc += f" {detail}属当期观测，曲线只画到已观测的月份，断口处不代表其余月份为零。"

    return {
        "kind": "monthCompare",
        "title": "年月对比",
        "desc": desc,
        "payload": {
            "labels": labels,
            "temp": {"unit": "℃", "datasets": temp},
            "precip": {"unit": "mm", "datasets": precip},
            "highlight": highlight,
        },
    }


def annual_overview_chart(df, year=None):
    if "year" not in df.columns or df["year"].nunique() < 2:
        return _empty("综合评估", "需要至少两个年份的数据才能做年度对比。")

    summary = annual_summary(df)
    # 年度指标只对完整年成立：十几天算出来的"年降水量""全年高温日"不是小一点，是量纲不对
    partial_years = []
    if "观测天数" in summary.columns:
        partial_rows = summary[summary["观测天数"] < FULL_YEAR_MIN_DAYS]
        partial_years = [(int(r["year"]), int(r["观测天数"])) for _, r in partial_rows.iterrows()]
        summary = summary[summary["观测天数"] >= FULL_YEAR_MIN_DAYS]
    if len(summary) < 2:
        return _empty("综合评估", "完整年份不足两年，无法做年度对比；当期观测请看上方「当期观测对照」。")

    years = [int(y) for y in summary["year"]]
    highlight = int(year) if int(year) in years else years[-1]
    panels = []
    for name, unit, digits in [
        ("平均气温", "℃", 1),
        ("年降水量", "mm", 1),
        ("高温日数", "天", 0),
        ("平均辐射", "MJ/m²/day", 1),
    ]:
        if name not in summary or not summary[name].notna().any():
            continue
        panels.append({
            "title": name,
            "unit": unit,
            "labels": [str(y) for y in years],
            "values": _nums(summary[name].values, digits),
            "mean": _num(summary[name].mean(), digits),
            "highlight": highlight,
        })
    if not panels:
        return _empty("综合评估")
    desc = f"年度指标与多年平均对比，绿色为 {highlight} 年；虚线为完整年的多年平均。"
    if partial_years:
        detail = "、".join(f"{y} 年（仅 {n} 天）" for y, n in partial_years)
        desc += f" {detail}不足一整年，年度指标不成立，未进本图。"

    return {
        "kind": "panelBars",
        "title": "综合评估",
        "desc": desc,
        "payload": {"panels": panels, "highlight": highlight},
    }


# 只有完整年才成立的指标：观测天数不够时这些数是窗口累计/窗口计数，不是全年值
ANNUAL_ONLY_KEYS = ("年降水量", "高温日数", "霜冻日数")


def _evaluation(df, year):
    result = evaluate_period(df, year)
    rows = []
    for row in result.get("rows", []):
        days = row.get("观测天数")
        partial_row = bool(days is not None and pd.notna(days) and int(days) < FULL_YEAR_MIN_DAYS)
        clean = {}
        for key, value in row.items():
            if key == "year":
                clean[key] = int(value)
            elif partial_row and key in ANNUAL_ONLY_KEYS:
                clean[key] = None          # 十几天算不出年降水量，留空比填假数好
            elif isinstance(value, (int, float, np.integer, np.floating)):
                clean[key] = _num(value, 2)
            else:
                clean[key] = value
        rows.append(clean)
    return {"year": result.get("year"), "text": result.get("text", ""), "rows": rows,
            "partial": bool(result.get("partial")), "days": result.get("days"),
            "span": result.get("span", "")}


# ---- 生育期灾害窗口 ----
import yaml as _yaml
from pathlib import Path as _Path

_CFG = _Path(__file__).resolve().parents[2] / "config" / "config.yaml"
_STAGES_CACHE = None


def _stages():
    global _STAGES_CACHE
    if _STAGES_CACHE is None:
        with open(_CFG, encoding="utf-8") as f:
            _STAGES_CACHE = _yaml.safe_load(f).get("growth_stages", [])
    return _STAGES_CACHE


def stage_windows(df, year=None):
    """把气象落到五段生育期，并标注灾害窗口。

    判据与面板回归口径一致（不另造阈值）：
      4 月低温 t2m_min<=0  → 每多 1 天 -1.96%/天（p<0.001）
      3 月低温 t2m_min<=0  → 每多 1 天 -0.57%/天（p<0.001）
      5 月干热风 t2m_max>=35 且 rh<=40 → 六种口径均不显著，只报暴露强度
    """
    # 生育期窗口必须落到单一年份；"全部年份"下不显示，避免把 15 年混在一起统计
    if not year:
        return None
    if df is None or len(df) == 0 or "date" not in df.columns:
        return None
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.dropna(subset=["date"])
    if year:
        d = d[d["date"].dt.year == int(year)]
    if d.empty:
        return None

    has_tmin = "t2m_min" in d.columns
    has_tmax = "t2m_max" in d.columns
    has_rh = "rh" in d.columns
    d["md"] = d["date"].dt.strftime("%m-%d")
    d = d.sort_values("date")

    # 档位必须按「全年 4 月累计天数」判定：同一场低温会被切进拔节期与抽穗扬花期，
    # 若按单期天数各判一次档，2 天 + 6 天会被读成两次损失。
    m_all = d["date"].dt.month
    apr_cold_total = int(((m_all == 4) & (d["t2m_min"] <= 0)).sum()) if has_tmin else 0
    mar_cold_total = int(((m_all == 3) & (d["t2m_min"] <= 0)).sum()) if has_tmin else 0
    if apr_cold_total >= 5:
        band_loss, band_txt = -7.75, "≥5 天档（n=12）"
    elif apr_cold_total >= 3:
        band_loss, band_txt = -2.58, "3—4 天档（n=27）"
    elif apr_cold_total >= 1:
        band_loss, band_txt = 0.0, "0—2 天档"
    else:
        band_loss, band_txt = 0.0, ""

    rows = []
    for st in _stages():
        seg = d[(d["md"] >= st["start"]) & (d["md"] <= st["end"])]
        if seg.empty:
            rows.append({"name": st["name"], "range": f'{st["start"]}~{st["end"]}',
                         "days": 0, "tavg": "—", "precip": "—", "cold": "—",
                         "hot": "—", "cold_mar": "—", "cold_apr": "—", "hot_may": "—",
                         "note": "该期无记录", "tone": "none"})
            continue

        tavg = round(float(seg["t2m"].mean()), 1) if "t2m" in seg.columns else "—"
        precip = round(float(seg["precip"].sum()), 1) if "precip" in seg.columns else "—"
        cold = int((seg["t2m_min"] <= 0).sum()) if has_tmin else 0
        hot_plain = int((seg["t2m_max"] >= 35).sum()) if has_tmax else 0

        months = seg["date"].dt.month
        cold_apr = int(((months == 4) & (seg["t2m_min"] <= 0)).sum()) if has_tmin else 0
        cold_mar = int(((months == 3) & (seg["t2m_min"] <= 0)).sum()) if has_tmin else 0
        hot_may = 0
        if has_tmax and has_rh:
            hot_may = int(((months == 5) & (seg["t2m_max"] >= 35) & (seg["rh"] <= 40)).sum())

        note, tone = "无灾害窗口", "none"
        # 行内只报该期事实天数；档位与折算统一在下方口径条按「全年累计」给出，
        # 避免同一场低温跨两个生育期被重复折算。
        bits = []
        if cold_apr > 0:
            bits.append(f"4 月低温 {cold_apr} 天")
        if cold_mar > 0:
            bits.append(f"3 月低温 {cold_mar} 天（分档 p>0.05，不折算）")
        if bits:
            note = "；".join(bits)
            tone = "alert" if (cold_apr > 0 and band_loss < 0) else "warn"
        elif cold > 0:
            note = f"≤0℃ {cold} 天（非 3—4 月，不折算）"
            tone = "warn"
        if hot_may > 0:
            extra = f"干热风 {hot_may} 天（仅提示）"
            note = extra if tone == "none" else note + "；" + extra
            if tone == "none":
                tone = "warn"

        rows.append({
            "name": st["name"], "range": f'{st["start"]}~{st["end"]}',
            "days": int(len(seg)), "tavg": tavg, "precip": precip,
            "cold": cold, "hot": hot_plain,
            "cold_mar": cold_mar, "cold_apr": cold_apr, "hot_may": hot_may,
            "note": note, "tone": tone,
        })

    # 全省/全县四月口径（不受生育期切段影响，用于与回归口径核对）
    apr = d[d["date"].dt.month == 4]
    may = d[d["date"].dt.month == 5]
    apr_cold = apr_cold_total
    may_hot = 0
    if len(may) and has_tmax and has_rh:
        may_hot = int(((may["t2m_max"] >= 35) & (may["rh"] <= 40)).sum())

    bl = abs(band_loss)
    parts = []
    if apr_cold > 0:
        tail = f"，折算约 −{bl:.1f}%" if bl else "，实测无显著损失"
        parts.append(f"4 月低温 {apr_cold} 天（全年累计），落在 {band_txt}{tail}")
    else:
        parts.append("4 月未出现 ≤0℃ 低温日，晚霜冻判据未触发")
    if apr_cold > 0:
        parts.append("分档为剔除县年固定效应后的组均值，线性弹性 −1.96%/天 是控制其他因子后的边际斜率，两者口径不同，此处取实测分档")
    if mar_cold_total > 0:
        parts.append(f"3 月低温 {mar_cold_total} 天，分档 p>0.05 且非单调（0—2 天 −0.7%、3—5 天 +0.4%），只报天数不折算")
    if may_hot > 0:
        parts.append(f"5 月干热风 {may_hot} 天，六种口径均不显著，只报暴露强度、不折算减产")
    else:
        parts.append("5 月未出现干热风日")
    parts.append("折算值仅供研判；实际减产还受灌溉、品种、田间管理，以及本系统未覆盖的灾害（如成熟期连阴雨）影响")

    return {
        "year": int(year) if year else None,
        "rows": rows,
        "summary": "；".join(parts),
        "apr_cold": apr_cold,
        "may_hot": may_hot,
    }



def obs_compare(df, year=None):
    """当期观测窗口 vs 该县常年同期。

    十几天算不出「年降水量」，但能回答「这十几天比常年同期暖多少、雨少多少」。
    短期观测撑得住的结论就在这一层，年尺度结论交给上面的完整年。
    基准取该县其他年份落在同一月日区间（同一窗口）的逐年值再平均，
    不是全年均值——否则四月的数据会被拿去跟全年降水比，必然"偏少九成"。
    """
    if df is None or len(df) == 0 or "date" not in df.columns:
        return None
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.dropna(subset=["date"])
    if d.empty:
        return None

    yr = int(year) if year is not None else int(d["date"].dt.year.max())
    cur = d[d["date"].dt.year == yr]
    if cur.empty:
        return None
    # 整年数据不走这里，年尺度看「综合评估」
    if len(cur) >= FULL_YEAR_MIN_DAYS:
        return None

    start, end = cur["date"].min(), cur["date"].max()
    smd, emd = f"{start:%m-%d}", f"{end:%m-%d}"
    days = int(len(cur))

    base = d[d["date"].dt.year != yr].copy()
    if base.empty:
        return None
    base["md"] = base["date"].dt.strftime("%m-%d")
    base = base[(base["md"] >= smd) & (base["md"] <= emd)]
    if base.empty:
        return None

    def one(frame):
        has = lambda c: c in frame.columns
        return {
            "tavg": float(frame["t2m"].mean()) if has("t2m") and len(frame) else None,
            "tmin": float(frame["t2m_min"].min()) if has("t2m_min") and len(frame) else None,
            "tmax": float(frame["t2m_max"].max()) if has("t2m_max") and len(frame) else None,
            "precip": float(frame["precip"].sum()) if has("precip") and len(frame) else None,
            "cold": int((frame["t2m_min"] <= 0).sum()) if has("t2m_min") else 0,
            "hot": int(((frame["t2m_max"] >= 35) & (frame["rh"] <= 40)).sum())
                   if has("t2m_max") and has("rh") else 0,
        }

    c = one(cur)
    per_year = [one(g) for _, g in base.groupby(base["date"].dt.year)]

    def avg(key):
        vals = [v[key] for v in per_year if v[key] is not None]
        return sum(vals) / len(vals) if vals else None

    rows = []

    def add_row(label, cur_val, base_val, unit, digits=1):
        if cur_val is None or base_val is None:
            return
        rows.append({
            "name": label,
            "cur": f"{cur_val:.{digits}f}",
            "base": f"{base_val:.{digits}f}",
            "unit": unit,
            "diff": f"{cur_val - base_val:+.{digits}f}",
            "tone": "down" if cur_val < base_val else ("up" if cur_val > base_val else "same"),
        })

    add_row("窗口平均气温", c["tavg"], avg("tavg"), "℃")
    add_row("窗口最低温", c["tmin"], avg("tmin"), "℃")
    add_row("窗口累计降水", c["precip"], avg("precip"), "mm")
    add_row("窗口内 ≤0℃ 天数", float(c["cold"]), avg("cold"), "天", 0)
    add_row("窗口内干热风天数", float(c["hot"]), avg("hot"), "天", 0)

    if not rows:
        return None

    # 一句人话结论：只说窗口内成立的事，不外推到全年
    bits = []
    t_cur, t_base = c["tavg"], avg("tavg")
    if t_cur is not None and t_base is not None:
        d_t = t_cur - t_base
        if abs(d_t) >= 0.5:
            bits.append(f"均温{'偏暖' if d_t > 0 else '偏冷'} {abs(d_t):.1f}℃")
    p_cur, p_base = c["precip"], avg("precip")
    if p_cur is not None and p_base is not None and p_base > 0:
        ratio = (p_cur - p_base) / p_base * 100
        if abs(ratio) >= 10:
            bits.append(f"降水{'偏多' if ratio > 0 else '偏少'}{abs(ratio):.0f}%")
    if c["cold"] > 0:
        bits.append(f"出现 ≤0℃ 低温 {c['cold']} 天")
    if c["hot"] > 0:
        bits.append(f"干热风 {c['hot']} 天")

    verdict = "；".join(bits) if bits else "各项与常年同期接近，窗口内无异常信号"

    return {
        "kind": "obsCompare",
        "title": "当期观测对照",
        "desc": f"{yr} 年只有 {days} 天观测（{start:%m-%d} ~ {end:%m-%d}），算不出年度指标，改与常年同期比。",
        "year": yr,
        "days": days,
        "window": f"{start:%m-%d} ~ {end:%m-%d}",
        "rows": rows,
        "verdict": verdict,
        "note": f"基准＝该县其他年份落在 {smd} ~ {emd} 同一区间的逐年值再平均（逐年相同权重），不是全年均值。",
    }


def build_web(df, year=None, month_from=None, month_to=None):
    """分析页一次性构建全部图表数据"""
    return {
        "charts": [
            temperature_chart(df, year, month_from, month_to),
            precip_chart(df, year, month_from, month_to),
            wind_chart(df, year, month_from, month_to),
            radiation_chart(df, year, month_from, month_to),
            dew_rh_chart(df, year, month_from, month_to),
            month_compare_chart(df, year),
            annual_overview_chart(df, year),
        ],
        "evaluation": _evaluation(df, year),
        "obs_compare": obs_compare(df, year),
        "stage_windows": stage_windows(df, year),
        "years": sorted(int(y) for y in df["year"].dropna().unique()) if "year" in df else [],
    }
