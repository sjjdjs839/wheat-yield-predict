"""
Web 交互图表数据（Chart.js）
============================
把分析页需要的序列整理成可直接 JSON 序列化的结构，前端用 Chart.js 渲染。
原 weather_charts.py 的 matplotlib 输出保留，用于离线大屏与图片导出。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .weather_charts import _sub, annual_summary, evaluate_period

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
    return {
        "kind": "monthCompare",
        "title": "年月对比",
        "desc": f"同月跨年对比，突出显示 {highlight} 年；其余年份为浅绿。",
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
    years = [int(y) for y in summary["year"]]
    highlight = int(year) if year in years else years[-1]
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
    return {
        "kind": "panelBars",
        "title": "综合评估",
        "desc": f"年度指标与多年平均对比，绿色为 {highlight} 年。",
        "payload": {"panels": panels, "highlight": highlight},
    }


def _evaluation(df, year):
    result = evaluate_period(df, year)
    rows = []
    for row in result.get("rows", []):
        clean = {}
        for key, value in row.items():
            if key == "year":
                clean[key] = int(value)
            elif isinstance(value, (int, float, np.integer, np.floating)):
                clean[key] = _num(value, 2)
            else:
                clean[key] = value
        rows.append(clean)
    return {"year": result.get("year"), "text": result.get("text", ""), "rows": rows}


# ---------------- 生育期灾害窗口（分析页主题化） ----------------
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
        loss = cold_apr * 1.96 + cold_mar * 0.57
        bits = []
        if cold_apr > 0:
            bits.append(f"4 月低温 {cold_apr} 天")
        if cold_mar > 0:
            bits.append(f"3 月低温 {cold_mar} 天")
        if bits:
            note = "、".join(bits) + f" ≈ −{loss:.1f}%"
            tone = "alert" if cold_apr > 0 else "warn"
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
    apr_cold = int((apr["t2m_min"] <= 0).sum()) if (len(apr) and has_tmin) else 0
    may_hot = 0
    if len(may) and has_tmax and has_rh:
        may_hot = int(((may["t2m_max"] >= 35) & (may["rh"] <= 40)).sum())

    parts = []
    if apr_cold > 0:
        parts.append(f"4 月低温 {apr_cold} 天，按面板回归 −1.96%/天折算约 −{apr_cold * 1.96:.1f}%")
    else:
        parts.append("4 月未出现 ≤0℃ 低温日，晚霜冻判据未触发")
    if may_hot > 0:
        parts.append(f"5 月干热风 {may_hot} 天，六种口径均不显著，只报暴露强度、不折算减产")
    else:
        parts.append("5 月未出现干热风日")
    parts.append("折算值仅供研判，实际减产还受灌溉、品种与田间管理影响")

    return {
        "year": int(year) if year else None,
        "rows": rows,
        "summary": "；".join(parts),
        "apr_cold": apr_cold,
        "may_hot": may_hot,
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
        "stage_windows": stage_windows(df, year),
        "years": sorted(int(y) for y in df["year"].dropna().unique()) if "year" in df else [],
    }
