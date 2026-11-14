"""
农事建议模块（对齐项目方案 2.4 + 表 1）
==========================================
输入：某县某年的逐日气象数据、该年预测单产、产量预警线
输出：按生育期（返青 / 拔节 / 抽穗扬花 / 灌浆 / 成熟）的气象统计、风险提示与农事建议

原则：
  - 建议方向对齐表 1，不直接给出具体施肥 / 灌溉剂量
  - 结合实际气象条件触发针对性提示，而不是只看产量预测值
"""

import yaml
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
with open(PROJECT_ROOT / "config" / "config.yaml", "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

STAGES = CONFIG["growth_stages"]


def stage_stats(d: pd.DataFrame) -> dict:
    """单个生育期的气象统计"""
    if d.empty:
        return {}
    stats = {
        "days": int(len(d)),
        "平均最高温": round(float(d["t2m_max"].mean()), 1),
        "平均最低温": round(float(d["t2m_min"].mean()), 1),
        "平均气温": round(float(d["t2m"].mean()), 1),
        "累计降水": round(float(d["precip"].sum()), 1),
        "降水日数": int((d["precip"] > 0.1).sum()),
        "高温日数": int((d["t2m_max"] >= 35).sum()),
        "霜冻日数": int((d["t2m_min"] < 0).sum()),
    }
    if "rh" in d.columns:
        stats["平均相对湿度"] = round(float(d["rh"].mean()), 1)
        stats["干热风日数"] = int(((d["t2m_max"] >= 35) & (d["rh"] <= 40)).sum())
    if "radiation" in d.columns:
        stats["平均辐射"] = round(float(d["radiation"].mean()), 1)
    if "wind" in d.columns:
        stats["平均风速"] = round(float(d["wind"].mean()), 1)
        stats["大风日数"] = int((d["wind"] >= 5).sum())
    return stats


def stage_climatology(df: pd.DataFrame) -> dict:
    """该站点多年各生育期降水与气温常年值"""
    out = {}
    d = df.assign(month_day=df["date"].dt.strftime("%m-%d"))
    for st in STAGES:
        sub = d[(d["month_day"] >= st["start"]) & (d["month_day"] <= st["end"])]
        per_year = sub.groupby("year").agg(precip=("precip", "sum"), t2m=("t2m", "mean"))
        if per_year.empty:
            out[st["key"]] = {}
        else:
            out[st["key"]] = {
                "precip": float(per_year["precip"].mean()),
                "t2m": float(per_year["t2m"].mean()),
            }
    return out


def stage_flags(key: str, stats: dict, climate: dict) -> list[str]:
    """按生育期判断气象风险（辅助信息）"""
    flags = []
    if not stats:
        return ["该生育期缺少气象记录"]
    cp = climate.get("precip")
    if cp and stats["累计降水"] < cp * 0.6:
        flags.append(f"降水偏少（{stats['累计降水']} mm，常年约 {cp:.0f} mm）")
    if cp and stats["累计降水"] > cp * 1.8:
        flags.append(f"降水偏多（{stats['累计降水']} mm，常年约 {cp:.0f} mm）")
    if stats["高温日数"] >= 3:
        flags.append(f"高温日数 {stats['高温日数']} 天")
    if stats.get("干热风日数", 0) >= 2:
        flags.append(f"干热风 {stats['干热风日数']} 天")
    frost_line = 8 if key == "greening" else 3      # 返青期霜冻本身常见，阈值放宽
    if stats["霜冻日数"] >= frost_line:
        flags.append(f"低温日数 {stats['霜冻日数']} 天")
    if stats.get("平均相对湿度", 0) >= 80 and stats["降水日数"] >= stats["days"] * 0.4:
        flags.append("湿度偏高、降水日数多")
    if stats.get("大风日数", 0) >= 3:
        flags.append(f"大风日数 {stats['大风日数']} 天")
    return flags


def stage_advice(key: str, stats: dict) -> list[str]:
    """生育期基础建议 + 气象条件触发建议"""
    base = {
        "greening": ["结合土壤墒情安排灌溉，做好苗情管理。"],
        "jointing": ["关注水肥管理，结合苗情合理追肥。"],
        "heading": ["关注田间湿度及病虫害防治。"],
        "filling": ["关注高温、干旱等天气条件，及时采取田间管理措施。"],
        "maturity": ["关注天气变化，合理安排收获。"],
    }.get(key, [])

    tips = []
    if not stats:
        return base + ["该生育期无气象记录，建议人工核查。"]
    if key == "greening":
        if stats["累计降水"] < 20:
            tips.append("返青期降水偏少，优先核查土壤墒情后再决定是否补灌。")
        if stats["霜冻日数"] >= 8:
            tips.append("出现持续低温过程，注意防范倒春寒，暂缓大水漫灌。")
    elif key == "jointing":
        if stats["累计降水"] < 30:
            tips.append("拔节期水分不足，结合苗情与墒情安排追肥与灌溉顺序。")
        if stats["霜冻日数"] >= 2:
            tips.append("拔节期遇低温，幼穗分化敏感，注意防冷害。")
        if stats["高温日数"] >= 3:
            tips.append("气温偏高、蒸发加快，注意保持田间水分。")
    elif key == "heading":
        if stats.get("平均相对湿度", 0) >= 80:
            tips.append("田间湿度偏大，重点排查赤霉病等穗部病害。")
        if stats["累计降水"] >= 40:
            tips.append("花期降水较多，注意雨后排涝与病害防治。")
        if stats["累计降水"] < 10:
            tips.append("花期偏干，关注花粉活力与授粉条件。")
    elif key == "filling":
        if stats.get("干热风日数", 0) >= 2 or stats["高温日数"] >= 3:
            tips.append("灌浆期高温 / 干热风明显，可采取喷灌等措施降低冠层温度，减轻逼熟。")
        if stats["累计降水"] < 20:
            tips.append("灌浆期降水不足，关注干旱对粒重的影响。")
        if stats.get("平均辐射", 99) < 14:
            tips.append("辐射偏弱，光合积累受限，注意保持叶片功能期。")
    elif key == "maturity":
        if stats["降水日数"] >= 5:
            tips.append("成熟期降水偏多，注意抢晴收获、及时晾晒，防止穗发芽。")
        if stats.get("大风日数", 0) >= 3:
            tips.append("成熟期大风偏多，注意防倒伏与落粒。")
        if not tips:
            tips.append("天气条件总体平稳，可按计划安排机械收获。")
    return base + tips


def generate_advice(weather: pd.DataFrame, county: str, year: int,
                    predicted_yield: float | None = None,
                    threshold: float | None = None) -> dict:
    """
    生成某县某年的生育期建议
    Returns: {"county","year","stages":[...],"yield_note","summary"}
    """
    climate = stage_climatology(weather)
    d = weather
    if "month_day" not in d.columns:
        d = d.assign(month_day=d["date"].dt.strftime("%m-%d"))

    stages = []
    main_mask = d["month_day"].apply(
        lambda m: any(s["start"] <= m <= s["end"] for s in STAGES))
    offseason = d[~main_mask]
    for st in STAGES:
        sub = d[(d["month_day"] >= st["start"]) & (d["month_day"] <= st["end"])]
        if year:
            sub = sub[sub["year"] == year]
        stats = stage_stats(sub)
        has_data = bool(stats)
        if has_data:
            flags = stage_flags(st["key"], stats, climate.get(st["key"], {}))
            advice = stage_advice(st["key"], stats)
            date_range = f"{sub['date'].min():%m-%d} ~ {sub['date'].max():%m-%d}"
        else:
            flags = ["待上传"]
            advice = ["该阶段暂无气象记录，可在上方上传本期观测数据。"]
            date_range = ""
        stages.append({
            "key": st["key"], "name": st["name"],
            "period": f"{st['start']} ~ {st['end']}",
            "focus": st["focus"],
            "stats": stats,
            "flags": flags,
            "advice": advice,
            "has_data": has_data,
            "date_range": date_range,
        })

    yield_note = ""
    if predicted_yield is not None and threshold is not None:
        diff = (predicted_yield - threshold) / threshold * 100
        if predicted_yield >= threshold:
            yield_note = (f"预测单产 {predicted_yield:.1f} kg/亩，高于预警线 "
                          f"{threshold:.1f} kg/亩（+{diff:.1f}%），产量预警状态：正常。")
        else:
            yield_note = (f"预测单产 {predicted_yield:.1f} kg/亩，低于预警线 "
                          f"{threshold:.1f} kg/亩（{diff:.1f}%），已触发产量预警。")
        yield_note += " 建议以气象条件为主安排农事，产量预测仅作参考。"

    return {
        "county": county,
        "year": int(year) if year else None,
        "stages": stages,
        "yield_note": yield_note,
        "offseason_days": int(len(offseason)),
        "summary": f"{county} {year} 年：共 {len(stages)} 个生育期，"
                   f"依据当期气象条件给出针对性建议。",
    }
