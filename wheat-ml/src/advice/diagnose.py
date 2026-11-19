"""上传数据后的即时诊断。

一份县区级逐日气象进来，要马上答三件事：里面有哪些灾害过程、哪几天、什么
类型、落在哪个生育期；对产量意味着什么；以及该不该做点什么（这部分交回
农事建议模块）。

判据跟建模那边保持一套，不能各说各话：
- 晚霜冻：4 月最低温 < 0℃，对应面板回归的 fz4_0，β = −1.96%/天
- 干热风：5 月最高温 ≥ 35℃ 且相对湿度 ≤ 40%，对应 hd_may，没到显著，只报天数
- 高温：3—6 月最高温 ≥ 35℃
- 低温：3 月、5 月最低温 < 0℃（4 月已经单列了）
- 暴雨：日降水 ≥ 25mm

减产幅度不搞线性外推，一律取剂量分档的实测均值（dose_frost.json）。
β 说的是"每多一天"的弹性，分档说的是"这一档平均掉多少"，两者不能混起来
算区间。
"""

import pandas as pd

from src.advice import stage_store

# 剂量分档：低、高、实测均值、样本数、说明（来源 output/reports/dose_frost.json）
DOSE_BANDS = [
    (0, 0, 0.25, 1232, "与常年基本持平"),
    (1, 1, -0.49, 310, "轻微影响，仍在常年波动内"),
    (2, 2, 0.10, 54, "仍在常年波动范围内"),
    (3, 4, -2.58, 27, "减产开始显性化，置信区间首次不跨零"),
    (5, 99, -7.75, 12, "重度减产"),
]

FROST_PER_DAY = -1.96   # 反向参考：每多 1 天的线性弹性（只用于说明，不参与区间计算）


def _dates(d: pd.DataFrame) -> list:
    return [pd.Timestamp(x).strftime("%m-%d") for x in d["date"]]


def _dose_loss(days: int):
    """霜冻天数 -> (减产均值, 样本数, 档位说明, 是否显著)。"""
    for lo, hi, mid, n, note in DOSE_BANDS:
        if lo <= days <= hi:
            return mid, n, note, (lo >= 3 or lo == 5)
    return None, 0, "", False


def diagnose(df: pd.DataFrame, county: str, year: int) -> dict:
    """对一份逐日气象做灾害识别与减产评估。"""
    base = {"empty": True, "county": county, "year": year, "events": [],
            "loss": None, "verdict": "", "stages": [], "days": 0, "month_span": ""}
    if df is None or df.empty or "date" not in df.columns:
        return base

    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d["month"] = d["date"].dt.month
    d["md"] = d["date"].dt.strftime("%m-%d")
    events = []

    # 4 月晚霜冻
    fz = d[(d["month"] == 4) & (d["t2m_min"] < 0)]
    if not fz.empty:
        days = len(fz)
        mid, n, note, sig = _dose_loss(days)
        events.append({
            "key": "frost", "label": "4 月晚霜冻", "days": days, "dates": _dates(fz),
            "level": "high" if sig else "watch",
            "loss_mid": mid, "loss_n": n, "loss_note": note, "loss_sig": sig,
            "basis": "面板回归：4 月最低温<0℃ 每多 1 天单产 −1.96%（p<0.001）；"
                     "本档减产取剂量分档实测均值",
            "action": "霜冻前灌水、喷施叶面肥提高抗寒性；冻后及时查苗补种",
        })

    # 5 月干热风
    if "rh" in d.columns:
        hd = d[(d["month"] == 5) & (d["t2m_max"] >= 35) & (d["rh"] <= 40)]
    else:
        hd = d[(d["month"] == 5) & (d["t2m_max"] >= 35)]
    if not hd.empty:
        events.append({
            "key": "hotdry", "label": "5 月干热风", "days": len(hd), "dates": _dates(hd),
            "level": "watch", "loss_mid": None,
            "basis": "面板回归中干热风系数未达显著（p=0.53），系统只报暴露强度，不折算减产",
            "action": "预报前 1—2 天灌水降温、喷施磷酸二氢钾延缓叶片衰老",
        })

    # 高温日
    hot = d[(d["month"].isin([3, 4, 5, 6])) & (d["t2m_max"] >= 35)]
    if not hot.empty:
        events.append({
            "key": "hot", "label": "高温日（≥35℃）", "days": len(hot), "dates": _dates(hot),
            "level": "info", "loss_mid": None,
            "basis": "用于判断灌浆期是否逼熟，未单独折算减产",
            "action": "关注灌浆期高温逼熟，必要时喷施叶面肥保绿叶",
        })

    # 非 4 月低温
    cold = d[(d["month"].isin([3, 5])) & (d["t2m_min"] < 0)]
    if not cold.empty:
        events.append({
            "key": "cold", "label": "低温日（3/5 月）", "days": len(cold), "dates": _dates(cold),
            "level": "info", "loss_mid": None,
            "basis": "4 月以外的低温记录，回归未单独验证，只作暴露提示",
            "action": "夜间骤降注意防冻，尤其是返青与灌浆阶段",
        })

    # 暴雨
    if "precip" in d.columns:
        rain = d[d["precip"] >= 25]
        if not rain.empty:
            events.append({
                "key": "rain", "label": "暴雨日（≥25mm）", "days": len(rain), "dates": _dates(rain),
                "level": "info", "loss_mid": None,
                "basis": "单日降水 ≥25mm，关注田间积水与倒伏",
                "action": "及时排水防渍害，雨后注意赤霉病",
            })

    loss = None
    frost = next((e for e in events if e["key"] == "frost"), None)
    if frost and frost["loss_mid"] is not None:
        loss = {"mid": frost["loss_mid"], "n": frost["loss_n"],
                "note": frost["loss_note"], "sig": frost["loss_sig"]}

    # 分生育期覆盖
    stages = []
    for st in stage_store.STAGES:
        seg = d[(d["md"] >= st["start"]) & (d["md"] <= st["end"])]
        hit = [e["label"] for e in events
               if any(st["start"] <= x <= st["end"] for x in e["dates"])]
        stages.append({"name": st["name"], "period": st["start"] + " ~ " + st["end"],
                       "days": int(len(seg)), "events": hit})

    ms = sorted(d["month"].unique().tolist())
    month_span = (f"{ms[0]}—{ms[-1]} 月" if len(ms) > 1 else f"{ms[0]} 月") if ms else ""

    if not events:
        verdict = (f"{year} 年这份数据覆盖 {len(d)} 天（{month_span}），未识别到霜冻、干热风等高影响灾害过程，"
                   f"按常规田间管理即可。")
    else:
        names = "、".join(f"{e['label']} {e['days']} 天" for e in events)
        if loss:
            verdict = (f"{year} 年识别到 {names}。4 月晚霜冻落在「{loss['note']}」档："
                       f"该档实测单产 {loss['mid']:+.2f}%（样本 n={loss['n']}），"
                       f"线性弹性口径为每多 1 天 −1.96%。")
        else:
            verdict = (f"{year} 年识别到 {names}，但均未达到回归验证的显著减产阈值，"
                       f"系统只提示暴露强度，不折算产量损失。")

    return {"empty": False, "county": county, "year": year, "days": int(len(d)),
            "month_span": month_span, "events": events, "loss": loss,
            "stages": stages, "verdict": verdict}
