"""近期观测的灾害判定。

数据源是当期观测库：上传的近期逐日气象按县累积在那里。
判定只看观测窗口内的日值，不碰模型，也不做年度回看。
致灾条件按这批数据落在哪一段生育期来取，不按日历月份硬编码：
  返青 / 拔节 / 抽穗扬花  晚霜冻：日最低温 ≤ 0℃ 的天数
  灌浆                    干热风：≥35℃ 且相对湿度 ≤40%；高温：≥34℃
  成熟                    高温逼熟 + 暴雨：日降水 ≥25 mm
  非生育期                只报极端：≥34℃、≥25 mm

晚霜冻、干热风的计数口径与面板回归一致（fz3/fz4 按天、干热风取自定义口径），
所以这里报的是「发生了几天」，不是「减产多少」。减产要另走回归。
"""

from datetime import date, datetime

import pandas as pd

from src.advice import stage_store

# 判定阈值
FROST_TMIN = 0.0          # 最低温 ≤ 0℃ 记晚霜冻
HOT_DRY_TMAX = 35.0       # 干热风：≥35℃
HOT_DRY_RH = 40.0         # 干热风：相对湿度 ≤40%
HOT_TMAX = 34.0           # 高温
RAIN_HEAVY = 25.0         # 暴雨

LEVEL_NONE, LEVEL_LIGHT, LEVEL_HEAVY = 0, 1, 2
LEVEL_TEXT = {0: "无", 1: "轻", 2: "重"}

STAGE_NAME = {s["key"]: s["name"] for s in stage_store.STAGES}


def _level_by_days(days: int) -> int:
    """0 天无、1—2 天轻、≥3 天重。和年度风险清单的分档一致。"""
    if days <= 0:
        return LEVEL_NONE
    return LEVEL_HEAVY if days >= 3 else LEVEL_LIGHT


def _stage_mix(d: pd.DataFrame) -> str:
    """窗口内按天数最多的生育期定档，决定判哪几种灾。"""
    md = d["date"].dt.strftime("%m-%d")
    keys = [stage_store.classify_stage(m) for m in md]
    if not keys:
        return "offseason"
    return max(set(keys), key=keys.count)


def judge(df: pd.DataFrame) -> dict:
    """对一份观测帧做判定，返回命中项与总体等级。"""
    out = {"hits": [], "level": LEVEL_NONE, "window": "", "days": 0,
           "tmin": None, "tmax": None, "rh": None, "precip": None,
           "stage_key": "offseason"}
    if df is None or df.empty:
        return out

    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d["month"] = d["date"].dt.month
    d = d.sort_values("date")

    out["window"] = "%s — %s" % (d["date"].min().strftime("%m-%d"),
                                 d["date"].max().strftime("%m-%d"))
    out["days"] = len(d)
    if "t2m_min" in d:
        out["tmin"] = round(float(d["t2m_min"].min()), 1)
    if "t2m_max" in d:
        out["tmax"] = round(float(d["t2m_max"].max()), 1)
    if "rh" in d:
        out["rh"] = round(float(d["rh"].min()), 0)
    if "precip" in d:
        out["precip"] = round(float(d["precip"].sum()), 1)

    hits = []
    stage = _stage_mix(d)
    out["stage_key"] = stage

    # 晚霜冻：返青到抽穗扬花这一段最怕 ≤0℃
    if "t2m_min" in d and stage in ("greening", "jointing", "heading"):
        frost = d[d["t2m_min"] <= FROST_TMIN]
        if not frost.empty:
            days = len(frost)
            hits.append({
                "key": "frost", "name": "晚霜冻", "days": days,
                "level": _level_by_days(days),
                "detail": "%s 最低 %.1f℃" % (
                    frost["date"].min().strftime("%m-%d"),
                    float(frost["t2m_min"].min())),
                "rule": "生育期内日最低温 ≤ 0℃",
            })

    # 干热风：灌浆期是主要受害段
    if "t2m_max" in d and "rh" in d and stage in ("filling", "maturity"):
        hd = d[(d["t2m_max"] >= HOT_DRY_TMAX) & (d["rh"] <= HOT_DRY_RH)]
        if not hd.empty:
            days = len(hd)
            hits.append({
                "key": "hot_dry", "name": "干热风", "days": days,
                "level": _level_by_days(days),
                "detail": "最高 %.1f℃ · 湿度低至 %.0f%%" % (
                    float(hd["t2m_max"].max()), float(hd["rh"].min())),
                "rule": "日最高温 ≥35℃ 且相对湿度 ≤40%",
            })

    # 高温：任何生育期都记，但非生育期不单独成灾
    if "t2m_max" in d:
        hot = d[d["t2m_max"] >= HOT_TMAX]
        if not hot.empty and stage != "offseason":
            days = len(hot)
            hits.append({
                "key": "hot", "name": "高温", "days": days,
                "level": _level_by_days(days),
                "detail": "峰值 %.1f℃" % float(hot["t2m_max"].max()),
                "rule": "日最高温 ≥34℃",
            })

    # 暴雨：成熟期抢收最被动，非生育期也算极端
    if "precip" in d:
        rain = d[d["precip"] >= RAIN_HEAVY]
        if not rain.empty:
            days = len(rain)
            hits.append({
                "key": "rain", "name": "暴雨", "days": days,
                "level": LEVEL_HEAVY if days >= 2 else LEVEL_LIGHT,
                "detail": "最大日雨量 %.1f mm" % float(rain["precip"].max()),
                "rule": "日降水量 ≥25 mm",
            })

    out["hits"] = sorted(hits, key=lambda x: (-x["level"], -x["days"]))
    out["level"] = max([h["level"] for h in hits], default=LEVEL_NONE)
    return out


def _stage_of(df: pd.DataFrame) -> str:
    """窗口主要落在哪个生育期，按天数最多的算。"""
    if df is None or df.empty or "date" not in df.columns:
        return ""
    md = pd.to_datetime(df["date"]).dt.strftime("%m-%d")
    keys = [stage_store.classify_stage(m) for m in md]
    if not keys:
        return ""
    top = max(set(keys), key=keys.count)
    return "" if top == "offseason" else top


def board(counties: list[str], year: int | None = None,
          window_days: int = 15, index=None) -> dict:
    """给一批县出近期预警看板。

    只有被标记为「当期观测」的数据集才进这里：年度数据集是回看用的，
    拿它算预警等于把整年的极端值都报成今天的险情。

    window_days 只截取观测尾部的一段；若该县观测不足这个窗口，就报现有的全部。
    """
    from src.data import upload_index as idx

    today = date.today()
    year = year or today.year
    rows, empty = [], []

    for county in counties:
        recs = [r for r in idx.by_county(county) if r.get("kind") == idx.KIND_RECENT]
        if not recs:
            continue
        frames = []
        for r in recs:
            for y in (r.get("years") or [year]):
                try:
                    one = stage_store.load(county, int(y))
                except Exception:
                    one = pd.DataFrame()
                if one is None or one.empty:
                    continue
                # 只取这份上传自己的区间：当期观测库里可能还躺着别的批次
                if r.get("start") and r.get("end"):
                    one = one[(one["date"] >= pd.Timestamp(r["start"])) &
                              (one["date"] <= pd.Timestamp(r["end"]))]
                if not one.empty:
                    frames.append(one)
        if not frames:
            empty.append(county)
            continue
        raw = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["date"], keep="last")

        d = raw.copy()
        d["date"] = pd.to_datetime(d["date"])
        d = d.sort_values("date")
        cutoff = d["date"].max() - pd.Timedelta(days=window_days - 1)
        win = d[d["date"] >= cutoff]

        verdict = judge(win)
        stage_key = _stage_of(win)
        gap = (today - d["date"].max().date()).days

        rows.append({
            "county": county,
            "stage_key": stage_key,
            "stage_name": STAGE_NAME.get(stage_key, "非生育期"),
            "window": verdict["window"],
            "days": verdict["days"],
            "level": verdict["level"],
            "level_text": LEVEL_TEXT[verdict["level"]],
            "hits": verdict["hits"],
            "hit_names": "、".join(h["name"] for h in verdict["hits"]) or "无",
            "tmin": verdict["tmin"],
            "tmax": verdict["tmax"],
            "rh": verdict["rh"],
            "precip": verdict["precip"],
            "last_date": d["date"].max().strftime("%Y-%m-%d"),
            "gap": gap,
            "stale": gap > 7,
        })

    rows.sort(key=lambda x: (-x["level"], x["gap"], x["county"]))
    return {
        "rows": rows,
        "empty": empty,
        "window_days": window_days,
        "year": year,
        "heavy": sum(1 for r in rows if r["level"] == LEVEL_HEAVY),
        "light": sum(1 for r in rows if r["level"] == LEVEL_LIGHT),
        "none": sum(1 for r in rows if r["level"] == LEVEL_NONE),
        "total": len(rows),
    }
