# -*- coding: utf-8 -*-
"""
从 NASA POWER 拉取当期（或指定区间）逐日气象，输出可直接在「上传数据」页上传的 CSV。

用法：
    python -m scripts.fetch_recent                       # 全部县，最近 14 天（截至前天）
    python -m scripts.fetch_recent --days 20             # 最近 20 天
    python -m scripts.fetch_recent --county 内黄县 义马市  # 只拉指定县
    python -m scripts.fetch_recent --start 20240501 --end 20240520   # 指定区间（演示灌浆期用）
    python -m scripts.fetch_recent --list                # 只列出将拉取的县

输出：
    data/raw/weather_recent/<城市>/<县>.csv     与 data/raw/weather 同结构（中文表头）
    data/raw/weather_recent/_fetch_meta.json    本次拉取记录（时间、区间、成败）

说明：
    NASA POWER 是公开的县级格点再分析产品，日尺度，近实时延迟约 1—2 天。
    参数与项目原始数据完全一致，因此拉取结果可直接替换原始文件或走上传通道。
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import ssl
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "config" / "config.yaml"
OUT_DIR = ROOT / "data" / "raw" / "weather_recent"
CTR_FILE = ROOT / "data" / "raw" / "henan_county_centers.csv"

PARAMS = "T2M_MAX,T2M_MIN,T2M,PRECTOTCORR,WS2M,ALLSKY_SFC_SW_DWN,T2MDEW,RH2M"
API = "https://power.larc.nasa.gov/api/temporal/daily/point"
HEADER_CN = ["年份", "日", "最高温", "最低温", "平均温", "日降水量",
             "2m风速", "地表短波辐射", "露点温度", "相对湿度"]

# 新设区 / 功能区 -> 用于取坐标的参照区（仅影响取哪个格点，不影响县名）
ALIAS = {
    "瀍河区": "瀍河回族区",
    "吉利区": "孟津区",
    "伊滨区": "洛龙区",
    "示范区": "龙亭区",
    "高新区": "涧西区",
    "龙管会": "洛龙区",
}


def _ssl_ctx():
    c = ssl.create_default_context()
    c.check_hostname = False
    c.verify_mode = ssl.CERT_NONE
    return c


def norm(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"(气象|观测|国家站|站)$", "", s)
    s = re.sub(r"(县|市|区)$", "", s)
    return s


def load_counties():
    """返回 [(城市, 文件名, 标准地名)]"""
    try:
        import yaml
        d = yaml.safe_load(io.open(CFG, encoding="utf-8").read())
        raw = d.get("counties") or {}
        out = []
        for city, items in raw.items():
            for fname, std in items.items():
                out.append((str(city), str(fname), str(std)))
        return out
    except Exception as e:
        raise SystemExit("读取 config.yaml 失败：%s" % e)


def load_centers():
    ctr = {}
    if CTR_FILE.exists():
        for r in csv.DictReader(io.open(CTR_FILE, encoding="utf-8")):
            ctr[r["name"]] = (float(r["lon"]), float(r["lat"]))
    return ctr


def resolve_coords(counties, centers):
    """为每个县确定经纬度：精确 -> 归一 -> 别名 -> 同城市均值兜底"""
    nidx = {norm(k): k for k in centers}
    city_pts = {}
    resolved = {}
    for city, fname, std in counties:
        key = None
        if std in centers:
            key = std
        elif norm(std) in nidx:
            key = nidx[norm(std)]
        elif ALIAS.get(std) in centers:
            key = ALIAS[std]
        elif norm(ALIAS.get(std, "")) in nidx:
            key = nidx[norm(ALIAS[std])]
        if key:
            resolved[(city, fname)] = centers[key]
            city_pts.setdefault(city, []).append(centers[key])
    # 兜底：同城市已解析县的平均坐标
    for city, fname, std in counties:
        if (city, fname) in resolved:
            continue
        pts = city_pts.get(city)
        if pts:
            lon = sum(p[0] for p in pts) / len(pts)
            lat = sum(p[1] for p in pts) / len(pts)
            resolved[(city, fname)] = (round(lon, 5), round(lat, 5))
        else:
            resolved[(city, fname)] = (113.65, 34.76)  # 郑州兜底
    return resolved


def fetch_one(lon, lat, start, end, retries=4, timeout=45):
    url = ("%s?parameters=%s&community=AG&longitude=%.5f&latitude=%.5f"
           "&start=%s&end=%s&format=CSV" % (API, PARAMS, lon, lat, start, end))
    last = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "wheat-yield/1.0"})
            with urllib.request.urlopen(req, timeout=timeout, context=_ssl_ctx()) as r:
                body = r.read().decode("utf-8", "ignore")
            return body
        except Exception as e:
            last = e
            time.sleep(2.0 * (attempt + 1) + 0.5)
    raise last


def parse_power_csv(body):
    """解析 NASA POWER CSV，返回 [(year, doy, tmax, tmin, t2m, precip, wind, rad, dew, rh)]"""
    lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
    # 找到表头行（以 YEAR,DOY 开头）
    hidx = None
    for i, ln in enumerate(lines):
        if ln.upper().startswith("YEAR,DOY"):
            hidx = i
            break
    if hidx is None:
        return [], []
    cols = [c.strip().upper() for c in lines[hidx].split(",")]
    idx = {c: i for i, c in enumerate(cols)}
    need = ["YEAR", "DOY", "T2M_MAX", "T2M_MIN", "T2M", "PRECTOTCORR"]
    if any(k not in idx for k in need):
        raise ValueError("返回缺少必要列：%s" % [k for k in need if k not in idx])

    rows = []
    dropped = [0]
    for ln in lines[hidx + 1:]:
        parts = ln.split(",")
        if len(parts) < len(cols):
            continue
        try:
            def g(k, default="-999"):
                i = idx.get(k)
                return parts[i] if i is not None and i < len(parts) else default
            core = [g("T2M_MAX"), g("T2M_MIN"), g("T2M"), g("PRECTOTCORR")]
            if any("-999" in str(v) for v in core):
                dropped[0] += 1          # 该日上游尚无有效数据，整行丢弃
                continue
            rows.append((
                int(float(g("YEAR"))), int(float(g("DOY"))),
                g("T2M_MAX"), g("T2M_MIN"), g("T2M"), g("PRECTOTCORR"),
                g("WS2M"), g("ALLSKY_SFC_SW_DWN"), g("T2MDEW"), g("RH2M"),
            ))
        except Exception:
            continue
    return rows, cols, dropped[0]


def write_county(city, fname, rows):
    d = OUT_DIR / city
    d.mkdir(parents=True, exist_ok=True)
    p = d / ("%s.csv" % fname)
    with io.open(p, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(HEADER_CN)
        for (yr, doy, tmax, tmin, t2m, pr, ws, rad, dew, rh) in rows:
            w.writerow([yr, doy, tmax, tmin, t2m, pr, ws, rad, dew, rh])
    return p


def main():
    ap = argparse.ArgumentParser(description="拉取当期逐日气象（NASA POWER）")
    ap.add_argument("--days", type=int, default=14, help="最近多少天（默认 14）")
    ap.add_argument("--start", type=str, default=None, help="起始日期 YYYYMMDD")
    ap.add_argument("--end", type=str, default=None, help="结束日期 YYYYMMDD")
    ap.add_argument("--county", nargs="*", default=None, help="只拉指定县（标准地名）")
    ap.add_argument("--city", nargs="*", default=None, help="只拉指定城市")
    ap.add_argument("--workers", type=int, default=3, help="并发数（默认 3）")
    ap.add_argument("--list", action="store_true", help="只列出将拉取的县，不发请求")
    ap.add_argument("--out", type=str, default=None,
                    help="输出目录（默认 data/raw/weather_recent）")
    ap.add_argument("--retry-failed", action="store_true",
                    help="只重跑上次失败的县，并沿用上次的日期区间")
    args = ap.parse_args()

    global OUT_DIR
    if args.out:
        OUT_DIR = Path(args.out).resolve()

    counties = load_counties()
    centers = load_centers()
    coords = resolve_coords(counties, centers)

    # 日期区间
    today = date.today()
    prev_meta = None
    if args.retry_failed:
        mp = OUT_DIR / "_fetch_meta.json"
        if mp.exists():
            prev_meta = json.loads(mp.read_text(encoding="utf-8"))
            fails = [x[1] for x in prev_meta.get("fail_detail", [])]
            if not fails:
                print("上次没有失败记录，无需重跑")
                return
            args.county = fails
            args.start, args.end = prev_meta["range"][0], prev_meta["range"][1]
            tail = "…" if len(fails) > 8 else ""
            print("重跑上次失败的 %d 个县：%s%s" % (len(fails), "、".join(fails[:8]), tail))

    if args.start and args.end:
        start, end = args.start, args.end
    else:
        e = today - timedelta(days=3)          # NASA POWER 有 2—3 天延迟
        s = e - timedelta(days=args.days - 1)
        start, end = s.strftime("%Y%m%d"), e.strftime("%Y%m%d")

    # 过滤
    picks = []
    for city, fname, std in counties:
        if args.county and std not in set(args.county):
            continue
        if args.city and city not in set(args.city):
            continue
        picks.append((city, fname, std))

    print("=" * 62)
    print("区间：%s — %s   县数：%d" % (start, end, len(picks)))
    print("输出：%s" % OUT_DIR)
    if args.list:
        for city, fname, std in picks:
            lon, lat = coords[(city, fname)]
            print("  %-8s %-12s (%.4f, %.4f)" % (city, std, lon, lat))
        return

    ok, fail = [], []
    drop_total = [0]
    done = 0

    def work(item):
        time.sleep(0.25)
        city, fname, std = item
        lon, lat = coords[(city, fname)]
        body = fetch_one(lon, lat, start, end)
        rows, cols, dropped = parse_power_csv(body)
        if len(rows) < 3:
            raise ValueError("有效日期不足 3 天（%d 行，丢弃 %d 行）" % (len(rows), dropped))
        p = write_county(city, fname, rows)
        return (city, fname, std, len(rows), dropped)

    t0 = datetime.now()
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        fut = {ex.submit(work, it): it for it in picks}
        for f in as_completed(fut):
            done += 1
            it = fut[f]
            try:
                city, fname, std, n, dropped = f.result()
                ok.append((city, std, n))
                if dropped:
                    drop_total[0] += dropped
                if done % 20 == 0 or done == len(picks):
                    print("  进度 %d/%d" % (done, len(picks)))
            except Exception as e:
                fail.append((it[0], it[2], str(e)[:70]))
                print("  [失败] %s %s -> %s" % (it[0], it[2], str(e)[:70]))

    meta = {
        "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "range": [start, end],
        "source": "NASA POWER daily point API (community=AG)",
        "params": PARAMS.split(","),
        "counties_total": (prev_meta.get("counties_total") if prev_meta else len(picks)),
        "ok": ((len(ok) + int(prev_meta.get("ok", 0))) if prev_meta else len(ok)),
        "failed": len(fail),
        "fail_detail": fail,
        "rows_total": sum(n for _, _, n in ok) + (prev_meta.get("rows_total", 0) if prev_meta else 0),
        "rows_dropped_999": drop_total[0],
        "elapsed_sec": round((datetime.now() - t0).total_seconds(), 1),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "_fetch_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")

    print("=" * 62)
    print("成功 %d / %d，共 %d 行（丢弃上游未就绪日期 %d 行），用时 %.1f 秒" % (
        len(ok), len(picks), meta["rows_total"], drop_total[0], meta["elapsed_sec"]))
    if fail:
        print("失败 %d 个（前 5）：" % len(fail))
        for x in fail[:5]:
            print("   ", x)
        sys.exit(1)
    print("下一步：打开「上传数据」页，选对应县上传 %s 下的 CSV" % OUT_DIR)


if __name__ == "__main__":
    main()
