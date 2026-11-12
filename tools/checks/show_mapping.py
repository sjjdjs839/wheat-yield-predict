# -*- coding: utf-8 -*-
import re, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from county_mapping import map_weather_to_yield

PROJECT_ROOT = Path(__file__).resolve().parents[2] / "wheat-ml"
WEATHER_DIR = PROJECT_ROOT / "data" / "raw" / "weather"
YIELD_CSV = PROJECT_ROOT / "data" / "raw" / "source_yield" / "河南省小麦产量标签_总表.csv"
yield_df = pd.read_csv(YIELD_CSV, encoding="utf-8-sig")

rows = []
for city_dir in sorted([d for d in WEATHER_DIR.iterdir() if d.is_dir()]):
    city = city_dir.name
    y_counties = set(yield_df[yield_df["城市"] == city]["区县"].unique())
    for f in sorted(city_dir.glob("*.csv")):
        raw = f.stem
        clean = re.sub(r"气象$", "", raw)
        mapped = map_weather_to_yield(city, clean)
        if mapped is None:
            status = "跳过(高新区)"
        elif mapped in y_counties:
            status = "匹配"
        else:
            status = "产量表无"
        rows.append((city, raw, mapped if mapped else "(跳过)", status))

for city in sorted(set(r[0] for r in rows)):
    sub = [r for r in rows if r[0] == city]
    y_count = len(set(yield_df[yield_df["城市"] == city]["区县"].unique()))
    print(f"=== {city} (产量标签{y_count}个区县, 气象{len(sub)}个文件) ===")
    for _, raw, mapped, status in sub:
        print(f"  {raw:<20} -> {mapped:<10} {status}")
    print()
