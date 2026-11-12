# -*- coding: utf-8 -*-
"""
用映射表重新对齐检查
"""
import re
import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2] / "wheat-ml"
WEATHER_DIR = PROJECT_ROOT / "data" / "raw" / "weather"
YIELD_CSV = PROJECT_ROOT / "data" / "raw" / "source_yield" / "河南省小麦产量标签_总表.csv"
sys.path.insert(0, str(Path(__file__).parent))
from county_mapping import map_weather_to_yield, should_skip_yield

yield_df = pd.read_csv(YIELD_CSV, encoding="utf-8-sig")

# 扫描气象文件
weather_dirs = [d for d in WEATHER_DIR.iterdir() if d.is_dir()]
weather_records = []  # (市, 气象文件名, 映射后区县名, 路径)

for city_dir in sorted(weather_dirs):
    city = city_dir.name
    for csv_file in sorted(city_dir.glob("*.csv")):
        raw = csv_file.stem
        clean = re.sub(r"气象$", "", raw)
        mapped = map_weather_to_yield(city, clean)
        weather_records.append((city, clean, mapped, str(csv_file)))

# 按市对齐
print("=" * 70)
print("映射后对齐检查")
print("=" * 70)

total_matched = 0
total_unmatched = 0

for city in sorted(set(w[0] for w in weather_records)):
    w_list = [w for w in weather_records if w[0] == city]
    y_counties = set(yield_df[yield_df["城市"] == city]["区县"].unique())
    
    if not y_counties:
        print(f"\n【{city}】无产量数据（气象文件 {len(w_list)} 个）")
        continue
    
    w_mapped = set(w[2] for w in w_list if w[2] is not None)
    
    only_weather = w_mapped - y_counties
    only_yield = y_counties - w_mapped
    
    # 过滤掉明确跳过的
    only_yield_filtered = {c for c in only_yield if not should_skip_yield(city, c)}
    
    status = "OK" if not only_weather and not only_yield_filtered else "WARN"
    print(f"\n【{city}】{status}")
    print(f"  产量区县: {len(y_counties)}, 气象文件: {len(w_list)}")
    
    if only_weather:
        print(f"  有气象但产量无: {sorted(only_weather)}")
    if only_yield_filtered:
        print(f"  有产量但无气象: {sorted(only_yield_filtered)}")
    if only_yield - only_yield_filtered:
        skipped = only_yield - only_yield_filtered
        print(f"  (已跳过的城区/开发区): {sorted(skipped)}")

print()
print("=" * 70)
print("对齐完成。")
