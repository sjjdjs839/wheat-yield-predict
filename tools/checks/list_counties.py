from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2] / "wheat-ml"
yield_path = PROJECT_ROOT / "data" / "raw" / "county_yield.csv"
df = pd.read_csv(yield_path)

# 列出所有区县，按首字母排序
counties = sorted(df['county'].unique())
print(f"共 {len(counties)} 个区县：")
for c in counties:
    count = len(df[df['county'] == c])
    mean = df[df['county'] == c]['wheat_yield'].mean()
    std = df[df['county'] == c]['wheat_yield'].std()
    print(f"  {c:10s}  {count:3d}条  平均{mean:6.1f}  标准差{std:5.1f}")
