from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2] / "wheat-ml"
yield_path = PROJECT_ROOT / "data" / "raw" / "county_yield.csv"
df = pd.read_csv(yield_path)

print("产量标签统计：")
print(f"总记录数：{len(df)}")
print(f"单产范围：{df['wheat_yield'].min():.1f} ~ {df['wheat_yield'].max():.1f} kg/亩")
print(f"平均单产：{df['wheat_yield'].mean():.1f} kg/亩")
print()

# 找异常值
print("=== 异常值（单产>700 kg/亩 或 <150 kg/亩）===")
abnormal = df[(df['wheat_yield'] > 700) | (df['wheat_yield'] < 150)]
print(abnormal.to_string())
print()

# 按县统计，看看同一个县的波动
print("=== 各县单产波动最大的前10个 ===")
county_stats = df.groupby('county')['wheat_yield'].agg(['mean', 'std', 'min', 'max'])
county_stats['range'] = county_stats['max'] - county_stats['min']
print(county_stats.sort_values('range', ascending=False).head(10))
