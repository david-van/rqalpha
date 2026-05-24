#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""逐年拆解 equal_weight 策略收益，与博主收益对比"""
import pickle
import pandas as pd
from pathlib import Path

from my_strategy.common_file import project_root

RESULT_DIR = Path(project_root) / 'my_strategy' / 'strategies' / 'batch_results' / 'xiaoe_pool'

# 博主年收益
blogger = {
    2020: 0.8814, 2021: 0.2300, 2022: 0.1027,
    2023: 0.3191, 2024: 0.0457, 2025: 0.5805,
}

# 读 equal_weight 的 pkl
pkl_path = RESULT_DIR / 'equal_weight.pkl'
if not pkl_path.exists():
    print(f'找不到 {pkl_path}，请先跑 run_xiaoe.py')
    exit(1)

with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

pf = data['portfolio']
nv = pf['unit_net_value']  # Series, index=date

# 按年计算收益率
yearly = {}
for year in range(2020, 2026):
    mask = (nv.index >= f'{year}-01-01') & (nv.index <= f'{year}-12-31')
    yr = nv[mask]
    if len(yr) < 2:
        continue
    ret = (yr.iloc[-1] / yr.iloc[0]) - 1
    yearly[year] = ret

print('逐年收益率对比:')
print(f"{'年份':<6} {'equal_weight':>12} {'博主':>10} {'差值':>10}")
print('-' * 42)
for y in sorted(yearly):
    ew = yearly[y]
    bg = blogger.get(y, float('nan'))
    diff = ew - bg
    print(f'{y:<6} {ew:>12.2%} {bg:>10.2%} {diff:>+10.2%}')

# 累计
ew_cum = 1.0
bg_cum = 1.0
for y in sorted(yearly):
    ew_cum *= (1 + yearly[y])
    bg_cum *= (1 + blogger.get(y, 0))
print('-' * 42)
print(f'累计:   {ew_cum-1:.2%} (我们) vs {bg_cum-1:.2%} (博主)')
