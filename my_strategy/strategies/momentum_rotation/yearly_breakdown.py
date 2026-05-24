#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 按年份拆解每个 m_days 实验的收益，看参数稳定性

import pickle
from pathlib import Path
import pandas as pd
import numpy as np

from my_strategy.common_file import project_root

RESULT_DIR = Path(project_root) / 'my_strategy' / 'strategies' / 'batch_results' / 'lihai_pool'


def yearly_return(nv: pd.Series) -> dict:
    """从单位净值算每年收益率"""
    nv = nv.sort_index()
    out = {}
    for y, g in nv.groupby(nv.index.year):
        out[y] = float(g.iloc[-1] / g.iloc[0] - 1)
    return out


def yearly_maxdd(nv: pd.Series) -> dict:
    out = {}
    for y, g in nv.groupby(nv.index.year):
        peak = g.cummax()
        out[y] = float(((g - peak) / peak).min())
    return out


def main():
    pkls = sorted(RESULT_DIR.glob('dyn_m*.pkl'))
    rows = []
    dd_rows = []
    for p in pkls:
        with open(p, 'rb') as f:
            data = pickle.load(f)
        nv = data['portfolio']['unit_net_value']
        bench_nv = data['portfolio'].get('benchmark_unit_net_value')
        ret = yearly_return(nv)
        dd = yearly_maxdd(nv)
        rows.append({'tag': p.stem, **ret})
        dd_rows.append({'tag': p.stem, **dd})
        if bench_nv is not None and 'bench' not in [r['tag'] for r in rows]:
            rows.append({'tag': '_benchmark_', **yearly_return(bench_nv)})
            dd_rows.append({'tag': '_benchmark_', **yearly_maxdd(bench_nv)})

    ret_df = pd.DataFrame(rows).set_index('tag').sort_index()
    dd_df = pd.DataFrame(dd_rows).set_index('tag').sort_index()

    # 把 benchmark 行排到底部
    if '_benchmark_' in ret_df.index:
        bench = ret_df.loc[['_benchmark_']]
        ret_df = pd.concat([ret_df.drop('_benchmark_'), bench])
        bench_dd = dd_df.loc[['_benchmark_']]
        dd_df = pd.concat([dd_df.drop('_benchmark_'), bench_dd])

    # 格式化
    print('\n========== 每年收益率 (%) ==========')
    print((ret_df * 100).round(1).to_string())

    print('\n========== 每年最大回撤 (%) ==========')
    print((dd_df * 100).round(1).to_string())

    # 稳定性指标：年度收益的标准差、最差年份、最好年份
    print('\n========== 参数稳定性 ==========')
    stable = pd.DataFrame({
        '年数': ret_df.notna().sum(axis=1),
        '平均年收益(%)': (ret_df.mean(axis=1) * 100).round(2),
        '中位年收益(%)': (ret_df.median(axis=1) * 100).round(2),
        '年收益std(%)': (ret_df.std(axis=1) * 100).round(2),
        '最差年份(%)': (ret_df.min(axis=1) * 100).round(2),
        '最差年份是': ret_df.idxmin(axis=1),
        '最好年份(%)': (ret_df.max(axis=1) * 100).round(2),
        '最好年份是': ret_df.idxmax(axis=1),
        '正年数': (ret_df > 0).sum(axis=1),
    })
    print(stable.to_string())

    # 保存
    out = RESULT_DIR / 'yearly_breakdown.csv'
    (ret_df * 100).round(2).to_csv(out, encoding='utf-8-sig')
    print(f'\n年度收益已保存: {out}')
    stable.to_csv(RESULT_DIR / 'yearly_stability.csv', encoding='utf-8-sig')


if __name__ == '__main__':
    main()
