#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author : david_van
# @Desc   : Walk-Forward 验证 —— m_days 是否是真信号
#
# 思路：用过去 L 年训练（选最优 m_days），下一年 OOS。
# 对比：
#   - 多种训练窗口长度 (1/2/3 年)
#   - 多种选优指标 (累计收益 / 平均年化 / 类Sharpe / 最差年)
#   - 固定 m=60 / m=75 基线
#   - 完美预知 oracle (每年选最优，事后偏差，仅作上界)
#   - 基准沪深300
#
# 输入：batch_results/lihai_pool/dyn_m*.pkl

import pickle
from pathlib import Path

from my_strategy.common_file import project_root

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULT_DIR = Path(project_root) / 'my_strategy' / 'strategies' / 'batch_results' / 'lihai_pool'


# ==================== 数据加载 ====================
def load_data():
    """返回:
       yearly: {m_days -> {year -> ret}}
       bench_yearly: {year -> bench_ret}
    """
    yearly = {}
    bench_yearly = None
    for p in sorted(RESULT_DIR.glob('dyn_m*.pkl')):
        with open(p, 'rb') as f:
            data = pickle.load(f)
        nv = data['portfolio']['unit_net_value'].sort_index()
        m = int(p.stem.replace('dyn_m', ''))
        yearly[m] = {y: float(g.iloc[-1] / g.iloc[0] - 1) for y, g in nv.groupby(nv.index.year)}
        if bench_yearly is None:
            bench_nv = data['portfolio'].get('benchmark_unit_net_value')
            if bench_nv is not None:
                bench_nv = bench_nv.sort_index()
                bench_yearly = {y: float(g.iloc[-1] / g.iloc[0] - 1)
                                for y, g in bench_nv.groupby(bench_nv.index.year)}
    return yearly, bench_yearly


# ==================== 选优指标 ====================
def score_metric(rets, metric):
    if not rets:
        return -np.inf
    arr = np.array(rets)
    if metric == 'cum':       # 累计收益
        return float(np.prod(1 + arr) - 1)
    elif metric == 'avg':     # 算术平均
        return float(arr.mean())
    elif metric == 'sharpe':  # 类 Sharpe
        return float(arr.mean() / (arr.std() + 1e-9))
    elif metric == 'min':     # 最差年（最稳健）
        return float(arr.min())
    raise ValueError(metric)


def pick_best_m(yearly, train_years, metric):
    scores = {}
    for m, yr in yearly.items():
        rets = [yr[y] for y in train_years if y in yr]
        scores[m] = score_metric(rets, metric)
    best = max(scores, key=scores.get)
    return best, scores[best]


# ==================== Walk-Forward ====================
def run_wf(yearly, all_years, train_len, metric):
    rows = []
    for y in all_years:
        train = [yr for yr in all_years if yr < y][-train_len:]
        if len(train) < min(train_len, 1):
            continue
        m_best, train_score = pick_best_m(yearly, train, metric)
        oos = yearly[m_best].get(y, np.nan)
        rows.append({
            'oos_year': y, 'train': f"{train[0]}-{train[-1]}",
            'm_pick': m_best, 'train_score': train_score, 'oos_ret': oos,
        })
    return pd.DataFrame(rows)


def cumret_stats(yearly_rets):
    arr = np.array([r for r in yearly_rets if not np.isnan(r)])
    cum = float(np.prod(1 + arr) - 1)
    avg = float(arr.mean()) if len(arr) else 0
    worst = float(arr.min()) if len(arr) else 0
    return cum, avg, worst


# ==================== 主流程 ====================
def main():
    yearly, bench_yearly = load_data()
    all_years = sorted({y for yr in yearly.values() for y in yr.keys()})
    m_list = sorted(yearly.keys())
    print(f"可用年份: {all_years}")
    print(f"可用 m_days: {m_list}")
    print(f"基准年度收益: {{{', '.join(f'{y}:{r*100:+.1f}%' for y, r in (bench_yearly or {}).items())}}}\n")

    # --- 1. Walk-forward 多方案 ---
    print('=' * 78)
    print('Walk-Forward 多方案 OOS 表现')
    print('=' * 78)
    schemes = {}
    for L in [1, 2, 3]:
        for metric in ['cum', 'avg', 'sharpe', 'min']:
            tag = f"L{L}_{metric}"
            wf = run_wf(yearly, all_years, L, metric)
            if wf.empty:
                continue
            schemes[tag] = wf
            cum, avg, worst = cumret_stats(wf['oos_ret'].tolist())
            n = len(wf)
            print(f"\n[{tag}]  累计OOS:{cum*100:+8.1f}%  平均年:{avg*100:+6.1f}%  "
                  f"最差年:{worst*100:+6.1f}%  覆盖{n}年")
            line = "  " + "  ".join(
                f"{int(r.oos_year)}:m{int(r.m_pick):03d}={r.oos_ret*100:+5.1f}%"
                for r in wf.itertuples()
            )
            print(line)

    # --- 2. Oracle (事后偏差，仅作上界) ---
    oracle_rows = []
    for y in all_years:
        best_m = max(yearly, key=lambda m: yearly[m].get(y, -np.inf))
        oracle_rows.append({'oos_year': y, 'm_pick': best_m, 'oos_ret': yearly[best_m].get(y)})
    oracle = pd.DataFrame(oracle_rows)
    o_cum, o_avg, o_worst = cumret_stats(oracle['oos_ret'].tolist())
    print(f"\n[ORACLE]  累计:{o_cum*100:+8.1f}%  平均年:{o_avg*100:+6.1f}%  最差年:{o_worst*100:+6.1f}%  ← 完美预知上界")
    print("  " + "  ".join(
        f"{int(r.oos_year)}:m{int(r.m_pick):03d}={r.oos_ret*100:+5.1f}%"
        for r in oracle.itertuples()
    ))

    # --- 3. 固定 m_days 在同一 OOS 范围 ---
    # 取最短 WF 方案的覆盖年作为公平比较窗口
    oos_window = sorted(schemes[min(schemes, key=lambda k: len(schemes[k]))]['oos_year'].tolist())
    print('\n' + '=' * 78)
    print(f'同窗口对比 (OOS = {oos_window})')
    print('=' * 78)
    rows = []
    for tag, wf in schemes.items():
        in_win = wf[wf['oos_year'].isin(oos_window)]['oos_ret'].tolist()
        cum, avg, worst = cumret_stats(in_win)
        rows.append({'方案': tag, '累计': f"{cum*100:+.1f}%", '平均': f"{avg*100:+.1f}%", '最差': f"{worst*100:+.1f}%"})
    # 固定 m
    for m in [50, 60, 65, 75, 85]:
        rets = [yearly[m].get(y, 0) for y in oos_window]
        cum, avg, worst = cumret_stats(rets)
        rows.append({'方案': f'固定 m={m}', '累计': f"{cum*100:+.1f}%",
                     '平均': f"{avg*100:+.1f}%", '最差': f"{worst*100:+.1f}%"})
    # Oracle in window
    o_in = oracle[oracle['oos_year'].isin(oos_window)]['oos_ret'].tolist()
    cum, avg, worst = cumret_stats(o_in)
    rows.append({'方案': 'oracle(事后)', '累计': f"{cum*100:+.1f}%",
                 '平均': f"{avg*100:+.1f}%", '最差': f"{worst*100:+.1f}%"})
    # 基准
    if bench_yearly:
        bench_in = [bench_yearly.get(y, 0) for y in oos_window]
        cum, avg, worst = cumret_stats(bench_in)
        rows.append({'方案': 'benchmark(沪深300)', '累计': f"{cum*100:+.1f}%",
                     '平均': f"{avg*100:+.1f}%", '最差': f"{worst*100:+.1f}%"})
    cmp_df = pd.DataFrame(rows)
    print(cmp_df.to_string(index=False))

    # --- 4. 画图 ---
    plt.figure(figsize=(14, 7))
    # WF 方案（聚焦最长可用窗口的方案）
    show_schemes = ['L2_cum', 'L2_min', 'L3_cum', 'L3_min', 'L3_sharpe']
    for tag in show_schemes:
        if tag not in schemes:
            continue
        wf = schemes[tag]
        wf = wf[wf['oos_year'].isin(oos_window)]
        ys = [oos_window[0] - 1] + wf['oos_year'].tolist()
        nv = [1.0]
        cur = 1.0
        for r in wf['oos_ret']:
            cur *= (1 + r); nv.append(cur)
        plt.plot(ys, nv, marker='o', linewidth=1.5, label=f'WF_{tag}')

    # 固定基准
    for m, color in [(50, '#d62728'), (60, '#1f77b4'), (75, '#2ca02c')]:
        rets = [yearly[m].get(y, 0) for y in oos_window]
        nv = [1.0]; cur = 1.0
        for r in rets: cur *= (1 + r); nv.append(cur)
        plt.plot([oos_window[0]-1]+oos_window, nv, marker='s', linestyle='--',
                 linewidth=1.4, color=color, label=f'fixed m={m}', alpha=0.8)

    # oracle
    o_in = oracle[oracle['oos_year'].isin(oos_window)]['oos_ret'].tolist()
    nv = [1.0]; cur = 1.0
    for r in o_in: cur *= (1 + r); nv.append(cur)
    plt.plot([oos_window[0]-1]+oos_window, nv, marker='*', linestyle=':',
             linewidth=2, color='gold', label='oracle (look-ahead)')

    # benchmark
    if bench_yearly:
        rets = [bench_yearly.get(y, 0) for y in oos_window]
        nv = [1.0]; cur = 1.0
        for r in rets: cur *= (1 + r); nv.append(cur)
        plt.plot([oos_window[0]-1]+oos_window, nv, marker='x', linestyle='-.',
                 linewidth=1.4, color='black', label='benchmark')

    plt.xlabel('Year'); plt.ylabel('累计净值 (start=1.0)')
    plt.title('Walk-Forward OOS 累计收益对比')
    plt.legend(loc='upper left', fontsize=9, ncol=2); plt.grid(alpha=0.3)
    plt.yscale('log')
    out_png = RESULT_DIR / 'walk_forward.png'
    plt.savefig(out_png, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'\n图: {out_png}')

    # --- 5. 详细 CSV ---
    detail = pd.concat([df.assign(scheme=tag) for tag, df in schemes.items()],
                       ignore_index=True)
    detail.to_csv(RESULT_DIR / 'walk_forward_details.csv', index=False, encoding='utf-8-sig')
    cmp_df.to_csv(RESULT_DIR / 'walk_forward_summary.csv', index=False, encoding='utf-8-sig')
    print(f'明细: {RESULT_DIR / "walk_forward_details.csv"}')
    print(f'汇总: {RESULT_DIR / "walk_forward_summary.csv"}')

    # --- 6. 结论 ---
    print('\n' + '=' * 78)
    print('结论判读')
    print('=' * 78)
    # 用 L2_cum 作为最常见的 WF
    if 'L2_cum' in schemes:
        wf = schemes['L2_cum']
        in_win = wf[wf['oos_year'].isin(oos_window)]['oos_ret'].tolist()
        wf_cum, _, _ = cumret_stats(in_win)
        m60_cum, _, _ = cumret_stats([yearly[60].get(y, 0) for y in oos_window])
        bench_cum, _, _ = cumret_stats([bench_yearly.get(y, 0) for y in oos_window]) if bench_yearly else (0, 0, 0)
        oracle_cum = float(np.prod(1 + np.array(o_in)) - 1)

        print(f"在 {oos_window[0]}-{oos_window[-1]} OOS 区间：")
        print(f"  WF (L2_cum)   累计 {wf_cum*100:+.1f}%")
        print(f"  固定 m=60     累计 {m60_cum*100:+.1f}%")
        print(f"  oracle 上界   累计 {oracle_cum*100:+.1f}%")
        print(f"  基准沪深300   累计 {bench_cum*100:+.1f}%")
        if wf_cum > m60_cum and wf_cum > bench_cum:
            print("\n  [PASS] WF 击败固定 m=60 和基准，m_days 选择有真实可捕捉的信号")
        elif wf_cum > bench_cum:
            print("\n  [WARN] WF 打过基准但不如固定 m=60 — 动态优化反而扣分")
        else:
            print("\n  [FAIL] WF 跑不过基准，m_days 最优是事后幻觉")


if __name__ == '__main__':
    main()
