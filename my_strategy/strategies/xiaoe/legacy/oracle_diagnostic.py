#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2026/5/16
# @Author  : david_van
# @Desc    : Oracle 诊断 —— 池内 top_n=1 的理论收益上下界
#
# 目的: 回答"问题在策略还是在池子"
#   - oracle  : 每次调仓选未来 N 日涨幅最大的（上界）
#   - worst   : 每次调仓选未来 N 日涨幅最小的（下界）
#   - median  : 池内中位数收益（随机猜的期望）
#   - equal   : 池中等权（无选股能力）
#   - first   : 永远买池中字典序第 1 个（无信号基线）
#
# 直接读 bundle 的 H5，跑得快，无需启动 RQAlpha。

import bisect
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from my_strategy.common_file import project_root

# ==================== 配置 ====================
BUNDLE = r"D:\datas\bundle"
POOL_DIR = Path(project_root) / "my_strategy" / "strategies" / "xiaoe_articles"
RESULT_DIR = Path(project_root) / 'my_strategy' / 'strategies' / 'batch_results' / 'lihai_pool'
RESULT_DIR.mkdir(parents=True, exist_ok=True)

START = "2020-01-01"
END = "2026-01-01"
BENCHMARK = "000300.XSHG"

# 调仓周期（交易日数），多个值会同时跑
REBAL_DAYS_LIST = [5, 10, 21, 63]   # 周/双周/月/季


# ==================== 数据加载 ====================
def load_pool_timeline(pool_dir):
    """从 CSV 加载 {date_str -> [codes]}"""
    pool_map = {}
    for f in Path(pool_dir).glob("*holdings*.csv"):
        df = pd.read_csv(f, encoding="utf-8-sig")
        for _, row in df.iterrows():
            codes_str = str(row.get("持有股票代码", ""))
            if not codes_str or codes_str == "nan":
                continue
            codes = [c.strip() for c in codes_str.split("|") if c.strip()]
            if codes:
                pool_map[str(row["日期"]).strip()] = codes
    return pool_map


def get_pool(pool_map, sorted_dates, date_str):
    """≤date 的最新池子"""
    idx = bisect.bisect_right(sorted_dates, date_str) - 1
    return pool_map[sorted_dates[idx]] if idx >= 0 else None


def load_adj_close(code, source='stocks.h5'):
    """读复权后(累计因子)收盘价，返回 Series indexed by date"""
    path = os.path.join(BUNDLE, source)
    with h5py.File(path, 'r') as f:
        if code not in f:
            return None
        arr = f[code][:]
    df = pd.DataFrame(arr)
    df['date'] = pd.to_datetime(df['datetime'] // 1000000, format='%Y%m%d')
    df = df[['date', 'close']].copy()

    if source == 'stocks.h5':
        with h5py.File(os.path.join(BUNDLE, 'ex_cum_factor.h5'), 'r') as f:
            if code in f:
                fac = pd.DataFrame(f[code][:])
                # start_date=0 表示自始 —— 转换后用 1900-01-01 占位
                ymd = (fac['start_date'] // 1000000).astype('int64')
                ymd = ymd.replace(0, 19000101)
                fac['date'] = pd.to_datetime(ymd, format='%Y%m%d', errors='coerce')
                fac = fac.dropna(subset=['date'])[['date', 'ex_cum_factor']].sort_values('date')
                df = pd.merge_asof(df.sort_values('date'), fac, on='date')
                df['ex_cum_factor'] = df['ex_cum_factor'].fillna(1.0)
                df['close'] = df['close'] * df['ex_cum_factor']
    return df.set_index('date')['close'].sort_index()


# ==================== Oracle 计算 ====================
def run_mode(pool_map, sorted_dates, price_data, trading_dates, rebal_days, mode):
    """
    返回:
      eq:  净值 Series (再平衡日索引)
      log: 每期记录 DataFrame
    """
    rebal_idx = list(range(0, len(trading_dates), rebal_days))
    if rebal_idx[-1] != len(trading_dates) - 1:
        rebal_idx.append(len(trading_dates) - 1)

    equity = [1.0]
    log = []
    skipped = 0
    for i in range(len(rebal_idx) - 1):
        t0 = trading_dates[rebal_idx[i]]
        t1 = trading_dates[rebal_idx[i + 1]]
        pool = get_pool(pool_map, sorted_dates, t0.strftime('%Y-%m-%d'))
        if not pool:
            equity.append(equity[-1])
            skipped += 1
            continue

        rets = {}
        for code in pool:
            s = price_data.get(code)
            if s is None or len(s) == 0:
                continue
            try:
                p0 = s.asof(t0)
                p1 = s.asof(t1)
                if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                    rets[code] = float(p1 / p0 - 1)
            except Exception:
                continue

        if not rets:
            equity.append(equity[-1])
            skipped += 1
            continue

        if mode == 'oracle':
            pick = max(rets, key=rets.get); ret = rets[pick]
        elif mode == 'worst':
            pick = min(rets, key=rets.get); ret = rets[pick]
        elif mode == 'median':
            ret = float(np.median(list(rets.values()))); pick = '__median__'
        elif mode == 'equal':
            ret = float(np.mean(list(rets.values()))); pick = '__equal__'
        elif mode == 'first':
            pick = sorted(rets.keys())[0]; ret = rets[pick]
        else:
            raise ValueError(mode)

        equity.append(equity[-1] * (1 + ret))
        log.append({
            'period_start': t0, 'period_end': t1,
            'pool_size': len(pool), 'pick': pick, 'ret': ret,
        })

    eq_dates = [trading_dates[rebal_idx[i]] for i in range(len(equity))]
    eq = pd.Series(equity, index=eq_dates)
    return eq, pd.DataFrame(log), skipped


def stats(eq):
    total = eq.iloc[-1] - 1
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    ann = (1 + total) ** (1 / years) - 1 if years > 0 else 0
    peak = eq.cummax()
    dd = ((eq - peak) / peak).min()
    rets = eq.pct_change().dropna()
    sharpe = (rets.mean() / rets.std() * np.sqrt(252 / max(1, (eq.index[1] - eq.index[0]).days))) if len(rets) > 1 and rets.std() > 0 else 0
    return total, ann, dd, sharpe


# ==================== 主流程 ====================
def main():
    print("=" * 70)
    print("Oracle 诊断: 池内 top_n=1 的理论上下界")
    print("=" * 70)

    # 1. 池子时间线
    pool_map = load_pool_timeline(POOL_DIR)
    sorted_dates = sorted(pool_map.keys())
    all_codes = set()
    for codes in pool_map.values():
        all_codes.update(codes)
    print(f"\n池子涉及股票总数: {len(all_codes)}")
    print(f"池子时间区间    : {sorted_dates[0]} ~ {sorted_dates[-1]}")
    print(f"快照数          : {len(sorted_dates)}")

    # 2. 加载价格
    print(f"\n加载 {len(all_codes)} 只股票的复权收盘价...")
    price_data = {}
    missing = []
    for code in sorted(all_codes):
        s = load_adj_close(code)
        if s is None or len(s) == 0:
            missing.append(code)
            continue
        price_data[code] = s[(s.index >= pd.Timestamp(START)) & (s.index <= pd.Timestamp(END))]
    if missing:
        print(f"  ⚠️ bundle 缺失 {len(missing)} 只: {missing[:5]}{'...' if len(missing)>5 else ''}")

    # 3. 构造交易日序列（用 000300 当基准日历）
    bench = load_adj_close(BENCHMARK, source='indexes.h5')
    bench = bench[(bench.index >= pd.Timestamp(START)) & (bench.index <= pd.Timestamp(END))]
    trading_dates = list(bench.index)
    print(f"交易日总数      : {len(trading_dates)}")

    # 4. 跑五个模式 × 多个频率
    all_results = {}      # {rebal: {mode: eq}}
    all_summaries = []
    for rebal in REBAL_DAYS_LIST:
        print(f"\n========== 调仓周期 = {rebal} 个交易日 ==========")
        per = {}
        for mode in ['oracle', 'equal', 'median', 'first', 'worst']:
            eq, log, skipped = run_mode(pool_map, sorted_dates, price_data, trading_dates, rebal, mode)
            t, a, d, sh = stats(eq)
            per[mode] = eq
            all_summaries.append({
                '调仓周期(日)': rebal, '模式': mode,
                '总收益': f"{t:.2%}", '年化': f"{a:.2%}",
                '最大回撤': f"{d:.2%}", '夏普(粗)': f"{sh:.2f}",
                '空仓期数': skipped,
            })
            print(f"  {mode:>7s}: 总{t:>8.2%}  年{a:>7.2%}  回{d:>7.2%}  空{skipped}")
            # 保存 oracle 的 pick 明细
            if mode == 'oracle' and rebal == 21:
                log.to_csv(RESULT_DIR / 'oracle_picks_21d.csv', index=False, encoding='utf-8-sig')
        all_results[rebal] = per

    # 5. 加上基准 + 实际策略
    bench_norm = bench / bench.iloc[0]
    bt, ba, bd, bs = stats(bench_norm)
    all_summaries.append({
        '调仓周期(日)': '-', '模式': 'benchmark(000300)',
        '总收益': f"{bt:.2%}", '年化': f"{ba:.2%}",
        '最大回撤': f"{bd:.2%}", '夏普(粗)': f"{bs:.2f}",
        '空仓期数': 0,
    })
    all_summaries.append({
        '调仓周期(日)': '-', '模式': 'actual_strategy',
        '总收益': '-2.69%', '年化': '-0.47%',
        '最大回撤': '-80.45%', '夏普(粗)': '0.20',
        '空仓期数': '-',
    })

    summary_df = pd.DataFrame(all_summaries)
    print("\n\n========== 汇总 ==========")
    print(summary_df.to_string(index=False))
    csv_path = RESULT_DIR / 'oracle_summary.csv'
    summary_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n汇总 CSV: {csv_path}")

    # 6. 画图（以 21d 为主图，画 5 个模式 + 基准）
    main_rebal = 21
    plt.figure(figsize=(14, 7))
    colors = {'oracle': '#d62728', 'equal': '#1f77b4', 'median': '#9467bd',
              'first': '#8c564b', 'worst': '#7f7f7f'}
    for mode, eq in all_results[main_rebal].items():
        plt.plot(eq.index, eq.values, label=mode, linewidth=1.6, color=colors.get(mode))
    plt.plot(bench_norm.index, bench_norm.values, label='benchmark 沪深300',
             linewidth=1.6, linestyle='--', color='black')
    # 标实际策略终值
    plt.axhline(y=1 - 0.0269, color='orange', linestyle=':', linewidth=1.5,
                label='actual_strategy 终值 (-2.69%)')
    plt.title(f'Oracle 诊断: 池内 top_n=1 收益上下界 (调仓 {main_rebal} 日)', fontsize=13)
    plt.xlabel('Date'); plt.ylabel('净值 (start=1.0)')
    plt.legend(loc='upper left', fontsize=10); plt.grid(True, alpha=0.3)
    plt.yscale('log')
    png_path = RESULT_DIR / 'oracle_diagnostic.png'
    plt.savefig(png_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"净值图  : {png_path}")

    # 7. 调仓频率对 oracle 上界的影响图
    plt.figure(figsize=(12, 6))
    for rebal in REBAL_DAYS_LIST:
        eq = all_results[rebal]['oracle']
        plt.plot(eq.index, eq.values, label=f'oracle_rebal{rebal}d', linewidth=1.4)
    plt.plot(bench_norm.index, bench_norm.values, label='benchmark', linestyle='--', color='black')
    plt.title('Oracle 上界 vs 调仓频率', fontsize=13)
    plt.xlabel('Date'); plt.ylabel('净值 (start=1.0, log)')
    plt.legend(); plt.grid(True, alpha=0.3); plt.yscale('log')
    png_path2 = RESULT_DIR / 'oracle_by_rebal.png'
    plt.savefig(png_path2, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"频率图  : {png_path2}")


if __name__ == "__main__":
    main()
