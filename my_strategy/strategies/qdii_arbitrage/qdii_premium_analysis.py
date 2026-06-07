#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""QDII ETF 折溢价套利 — 信号分析脚本。

从缓存 CSV 加载 NAV + 日线数据，计算折溢价率，扫描买卖阈值，
输出策略表现的 CSV 汇总。

用法:
    # 默认：分析所有已缓存的 QDII ETF
    python qdii_premium_analysis.py

    # 指定 ETF
    python qdii_premium_analysis.py --etfs "513100.SH,513030.SH"

    # 自定义阈值扫描范围
    python qdii_premium_analysis.py --min-premium -0.05 --max-premium 0.05 --step 0.002

输出文件:
    - premium_overall.csv    各阈值下各 ETF 的整体表现
    - premium_forward.csv    各折溢价分档的 forward return
    - premium_cycles.csv     套利周期明细
    - top_premium_etfs.csv   最优 ETF 排名
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CACHE_DIR = Path(__file__).resolve().parent / 'cache'
OUT_DIR = Path(__file__).resolve().parent / 'analysis_results'


def load_etf_data(ts_code: str) -> pd.DataFrame | None:
    """加载单只 ETF 的 NAV 和日线，合并为带折溢价率的日线 DataFrame。"""
    nav_path = CACHE_DIR / f"{ts_code}_nav.csv"
    daily_path = CACHE_DIR / f"{ts_code}_daily.csv"

    if not nav_path.exists() or not daily_path.exists():
        return None

    nav = pd.read_csv(nav_path)
    daily = pd.read_csv(daily_path)

    if nav.empty or daily.empty:
        return None

    # 标准化日期列（整数 YYYYMMDD → datetime）
    nav['nav_date'] = pd.to_datetime(nav['nav_date'].astype(str), format='%Y%m%d')
    daily['trade_date'] = pd.to_datetime(daily['trade_date'].astype(str), format='%Y%m%d')

    # 净值去重（同一天可能多条）
    nav = nav.drop_duplicates(subset=['nav_date']).sort_values('nav_date')

    # 对齐：交易日取最近一个已公布的净值（nav_date <= trade_date）
    nav_sorted = nav[['nav_date', 'unit_nav', 'adj_nav']].copy()
    nav_sorted = nav_sorted.set_index('nav_date')

    daily = daily.sort_values('trade_date')

    # 对每个交易日找最近净值
    merged_rows = []
    nav_dates = nav_sorted.index.sort_values()
    for _, row in daily.iterrows():
        td = row['trade_date']
        # 找 <= td 的最大 nav_date
        mask = nav_dates <= td
        if not mask.any():
            continue
        best_nav_date = nav_dates[mask][-1]
        nav_row = nav_sorted.loc[best_nav_date]
        merged_rows.append({
            'trade_date': td,
            'close': float(row['close']),
            'open': float(row['open']),
            'pre_close': float(row['pre_close']),
            'vol': float(row['vol']),
            'amount': float(row['amount']),
            'nav_date': best_nav_date,
            'unit_nav': float(nav_row['unit_nav']),
            'adj_nav': float(nav_row['adj_nav']) if 'adj_nav' in nav_row else float(nav_row['unit_nav']),
        })

    if not merged_rows:
        return None

    df = pd.DataFrame(merged_rows)
    df['premium'] = df['close'] / df['unit_nav'] - 1.0  # >0 溢价, <0 折价
    df['gap'] = df['open'] - df['pre_close']            # 开盘跳空
    df['return'] = df['close'].pct_change()              # 当日收益率
    df['next_1d_ret'] = df['close'].shift(-1) / df['close'] - 1
    df['next_5d_ret'] = df['close'].shift(-5) / df['close'] - 1
    df['next_10d_ret'] = df['close'].shift(-10) / df['close'] - 1
    df['next_20d_ret'] = df['close'].shift(-20) / df['close'] - 1
    return df


def summarize_cycle_trades(trades: list[dict]) -> dict:
    """汇总一组套利周期的交易统计。"""
    if not trades:
        return {
            'n_cycles': 0, 'win_rate': 0, 'avg_return': 0,
            'total_return': 0, 'avg_hold_days': 0, 'profit_loss_ratio': np.nan,
        }
    returns = [t['cycle_return'] for t in trades]
    n = len(returns)
    gains = sum(r for r in returns if r > 0)
    losses = sum(-r for r in returns if r < 0)
    hold_days = [t.get('hold_days', 0) for t in trades]
    compounded = np.prod([1 + r for r in returns]) - 1
    return {
        'n_cycles': n,
        'win_rate': sum(1 for r in returns if r > 0) / n,
        'avg_return': np.mean(returns),
        'total_return': compounded,
        'avg_hold_days': np.mean(hold_days) if hold_days else 0,
        'profit_loss_ratio': gains / losses if losses > 0 else np.nan,
        'max_return': max(returns),
        'min_return': min(returns),
    }


def max_drawdown(returns: list[float]) -> float:
    if not returns:
        return 0.0
    eq = np.r_[1.0, np.cumprod(1 + np.array(returns))]
    peak = np.maximum.accumulate(eq)
    return float((eq / peak - 1).min())


def simulate_arbitrage(df: pd.DataFrame, buy_threshold: float,
                       sell_threshold: float, max_hold_days: int = 60) -> list[dict]:
    """模拟折溢价套利：折价买、溢价卖。

    buy_threshold: 买入条件 premium <= buy_threshold (<0 表示折价超过|threshold|)
    sell_threshold: 卖出条件 premium >= sell_threshold (>0 表示溢价超过threshold)
    max_hold_days: 最长持仓天数，超时强制平仓
    """
    cycles = []
    holding = False
    buy_idx = 0
    buy_price = 0.0

    for i in range(len(df)):
        prem = df['premium'].iloc[i]
        close = df['close'].iloc[i]
        trade_date = df['trade_date'].iloc[i]

        if not holding:
            # 等待买点：折价超过阈值
            if prem <= buy_threshold:
                holding = True
                buy_idx = i
                buy_price = close
        else:
            hold_days = i - buy_idx
            # 卖出条件：溢价超过阈值 或 持仓超时 或 到数据末尾
            should_sell = (
                prem >= sell_threshold
                or hold_days >= max_hold_days
                or i == len(df) - 1
            )
            if should_sell:
                cycle_return = close / buy_price - 1.0
                exit_reason = (
                    'take_profit' if prem >= sell_threshold
                    else 'timeout' if hold_days >= max_hold_days
                    else 'end_of_data'
                )
                cycles.append({
                    'buy_date': str(df['trade_date'].iloc[buy_idx])[:10],
                    'sell_date': str(trade_date)[:10],
                    'buy_price': buy_price,
                    'sell_price': close,
                    'hold_days': hold_days,
                    'cycle_return': cycle_return,
                    'exit_reason': exit_reason,
                    'buy_premium': df['premium'].iloc[buy_idx],
                    'sell_premium': prem,
                })
                holding = False

    return cycles


def build_threshold_table(dfs: dict[str, pd.DataFrame], args) -> pd.DataFrame:
    """遍历买卖阈值 × ETF，生成汇总表。"""
    buy_thresholds = np.arange(args.min_buy, args.max_buy + args.step / 2, args.step).round(6)
    sell_thresholds = np.arange(args.min_sell, args.max_sell + args.step / 2, args.step).round(6)

    # 优化：只扫描 buy < sell 的组合
    rows = []
    for bt in buy_thresholds:
        for st in sell_thresholds:
            if bt >= st:
                continue  # 买入阈值必须 < 卖出阈值
            for etf, df in dfs.items():
                if df is None or df.empty:
                    continue
                cycles = simulate_arbitrage(df, float(bt), float(st), args.max_hold)
                stats = summarize_cycle_trades(cycles)
                # 附加 ETF 维度指标
                n_days = len(df)
                years = n_days / 250
                ann_ret = (1 + stats['total_return']) ** (1 / years) - 1 if years > 0 and stats[
                    'total_return'] > -1 else np.nan
                returns = [c['cycle_return'] for c in cycles]
                dd = max_drawdown(returns) if returns else 0.0
                rows.append({
                    'buy_threshold': float(bt),
                    'sell_threshold': float(st),
                    'order_book_id': etf,
                    'years': years,
                    **stats,
                    'annualized_return': ann_ret,
                    'max_drawdown': dd,
                })

    return pd.DataFrame(rows)


def build_forward_table(dfs: dict[str, pd.DataFrame], args) -> pd.DataFrame:
    """按折溢价率分档，计算 forward return（1/5/10/20日）。"""
    rows = []
    buckets = np.arange(args.min_buy, args.max_sell + args.step / 2, args.step).round(6)

    for etf, df in dfs.items():
        if df is None or df.empty:
            continue
        for bucket in buckets:
            half = args.step / 2
            mask = (df['premium'] >= bucket - half) & (df['premium'] < bucket + half)
            subset = df[mask]
            if len(subset) < 5:
                continue
            row = {
                'order_book_id': etf,
                'premium_bucket': float(bucket),
                'n_days': len(subset),
                'avg_premium': float(subset['premium'].mean()),
            }
            for horizon in [1, 5, 10, 20]:
                col = f'next_{horizon}d_ret'
                if col in subset.columns:
                    valid = subset[col].dropna()
                    row[f'fwd_{horizon}d_avg'] = float(valid.mean()) if len(valid) > 0 else np.nan
                    row[f'fwd_{horizon}d_win'] = float((valid > 0).mean()) if len(valid) > 0 else np.nan
            rows.append(row)

    return pd.DataFrame(rows)


def build_cycles_detail(dfs: dict[str, pd.DataFrame], args) -> pd.DataFrame:
    """对每个 ETF 的最优阈值生成详细的套利周期记录。"""
    all_cycles = []
    for etf, df in dfs.items():
        if df is None or df.empty:
            continue
        # 使用默认阈值
        cycles = simulate_arbitrage(df, args.default_buy, args.default_sell, args.max_hold)
        for c in cycles:
            c['order_book_id'] = etf
        all_cycles.extend(cycles)
    return pd.DataFrame(all_cycles)


def find_best(df: pd.DataFrame, args) -> pd.DataFrame:
    """找每个 ETF 年化收益最高的参数组合。"""
    idx = df.groupby('order_book_id')['total_return'].idxmax()
    best = df.loc[idx].copy()
    best = best.sort_values('total_return', ascending=False).reset_index(drop=True)
    return best[[
        'order_book_id', 'buy_threshold', 'sell_threshold', 'years',
        'n_cycles', 'win_rate', 'avg_return', 'total_return',
        'annualized_return', 'max_drawdown', 'profit_loss_ratio', 'avg_hold_days',
    ]]


def write_outputs(threshold_df, forward_df, cycles_df, best_df):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    threshold_df.to_csv(OUT_DIR / 'premium_overall.csv', index=False, encoding='utf-8-sig')
    forward_df.to_csv(OUT_DIR / 'premium_forward.csv', index=False, encoding='utf-8-sig')
    cycles_df.to_csv(OUT_DIR / 'premium_cycles.csv', index=False, encoding='utf-8-sig')
    best_df.to_csv(OUT_DIR / 'top_premium_etfs.csv', index=False, encoding='utf-8-sig')
    print(f'\n输出文件 -> {OUT_DIR.resolve()}/')
    for f in ['premium_overall.csv', 'premium_forward.csv', 'premium_cycles.csv', 'top_premium_etfs.csv']:
        print(f'  {f}')


def print_summary(best_df: pd.DataFrame):
    print(f"\n{'=' * 90}")
    print("各 ETF 最优阈值（按累计收益率排序）")
    print(f"{'=' * 90}")
    cols = ['order_book_id', 'buy_threshold', 'sell_threshold', 'years', 'n_cycles',
            'win_rate', 'total_return', 'annualized_return', 'max_drawdown', 'avg_hold_days']
    display = best_df[cols].copy()
    for c in ['win_rate', 'total_return', 'annualized_return', 'max_drawdown']:
        display[c] = display[c].apply(lambda x: f"{x:.2%}" if pd.notna(x) else '-')
    display['buy_threshold'] = display['buy_threshold'].apply(lambda x: f"{x:.3f}")
    display['sell_threshold'] = display['sell_threshold'].apply(lambda x: f"{x:.3f}")
    display['years'] = display['years'].apply(lambda x: f"{x:.2f}年" if pd.notna(x) else '-')
    display['avg_hold_days'] = display['avg_hold_days'].apply(lambda x: f"{x:.0f}天" if pd.notna(x) else '-')
    print(display.to_string(index=False))


def run(etf_codes: list[str] | None = None,
        min_buy: float = -0.06, max_buy: float = 0.0,
        min_sell: float = 0.0, max_sell: float = 0.06,
        step: float = 0.005,
        default_buy: float = -0.01, default_sell: float = 0.02,
        max_hold: int = 60):
    # 1. 加载数据
    dfs = {}
    if etf_codes is None:
        # 自动发现所有缓存
        nav_files = sorted(CACHE_DIR.glob('*_nav.csv'))
        etf_codes = [f.stem.replace('_nav', '') for f in nav_files]

    print(f"加载 {len(etf_codes)} 只 ETF 数据 ...")
    for code in etf_codes:
        df = load_etf_data(code)
        if df is not None and not df.empty:
            dfs[code] = df
            pr = df['premium']
            print(f"  {code:16s} {len(df):5d}天  "
                  f"premium avg={pr.mean()*100:+.2f}%  "
                  f"min={pr.min()*100:+.2f}%  max={pr.max()*100:+.2f}%")
        else:
            print(f"  {code:16s} 无数据")

    if not dfs:
        print('[ERROR] 无数据。请先运行 download_qdii_nav.py')
        return

    args = argparse.Namespace(
        min_buy=min_buy, max_buy=max_buy,
        min_sell=min_sell, max_sell=max_sell,
        step=step, default_buy=default_buy,
        default_sell=default_sell, max_hold=max_hold,
    )

    # 2. Forward return 分析
    print(f"\n[1/3] 计算 forward return ...")
    forward_df = build_forward_table(dfs, args)
    print(f"  {len(forward_df)} 条分档记录")

    # 3. 阈值扫描（套利模拟）
    n_buy = len(np.arange(args.min_buy, args.max_buy + args.step / 2, args.step))
    n_sell = len(np.arange(args.min_sell, args.max_sell + args.step / 2, args.step))
    print(f"\n[2/3] 扫描阈值 buy=[{args.min_buy},{args.max_buy}] "
          f"sell=[{args.min_sell},{args.max_sell}] step={args.step} ...")
    print(f"  组合数: {n_buy}×{n_sell}×{len(dfs)}只ETF = {n_buy*n_sell*len(dfs)}")
    threshold_df = build_threshold_table(dfs, args)
    print(f"  {len(threshold_df)} 条阈值组合记录")

    # 4. 最优参数 + 明细
    print(f"\n[3/3] 生成报告 ...")
    best_df = find_best(threshold_df, args)
    cycles_df = build_cycles_detail(dfs, args)

    write_outputs(threshold_df, forward_df, cycles_df, best_df)
    print_summary(best_df)


def main():
    parser = argparse.ArgumentParser(description='QDII ETF 折溢价套利分析')
    parser.add_argument('--etfs', default=None, help='指定 ETF 代码，逗号分隔（默认分析所有缓存）')
    parser.add_argument('--min-buy', type=float, default=-0.06, help='买入阈值下限（折价）')
    parser.add_argument('--max-buy', type=float, default=0.0, help='买入阈值上限')
    parser.add_argument('--min-sell', type=float, default=0.0, help='卖出阈值下限（溢价）')
    parser.add_argument('--max-sell', type=float, default=0.06, help='卖出阈值上限')
    parser.add_argument('--step', type=float, default=0.005, help='阈值扫描步长')
    parser.add_argument('--default-buy', type=float, default=-0.01, help='默认买入阈值')
    parser.add_argument('--default-sell', type=float, default=0.02, help='默认卖出阈值')
    parser.add_argument('--max-hold', type=int, default=60, help='最大持仓天数')
    args = parser.parse_args()

    etf_codes = None
    if args.etfs:
        etf_codes = [s.strip() for s in args.etfs.split(',') if s.strip()]

    run(etf_codes=etf_codes,
        min_buy=args.min_buy, max_buy=args.max_buy,
        min_sell=args.min_sell, max_sell=args.max_sell,
        step=args.step,
        default_buy=args.default_buy, default_sell=args.default_sell,
        max_hold=args.max_hold)


if __name__ == '__main__':
    main()
